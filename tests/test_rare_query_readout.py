from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from lami_dino.pairing_diagnostic_ops import replay_classifier
from tools import analyze_rare_query_readout as cli
from tools.diagnose_detector_tpa_pairing import locked_manifest
from tools.rare_gt_query_ops import analyze_queries
from tools.rare_query_readout_ops import check_banks, primary_attribution, readout_grid, shapley_three


def toy():
    p = torch.tensor([[[1., 0.], [1., 0.]], [[0., 1.], [0., 1.]], [[-1., 0.], [-1., 0.]]])
    base = dict(prototypes=p, category_ids=[920, 2, 3], temperature=.07, logit_scale=4.,
                cls_bias=-1., tpa_tau=.004375, vlm_temperature=2., prompt_sha256="same",
                vlm_text=torch.eye(3), novel_mask=torch.tensor([True, False, False]))
    banks = {"old": base, "new": {**base, "prototypes": p.flip(-1), "cls_bias": -2.}}
    samples = {"old": {"features": torch.tensor([[1., .1], [.4, .8]])},
               "new": {"features": torch.tensor([[.3, .9], [.8, .2]])}}
    entry = dict(box_xyxy=[1., 1., 10., 10.], region_iou=.9,
                 clip_log_probability=-.05, image_topk_threshold=.4)
    entries = {"old": [{**entry, "query_id": 0}, {**entry, "query_id": 1}],
               "new": [{**entry, "query_id": 1, "clip_log_probability": -.06, "image_topk_threshold": .42}]}
    return samples, banks, entries


def test_readout_grid_fixed_candidates_and_bias_does_not_change_category_order():
    samples, banks, entries = toy()
    original = deepcopy(samples)
    rows = readout_grid(samples, banks, entries, 0, dict(beta=.3, novel_scale=3.))
    assert len(rows) == 12
    for side in ("old", "new"):
        assert torch.equal(samples[side]["features"], original[side]["features"])
    for row in rows:
        other_bias = next(x for x in rows if (x['feature_side'], x['query_id'], x['bank_side']) ==
                          (row['feature_side'], row['query_id'], row['bank_side']) and x['bias_side'] != row['bias_side'])
        assert row['detector_class_rank_interval'] == other_bias['detector_class_rank_interval']
        assert row['detector_true_minus_best_other'] == pytest.approx(other_bias['detector_true_minus_best_other'])
        assert row['detector_logit']-other_bias['detector_logit'] == pytest.approx(row['bias']-other_bias['bias'])
        entry = next(e for e in entries[row['feature_side']] if e['query_id'] == row['query_id'])
        assert row['box_xyxy'] == entry['box_xyxy']
        assert row['fixed_clip_log_probability'] == entry['clip_log_probability']
        assert row['frozen_native_cutoff'] == entry['image_topk_threshold']
        assert row['counterfactual_image_rank'] is None and row['counterfactual_tp'] is None
    result = primary_attribution(rows, {"old": 0, "new": 1}, .3)
    assert result['logit_contributions']['scalar_bias'] == pytest.approx(-1.)
    assert max(result['closure_errors'].values()) < 1e-10
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('factor', ['projected_query', 'terminal_bank', 'scalar_bias'])
def test_factor_only_change_assigned_only_to_that_factor(factor):
    samples, banks, entries = toy()
    samples['new']['features'][1] = samples['old']['features'][0].clone()
    banks['new'] = deepcopy(banks['old'])
    if factor == 'projected_query':
        samples['new']['features'][1] = torch.tensor([.2, .9])
    elif factor == 'terminal_bank':
        banks['new']['prototypes'] = banks['old']['prototypes'].flip(-1)
    else:
        banks['new']['cls_bias'] += .5
    rows = readout_grid(samples, banks, entries, 0, dict(beta=.3, novel_scale=3.))
    result = primary_attribution(rows, {"old": 0, "new": 1}, .3)
    for key, value in result['logit_contributions'].items():
        assert (abs(value) > 1e-4) if key == factor else abs(value) < 1e-10
    for key, value in result['weighted_log_detector_contributions'].items():
        if key != factor:
            assert abs(value) < 1e-10


def test_shapley_known_interaction_and_exact_closure():
    values = {k: 2*k[0]+3*k[1]+5*k[2]+4*k[0]*k[1] for k in product((0, 1), repeat=3)}
    terms, error = shapley_three(values)
    assert terms == pytest.approx(dict(projected_query=4., terminal_bank=5., scalar_bias=5.))
    assert error < 1e-10
    with pytest.raises(ValueError):
        shapley_three({**values, (0, 0, 0): float('nan')})


@pytest.mark.parametrize('key,value', [('temperature', .08), ('logit_scale', 5.), ('prompt_sha256', 'changed'),
                                    ('vlm_text', torch.zeros(3, 3)), ('cls_bias', float('nan'))])
def test_do_not_conflate_other_settings_with_query_bank_or_bias(key, value):
    _, banks, _ = toy()
    banks['new'][key] = value
    with pytest.raises(ValueError):
        check_banks(banks)


def make_source(tmp_path):
    """Real 900x1203x768 cache schema, evaluated cheaply using two unique vectors."""
    gt = dict(id=137708, image_id=218917, category_id=920, bbox=[0., 0., 10., 10.])
    image = dict(id=218917, width=100, height=100)
    ids, c = list(range(1, 1204)), 919
    protocol = dict(alpha=0., beta=.3, novel_scale=3., tpa_tau=.004375, cls_tau=.07, max_dets=300,
                    annotations_sha256='ann', asset_sha256={'text': 'same'}, code_sha256='capture',
                    old_sha256='checkpoint8', new_sha256='checkpoint12')
    cache = tmp_path/'cache'
    signature = locked_manifest(cache/'manifest.json', {**protocol, 'image_ids': [218917]})
    report = dict(complete=True, fingerprint='source-fingerprint', protocol=protocol, gt=gt, image=image, endpoints={})
    for side, iteration, primary in (('old', 56799, 3), ('new', 85199, 7)):
        p = torch.zeros(1203, 5, 768)
        p[c, :, 0] = 1.
        if side == 'new':
            p[c, :, 1] = .1
        text = torch.zeros(1203, 768)
        text[c, 0] = 1.
        bank = dict(fingerprint=signature, label=side, iteration=iteration, category_ids=ids,
                    prototypes=p, vlm_text=text, temperature=.07, logit_scale=4., cls_bias=-2.,
                    vlm_temperature=10., novel_mask=torch.tensor([i == 920 for i in ids]),
                    tpa_tau=.004375, slot_prior_strength=.2, prototype_mode_strength=0., prompt_sha256='same')
        unique = torch.zeros(2, 768)
        unique[:, 0] = torch.tensor([-1., 1.])
        if side == 'new':
            unique[1, 1] = .2
        which = torch.zeros(900, dtype=torch.long)
        which[primary] = 1
        features = unique[which]
        roi = torch.zeros(900, 768)
        roi[:, 0] = 1.
        boxes = torch.tensor([[30., 30., 40., 40.]]).repeat(900, 1)
        boxes[primary] = torch.tensor([0., 0., 10., 10.])
        logits = replay_classifier(unique, p, temperature=.07, logit_scale=4., cls_bias=-2.)[which]
        clips = (roi[:1] @ text.t()*10).repeat(900, 1)
        logdet, logclip = F.logsigmoid(logits), F.log_softmax(clips, -1)
        logscore = logdet.clone()
        logscore[:, c] = .7*logdet[:, c]+.3*logclip[:, c]+torch.log(torch.tensor(3.))
        score = logscore.exp()
        mask = torch.zeros_like(score, dtype=torch.bool)
        mask.view(-1)[score.flatten().topk(300).indices] = True
        cutoff = float(score[mask].min())
        sample = dict(fingerprint=signature, label=side, image_id=218917, width=100, height=100,
                      features=features, roi_features=roi, query_boxes=boxes,
                      native_replay_check=dict(logit_max_abs_error=0., score_max_abs_error=0.))
        dense = dict(fingerprint=report['fingerprint'], image_id=218917, iteration=iteration, category_ids=ids,
                     boxes_xyxy=boxes, detector_logits=logits, clip_logits=clips, fused_scores=score, selected=mask)
        branch = cache/side
        branch.mkdir()
        for name, value in (('bank', bank), ('sample', sample), ('dense', dense)):
            torch.save(value, branch/(name+'.pt'))
        replay = dict(boxes=boxes, det_logits=logits, clip_logits=clips, det_logp=logdet, clip_logp=logclip,
                      scores=score, log_scores=logscore, selected=mask, cutoff=cutoff, category_ids=ids)
        endpoint = analyze_queries(replay, gt, {i: {'name': str(i)} for i in ids})
        endpoint.update(iteration=iteration, saved_predictions_check=dict(all_predictions_reproduced=True, matched=300),
            cache={**{name: cli.file_identity(branch/(name+'.pt')) for name in ('bank', 'sample')},
                   'manifest': cli.file_identity(cache/'manifest.json'), 'source_label': side},
            dense_scores=cli.file_identity(branch/'dense.pt'))
        report['endpoints'][side] = endpoint
    path = tmp_path/'source.json'
    path.write_text(json.dumps(report))
    return SimpleNamespace(source_json=str(path), output=str(tmp_path/'result/report.json'), cpu_threads=1), report


def test_real_cache_cpu_pipeline_no_cuda_no_forward_no_source_changes(tmp_path, monkeypatch):
    args, source = make_source(tmp_path)
    def no_cuda(*args, **kwargs):
        pytest.fail('CPU diagnostic must not use CUDA')
    monkeypatch.setattr(torch.cuda, 'is_available', no_cuda)
    before = {p: cli.file_identity(p) for p in tmp_path.rglob('*') if p.is_file()}
    result = cli.run(args)
    assert result['complete'] and result['new_forward_calls'] == result['training_updates'] == 0
    assert result['primary_attribution']['primary_query_ids'] == {'old': 3, 'new': 7}
    assert len(result['all_candidate_readouts']) == 8
    assert max(result['primary_attribution']['closure_errors'].values()) < 1e-8
    assert {p: cli.file_identity(p) for p in before} == before


def test_modified_or_missing_cache_stops_without_report(tmp_path):
    args, source = make_source(tmp_path)
    path = Path(source['endpoints']['old']['cache']['sample']['path'])
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='bytes changed'):
        cli.run(args)
    assert not Path(args.output).exists()
    path.unlink()
    with pytest.raises(FileNotFoundError):
        cli.run(args)
    assert not Path(args.output).exists()


def test_native_full_vocabulary_check_detects_non_target_logit_drift(tmp_path):
    args, source = make_source(tmp_path)
    path = Path(source['endpoints']['old']['dense_scores']['path'])
    dense = cli.load_trusted_torch_file(path)
    primary = source['endpoints']['old']['by_iou']['0.50']['best_fused_true_class_eligible']['query_id']
    dense['detector_logits'][primary, 0] += .05  # Not the true class (index 919).
    torch.save(dense, path)
    source['endpoints']['old']['dense_scores'] = cli.file_identity(path)
    Path(args.source_json).write_text(json.dumps(source))
    with pytest.raises(ValueError, match='C-way detector replay failed'):
        cli.run(args)
    assert not Path(args.output).exists()


def test_source_requires_complete_verified_stages(tmp_path):
    _, source = make_source(tmp_path)
    cli.validate_source(source)
    source['endpoints']['new']['iteration'] = 70999
    with pytest.raises(ValueError, match='endpoints'):
        cli.validate_source(source)
    source['complete'] = False
    with pytest.raises(ValueError, match='completed'):
        cli.validate_source(source)


def test_cli_help_without_detectron_or_lvis():
    proc = subprocess.run([sys.executable, cli.__file__, '--help'], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert '--source-json' in proc.stdout and '--device' not in proc.stdout
