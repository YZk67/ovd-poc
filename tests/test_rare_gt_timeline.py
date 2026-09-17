from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from tools import trace_rare_gt_timeline as cli
from tools.rare_gt_timeline_ops import EPOCH_ITERATIONS, IOU_KEYS, stage_row, summarize_timeline
from test_rare_gt_queries import checkpoint


def analysis(epoch, *, hit=True, eligible=1, score=None):
    score = (.6 if hit else .4) if score is None else score
    best = dict(query_id=epoch, region_iou=.95, detector_probability=.2, clip_probability=.9,
                detector_rank=19, clip_rank=1, fused_score=score, score_threshold_ratio=score/.5,
                image_pair_rank_interval=[10, 10] if hit else [332, 332]) if eligible else None
    row = dict(raw_eligible_queries=eligible, retained_true_class_queries=int(hit and bool(eligible)),
               retained_any_class_pairs=int(hit and bool(eligible)), best_fused_true_class_eligible=best,
               reason="eligible_true_class_retained" if hit and eligible else "excluded")
    return dict(iteration=EPOCH_ITERATIONS[epoch], image_cutoff=.5, raw_queries=900, vocabulary_size=1203,
                best_iou_query=dict(region_iou=.95 if eligible else .4),
                by_iou={k: deepcopy(row) for k in IOU_KEYS},
                saved_predictions_check=dict(all_predictions_reproduced=True, matched=300))


def test_chronological_all_losses_and_recoveries_not_binary_search():
    stages = {e: analysis(e, hit=e in (8, 10)) for e in (12, 10, 9, 8, 11)}
    result = summarize_timeline(stages, [])
    for value in result.values():
        assert [r['epoch'] for r in value['rows']] == [8, 9, 10, 11, 12]
        assert [(t['kind'], t['from_epoch'], t['to_epoch']) for t in value['transitions']] == [
            ('loss', 8, 9), ('recovery', 9, 10), ('loss', 10, 11)]
        assert value['first_observed_loss_bracket']['update_interval_inclusive'] == [56800, 63899]
        assert value['last_observed_loss_bracket_before_final_miss']['update_interval_inclusive'] == [71000, 78099]
        assert value['has_observed_recovery']
    json.dumps(result, allow_nan=False)


def test_missing_checkpoints_widen_bracket_and_do_not_become_misses():
    stages = {8: analysis(8), 10: analysis(10), 12: analysis(12, hit=False)}
    result = summarize_timeline(stages, [9, 11])['0.50']
    assert len(result['rows']) == 3 and len(result['transitions']) == 1
    assert result['transitions'][0]['missing_epochs_inside'] == [11]
    assert result['transitions'][0]['update_interval_inclusive'] == [71000, 85199]
    bare = summarize_timeline({8: stages[8], 12: stages[12]}, [9, 10, 11])['0.50']
    assert bare['resolution'] == 'endpoints_only_not_narrowed'
    assert bare['transitions'][0]['missing_epochs_inside'] == [9, 10, 11]


def test_box_loss_selection_loss_recovery_and_boundary_are_explicit():
    stages = {8: analysis(8), 9: analysis(9, eligible=0), 10: analysis(10, hit=False),
              11: analysis(11, score=.500001), 12: analysis(12)}
    rows = summarize_timeline(stages, [])['0.50']
    assert [r['state'] for r in rows['rows']] == ['retained', 'no_eligible_box', 'excluded', 'retained', 'retained']
    assert rows['transitions'][0]['to_state'] == 'no_eligible_box'
    assert rows['transitions'][1]['boundary_sensitive']
    assert rows['last_observed_loss_bracket_before_final_miss'] is None


def test_invalid_stage_and_unaccounted_epochs_rejected():
    with pytest.raises(ValueError, match='Both authenticated'):
        summarize_timeline({8: analysis(8)}, [9, 10, 11])
    with pytest.raises(ValueError, match='Account'):
        summarize_timeline({8: analysis(8), 12: analysis(12)}, [])
    wrong = analysis(9)
    with pytest.raises(ValueError, match='iteration'):
        stage_row(8, wrong, '0.50')
    wrong = analysis(8, score=float('nan'))
    with pytest.raises(ValueError, match='Nonfinite'):
        stage_row(8, wrong, '0.50')


@pytest.mark.parametrize('epoch', [9, 10, 11])
def test_intermediate_checkpoint_validator_uses_explicit_iteration(epoch):
    iteration = EPOCH_ITERATIONS[epoch]
    cli.validate_checkpoint(checkpoint(iteration), f'{epoch}ep', expected_iteration=iteration)
    with pytest.raises(ValueError, match='expected iteration'):
        cli.validate_checkpoint(checkpoint(iteration-1), f'{epoch}ep', expected_iteration=iteration)


def setup_run(tmp_path, monkeypatch, available=(9, 10, 11)):
    gt = dict(id=cli.GT_ID, image_id=cli.IMAGE_ID, category_id=920, bbox=[0., 0., 10., 10.])
    image = dict(id=cli.IMAGE_ID, width=100, height=100)
    ann = tmp_path/'annotations.json'
    ann.write_text(json.dumps(dict(images=[image], annotations=[gt],
                                   categories=[dict(id=920, name='scarecrow')])))
    run = tmp_path/'training'
    run.mkdir()
    for epoch in available:
        torch.save(checkpoint(EPOCH_ITERATIONS[epoch]), run/f'model_{EPOCH_ITERATIONS[epoch]:07d}.pth')
    config = tmp_path/'config.py'
    config.write_text('# no model instantiation in this test')
    protocol = dict(cli.PROTOCOL, annotations_sha256=cli.file_identity(ann)['sha256'],
                    code_sha256='model-code', asset_sha256={'asset': 'hash'}, seed=42,
                    old_checkpoint=str(run/'model_0056799.pth'), old_sha256='8ep-hash',
                    new_checkpoint=str(run/'model_final.pth'), new_sha256='12ep-hash')
    endpoints = {s: analysis(e, hit=e == 8) for s, e in (('old', 8), ('new', 12))}
    for side in endpoints:
        endpoints[side]['cache'] = dict(manifest={'path': str(tmp_path/'source_cache'/'manifest.json')})
    source = dict(complete=True, protocol=protocol, gt=gt, image=image, endpoints=endpoints,
                  sources={'annotations': cli.file_identity(ann)})
    path = tmp_path/'source.json'
    path.write_text(json.dumps(source))
    monkeypatch.setattr(cli, 'endpoint_analysis', lambda s, side, c: deepcopy(s['endpoints'][side]))
    monkeypatch.setattr(cli, 'cached_stage', lambda found, e, *a: analysis(e, hit=e in (9, 11)))
    monkeypatch.setattr(cli, 'model_code_hash', lambda p: 'model-code')
    monkeypatch.setattr(cli, 'input_asset_hashes', lambda *a: {'asset': 'hash'})
    calls = []

    def capture(args, side, panel, dataset, signature, cache):
        assert side == 'old' and panel == {'image_ids': [cli.IMAGE_ID]}
        assert args.device == 'cpu' and args.max_dets == 300
        ckpt = cli.load_trusted_torch_file(args.old_checkpoint)
        calls.append(ckpt['iteration'])
        branch = cache/side
        branch.mkdir(parents=True, exist_ok=True)
        torch.save({}, branch/'bank.pt')
        torch.save({}, branch/f'{cli.IMAGE_ID}.pt')

    monkeypatch.setattr(cli, 'dump_checkpoint', capture)
    args = SimpleNamespace(source_json=str(path), output_dir=str(tmp_path/'output'), checkpoint_dir=None,
                           annotations=None, config_file=str(config), cpu_threads=1, device='cpu',
                           reuse_cache=[], cache_search_root=None, cache_only=False, prepare_only=False)
    return args, source, calls


def test_three_single_image_calls_then_retry_uses_zero_calls(tmp_path, monkeypatch):
    args, source, calls = setup_run(tmp_path, monkeypatch)
    before = {p: cli.file_identity(p) for p in tmp_path.rglob('*') if p.is_file()}
    report = cli.run(args)
    assert calls == [63899, 70999, 78099]
    assert report['complete'] and report['new_forward_calls'] == 3 and report['training_updates'] == 0
    assert report['timeline']['0.50']['has_observed_recovery']
    assert report['timeline']['0.50']['last_observed_loss_bracket_before_final_miss']['from_epoch'] == 11
    args.cache_only = True
    again = cli.run(args)
    assert again['complete'] and again['new_forward_calls'] == 0 and len(calls) == 3
    for path, identity in before.items():
        assert cli.file_identity(path) == identity


def test_missing_weights_are_skipped_without_endpoint_capture(tmp_path, monkeypatch):
    args, source, calls = setup_run(tmp_path, monkeypatch, available=(10,))
    report = cli.run(args)
    assert calls == [70999] and set(report['missing_checkpoints']) == {9, 11}
    assert len(report['timeline']['0.50']['rows']) == 3


def test_no_middle_weights_cpu_only_cannot_claim_narrowed_window(tmp_path, monkeypatch):
    args, source, calls = setup_run(tmp_path, monkeypatch, available=())
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: pytest.fail('No CUDA check for cache-only replay'))
    report = cli.run(args)
    assert report['complete'] and report['new_forward_calls'] == 0 and not calls
    assert report['timeline']['0.50']['resolution'] == 'endpoints_only_not_narrowed'


@pytest.mark.parametrize('kind', ['endpoint', 'code', 'assets', 'wrong_iteration', 'radius', 'cache_only'])
def test_preflight_failures_never_start_any_forward(tmp_path, monkeypatch, kind):
    args, source, calls = setup_run(tmp_path, monkeypatch)
    if kind == 'endpoint':
        def broken(*a):
            raise FileNotFoundError('Missing endpoint cache; cannot recapture')
        monkeypatch.setattr(cli, 'endpoint_analysis', broken)
    elif kind == 'code':
        monkeypatch.setattr(cli, 'model_code_hash', lambda p: 'different')
    elif kind == 'assets':
        monkeypatch.setattr(cli, 'input_asset_hashes', lambda *a: {})
    elif kind in ('wrong_iteration', 'radius'):
        torch.save(checkpoint(56799 if kind == 'wrong_iteration' else 78099, mode=1.5 if kind == 'radius' else 0.),
                   tmp_path/'training/model_0078099.pth')
    else:
        args.cache_only = True
    with pytest.raises((ValueError, FileNotFoundError)):
        cli.run(args)
    assert not calls and not (Path(args.output_dir)/'report.json').exists()


def test_prepare_only_records_plan_not_completed_result(tmp_path, monkeypatch):
    args, source, calls = setup_run(tmp_path, monkeypatch)
    args.prepare_only = True
    report = cli.run(args)
    assert not report['complete'] and not calls
    assert (Path(args.output_dir)/'plan.json').exists()
    assert not (Path(args.output_dir)/'report.json').exists()
    args.prepare_only = False
    assert cli.run(args)['complete'] and len(calls) == 3


def test_failed_analysis_reuses_successful_capture_without_extra_forward(tmp_path, monkeypatch):
    args, source, calls = setup_run(tmp_path, monkeypatch)
    good = cli.cached_stage
    def fail(*a):
        raise ValueError('analysis interrupted')
    monkeypatch.setattr(cli, 'cached_stage', fail)
    with pytest.raises(ValueError, match='interrupted'):
        cli.run(args)
    assert calls == [63899]
    assert not cli.load_json(Path(args.output_dir)/'report.json')['complete']
    monkeypatch.setattr(cli, 'cached_stage', good)
    assert cli.run(args)['complete'] and calls == [63899, 70999, 78099]


def test_changed_run_or_unrelated_nonempty_output_refused(tmp_path, monkeypatch):
    args, source, calls = setup_run(tmp_path, monkeypatch)
    Path(args.output_dir).mkdir()
    (Path(args.output_dir)/'important.txt').write_text('user data')
    with pytest.raises(ValueError, match='Nonempty'):
        cli.run(args)
    assert not calls
    args.output_dir = str(tmp_path/'training')
    with pytest.raises(ValueError, match='separate'):
        cli.run(args)


def test_find_cache_uses_checkpoint_hash_not_label_and_rejects_tamper(tmp_path):
    identity = {'path': '/checkpoint', 'sha256': '9ep'}
    protocol = dict(cli.PROTOCOL, annotations_sha256='ann', asset_sha256={}, code_sha256='code')
    root = tmp_path/'external'
    inputs = dict(protocol, image_ids=[cli.IMAGE_ID], old_sha256='wrong', new_sha256='9ep')
    cli.locked_manifest(root/'manifest.json', inputs)
    branch = root/'new'
    branch.mkdir()
    torch.save({}, branch/'bank.pt')
    torch.save({}, branch/f'{cli.IMAGE_ID}.pt')
    assert cli.find_stage_cache([root/'manifest.json'], protocol, identity) == (root/'manifest.json', 'new')
    assert cli.find_stage_cache([root/'manifest.json'], protocol, {**identity, 'sha256': '12ep'}) is None
    raw = cli.load_json(root/'manifest.json')
    raw['fingerprint'] = 'tampered'
    (root/'manifest.json').write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='fingerprint'):
        cli.find_stage_cache([root/'manifest.json'], protocol, identity)


def test_real_endpoint_dense_cache_is_checked_without_any_forward(tmp_path, monkeypatch):
    from test_rare_query_readout import make_source
    torch.set_num_threads(1)
    args, source = make_source(tmp_path)
    def forbidden(*a, **kw):
        pytest.fail('Authenticated endpoint replay must not call model/CUDA')
    monkeypatch.setattr(cli, 'dump_checkpoint', forbidden)
    monkeypatch.setattr(torch.cuda, 'is_available', forbidden)
    categories = {i: {'name': str(i)} for i in range(1, 1204)}
    result = cli.endpoint_analysis(source, 'old', categories)
    assert result['iteration'] == 56799 and result['raw_queries'] == 900
    assert result['saved_predictions_check']['matched'] == 300
    assert result['by_iou']['0.50']['retained_true_class_queries'] == 1
    record = source['endpoints']['new']['cache']['bank']
    with Path(record['path']).open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError, match='bytes changed'):
        cli.endpoint_analysis(source, 'new', categories)


def test_real_768d_intermediate_cache_iteration_and_radius_validation(tmp_path):
    from test_rare_query_readout import make_source
    torch.set_num_threads(1)
    args, source = make_source(tmp_path)
    endpoint = source['endpoints']['old']
    record = endpoint['cache']
    bank = cli.load_trusted_torch_file(record['bank']['path'])
    bank['iteration'] = 63899
    torch.save(bank, record['bank']['path'])
    sample = cli.load_trusted_torch_file(record['sample']['path'])
    branch = Path(record['bank']['path']).parent
    torch.save(sample, branch/f'{cli.IMAGE_ID}.pt')
    dataset = dict(categories=[dict(id=i, name=str(i), frequency='r' if i == 920 else 'f')
                               for i in range(1, 1204)])
    found = (Path(record['manifest']['path']), 'old')
    result = cli.cached_stage(found, 9, dataset, source)
    assert result['iteration'] == 63899 and result['raw_queries'] == 900
    with pytest.raises(ValueError, match='does not match'):
        cli.cached_stage(found, 10, dataset, source)
    bank['prototype_mode_strength'] = 1.5
    torch.save(bank, record['bank']['path'])
    with pytest.raises(ValueError, match='does not match'):
        cli.cached_stage(found, 9, dataset, source)


def test_cli_help_direct_script_without_tools_package_collision():
    result = subprocess.run([sys.executable, str(cli.ROOT/'tools/trace_rare_gt_timeline.py'), '--help'],
                            cwd='/tmp', capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--cache-only' in result.stdout and '--prepare-only' in result.stdout
