from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lami_dino.models import TextPrototypeAggregator
from tools import apr_projection_trial_ops as ops
from tools import run_apr_projection_trial as runner
from tools import train_apr_projection_arm as worker
from tools.decoder_aux_ablation_ops import START,HORIZON,state_digest,validate_resume
from tools.gt_normalization_trial_ops import bank_health
from test_decoder_aux_ablation import native_class
from test_gt_normalization_trial import PROMPTS,Data


def make_tpa():
    tpa = TextPrototypeAggregator(dim=5,hidden_dim=5,num_prototypes=5,dropout=.1,
        warmup_steps=0,tau=.004375,slot_prior_strength=.2,prototype_mode_strength=0)
    with torch.no_grad():
        tpa.prototype_queries.copy_(torch.eye(5))
        tpa.key_proj.weight.copy_(torch.eye(5))
        tpa.value_proj.weight.copy_(torch.eye(5)+.04)
        tpa.key_proj.bias.zero_()
        tpa.value_proj.bias.zero_()
    return tpa


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        decoder = self.transformer.decoder = nn.Module()
        decoder.layers = nn.ModuleList([nn.Linear(2,1)])
        head = nn.Module()
        head.tpa,head.tpa_train_aggregation = make_tpa(),"calibrated"
        decoder.class_embed = nn.ModuleList([head,head])
        self.novel_idx = torch.tensor([False,False,False,True,True])
        self.criterion = SimpleNamespace(weight_dict={"loss_apr":1.})

    def filter_content_info(self,data):
        return torch.randperm(3),data

    def forward(self,data):
        self.filter_content_info(data)
        tpa = self.transformer.decoder.class_embed[0].tpa
        protos,apr = tpa(PROMPTS,advance_step=getattr(self,"tpa_advance_step",True))
        y = self.transformer.decoder.layers[0](torch.stack([d["image"] for d in data]).mean(0)).sum()
        z = protos.sum()*.01
        denominator = max(sum(len(d["instances"].gt_classes) for d in data)/4,1.)
        loss = (y+z-2-torch.rand(()))**2/denominator
        return {"loss_class":loss,"loss_bbox":(y-z)**2/denominator,"loss_giou":(y+1)**2/denominator,
                "loss_class_0":loss*.2,"loss_class_dn":loss*.1,"loss_bbox_enc":loss*.15,
                "loss_apr":apr,"loss_rpsa":(y+z).square()*.01}


def experiment(tmp_path,patch,rank=0,distributed=False):
    patch.setattr(torch.cuda,"is_available",lambda:True)
    patch.setattr(torch.cuda,"is_current_stream_capturing",lambda:False)
    patch.setattr(torch.cuda.amp,"autocast",lambda **kw:nullcontext())
    torch.manual_seed(23)
    native = native_class()
    model = Model()
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    sum(p.square().sum() for p in model.parameters()).backward()
    optimizer.step()
    optimizer.zero_grad()
    kwargs = dict(amp=True,separate_tpa_grad_clip=True,tpa_conflict_projection=True,
                  clip_grad_params={"max_norm":.5,"norm_type":2},lr_scheduler_max_iter=HORIZON,gradient_accumulation_steps=2)
    initial = native(model,Data(),optimizer,grad_scaler=torch.amp.GradScaler("cpu",init_scale=1024.),**kwargs)
    state,weights = deepcopy(initial.state_dict()),deepcopy(model.state_dict())
    m = {"output_dir":str(tmp_path),"updates":2,"seed":42,"fingerprint":"test",
         "resume":validate_resume({"iteration":START-1,"trainer":state}),"model_digest":state_digest(weights)}
    def build(arm,controlled=True):
        new = Model()
        new.load_state_dict(weights)
        opt = torch.optim.AdamW(new.parameters(),lr=1e-4,weight_decay=1e-4)
        if distributed:
            new = nn.parallel.DistributedDataParallel(new,find_unused_parameters=True)
        cls = ops.make_trainer_class(native,m,arm,rank,PROMPTS) if controlled else native
        options = {**kwargs,"tpa_conflict_projection":ops.ARMS[arm]["projection"]}
        t = cls(new,Data(),opt,grad_scaler=torch.amp.GradScaler("cpu",init_scale=1.),**options)
        t.load_state_dict(deepcopy(state))
        return t
    return m,build,weights


def dirs(path):
    for arm in ops.ARMS:
        (path/arm).mkdir()


def test_barrier_off_keeps_exact_balance_gradient_and_identical_forward_geometry():
    torch.manual_seed(13)
    a,b = make_tpa(),make_tpa()
    b.load_state_dict(a.state_dict())
    b.lambda_orth_base = 0.
    a.train(); b.train()
    before = state_digest(b.state_dict())
    x = torch.randn(6,8,5)
    outputs = []
    for tpa in (a,b):
        with ops.isolated_rng(77):
            outputs.append(tpa(x,advance_step=False))
    torch.testing.assert_close(outputs[0][0],outputs[1][0],rtol=0,atol=0)
    assert state_digest(b.state_dict()) == before
    parameters = list(b.parameters())
    logits = torch.einsum("kh,cnh->ckn",b.prototype_queries,b.key_proj(x))
    logits = b._add_slot_prior(logits)
    expected = .03*b._balance_term(logits)
    actual_grad = torch.autograd.grad(outputs[1][1],parameters,allow_unused=True)
    expected_grad = torch.autograd.grad(expected,parameters,allow_unused=True)
    for p,a_grad,b_grad in zip(parameters,actual_grad,expected_grad):
        torch.testing.assert_close(torch.zeros_like(p) if a_grad is None else a_grad,
                                   torch.zeros_like(p) if b_grad is None else b_grad,rtol=0,atol=0)
    assert any(g is not None and g.abs().sum()>0 for g in expected_grad)
    assert outputs[0][1] > outputs[1][1]


def test_native_control_equivalence_and_three_arm_pairing(tmp_path,monkeypatch):
    dirs(tmp_path)
    m,build,weights = experiment(tmp_path,monkeypatch)
    trainers = {}
    for arm in ops.ARMS:
        t = trainers[arm] = build(arm)
        assert t.initial_state["model"] == state_digest(weights)
        for update in range(2):
            t.iter = START+update
            t.run_step()
        assert t.actual_updates == 2
        assert t._get_tpa().lambda_div_base == .03
        assert all(row["guard_pass"] for row in t.health_records)
        assert t.state_dict()["apr_projection_trial"]["policy"] == ops.ARMS[arm]
        t.close_streams()
    rows = {a:runner.read_rows(tmp_path/a/"pairing_rank0.jsonl") for a in ops.ARMS}
    assert all(t.initial_state == trainers["A"].initial_state for t in trainers.values())
    for arm in ("P","R"):
        for current,reference in zip(rows[arm],rows["A"]):
            ops.verify_pair(current,reference)
        assert all(u["routing"] == {"enabled":False} for u in runner.read_rows(tmp_path/arm/"updates_rank0.jsonl"))
    assert rows["R"][0]["losses"]["loss_apr"] == pytest.approx(.03*rows["R"][0]["apr_components"]["loss_balance"])
    assert rows["A"][0]["losses"]["loss_apr"] > rows["R"][0]["losses"]["loss_apr"]
    for a,t in trainers.items():
        assert all(int(s["step"]) == 3 for s in t.optimizer.state.values())
        assert any(not torch.equal(v,weights[k]) for k,v in t.model.state_dict().items() if v.is_floating_point())
    # A with the paired wrapper must match actual native training, not a
    # separate implementation of projection/backward/clipping/AdamW.
    reference = build("A",controlled=False)
    data = []
    source = iter(Data())
    for update in range(2):
        for micro in range(2):
            with ops.isolated_rng(42+update*128+micro*2):
                data.append(next(source))
    reference._data_loader_iter_obj = iter(data)
    original = reference.model.forward
    counter = [0]
    def paired(batch):
        i = counter[0]
        with ops.isolated_rng(43+(i//2)*128+(i%2)*2):
            result = original(batch)
        counter[0] += 1
        return result
    reference.model.forward = paired
    for update in range(2):
        reference.iter = START+update
        reference.run_step()
    assert state_digest(reference.model.state_dict()) == state_digest(trainers["A"].model.state_dict())
    assert state_digest(reference.optimizer.state_dict()) == state_digest(trainers["A"].optimizer.state_dict())


def test_guard_failure_stops_before_any_update(tmp_path,monkeypatch):
    dirs(tmp_path)
    _,build,_ = experiment(tmp_path,monkeypatch)
    t = build("A")
    t.iter = START
    monkeypatch.setattr(ops,"bank_health",lambda *a:{"guard_pass":False})
    with pytest.raises(ValueError,match="health guard failed"):
        t.run_step()
    assert t.actual_updates == 0
    t.close_streams()


def test_changed_balance_or_global_apr_weight_is_rejected(tmp_path,monkeypatch):
    dirs(tmp_path)
    _,build,_ = experiment(tmp_path,monkeypatch)
    t = build("A")
    t._get_tpa().lambda_div_base = 0
    with pytest.raises(ValueError,match="balance"):
        ops.configure_policy(t,"A")
    t._get_tpa().lambda_div_base = .03
    t.raw_model.criterion.weight_dict["loss_apr"] = 0
    with pytest.raises(ValueError,match="APR weight1"):
        ops.configure_policy(t,"A")
    t.close_streams()


def test_pair_requires_same_initial_features_and_unweighted_terms_but_not_apr_value():
    row = {"iteration":START,"losses":{"loss_apr":.013,"loss_class":3.},"policy":ops.ARMS["A"],
           "apr_components":{"loss_prototype_diversity":.1,"loss_balance":.1}}
    r = deepcopy(row)
    r["losses"]["loss_apr"] = .003
    r["policy"] = ops.ARMS["R"]
    ops.verify_pair(r,row)
    r["apr_components"]["loss_balance"] = .2
    with pytest.raises(ValueError,match="First-window"):
        ops.verify_pair(r,row)
    r["iteration"] = START+1
    row["iteration"] = START+1
    ops.verify_pair(r,row)


def test_locked_options_and_native_normalization():
    m = {"output_dir":"/tmp/apr-test","checkpoint":{"path":"/tmp/source8ep.pth"},"updates":2000}
    for arm in ops.ARMS:
        options = worker.training_options(m,arm)
        assert f'train.output_dir="/tmp/apr-test/{arm}"' in options
        assert f"train.tpa_conflict_projection={str(ops.ARMS[arm]['projection']).lower()}" in options
        assert "train.max_iter=58800" in options and "train.lr_scheduler_max_iter=85200" in options
        assert "train.checkpointer.period=57300" in options
        assert "model.classifier.tpa_prototype_mode_strength=0.0" in options
        assert "model.classifier.tpa_train_aggregation=calibrated" in options
        assert not any("loss_apr=" in o or "lambda_div=" in o for o in options)


def mock_pipeline(tmp_path,monkeypatch,fail_arm=None):
    source = tmp_path/"source.json"
    source.write_text("{}"); identity = runner.file_identity(source)
    output = tmp_path/"trial"
    output.mkdir()
    for arm in ops.ARMS:
        (output/arm).mkdir()
        (output/arm/"last_checkpoint").write_text("source")
    m = {"output_dir":str(output),"updates":2000,"fingerprint":"test","scope":[],"assets":[],
         **{k:identity for k in ("checkpoint","config","prompt_bank","reference_manifest","train_annotations","val_annotations")}}
    monkeypatch.setattr(runner,"read_manifest",lambda p:m)
    events = []
    def run(command,directory):
        if "--arm" in command:
            arm = command[command.index("--arm")+1]
            events.append("train_"+arm)
            if arm == fail_arm:
                raise ValueError("simulated health guard failure")
        else:
            events.append(directory.name)
    monkeypatch.setattr(runner,"run_evaluation",run)
    monkeypatch.setattr(runner,"verify_arm",lambda m,a:{"checkpoint":identity,"final_rank":{"guard_pass":True,"rare":{"mean_rank":4.2}}})
    monkeypatch.setattr(runner,"collect_evaluation",lambda d:{"metrics":{k:{"eval_A":42.,"eval_P":43.,"eval_R":41.}[d.name] for k in runner.METRICS}})
    return m,events


def test_pipeline_shares_middle_arm_and_evaluates_each_once(tmp_path,monkeypatch):
    m,events = mock_pipeline(tmp_path,monkeypatch)
    result = runner.execute(m)
    assert events == ["train_A","eval_A","train_P","eval_P","train_R","eval_R"]
    assert result["complete"]
    assert result["comparisons"]["round1_P_minus_A"]["APr"] == 1.
    assert result["comparisons"]["round2_R_minus_P"]["APr"] == -2.
    assert runner.load_json(Path(m["output_dir"])/"STATUS.json")["state"] == "COMPLETE"
    assert "round2_R_minus_P" in Path(m["output_dir"],"results.txt").read_text()


def test_failed_round2_preserves_round1_and_writes_failure_status(tmp_path,monkeypatch):
    m,events = mock_pipeline(tmp_path,monkeypatch,fail_arm="R")
    monkeypatch.setattr(runner,"prepare",lambda args:m)
    with pytest.raises(ValueError,match="health guard"):
        runner.main(["--reference-trial","unused","--output-dir",m["output_dir"]])
    summary = runner.load_json(Path(m["output_dir"])/"summary.json")
    assert not summary["complete"] and "round1_P_minus_A" in summary["comparisons"]
    assert "round2_R_minus_P" not in summary["comparisons"] and "eval_R" not in events
    status = runner.load_json(Path(m["output_dir"])/"STATUS.json")
    assert status["state"] == "FAILED" and status["phase"] == "train_R"


def test_preexisting_or_nested_output_and_unbounded_budget_rejected(tmp_path):
    reference = tmp_path/"reference"
    reference.mkdir()
    args = SimpleNamespace(reference_trial=str(reference),output_dir=str(reference/"inside"),updates=2000,num_gpus=4,cpu_threads=2)
    with pytest.raises(ValueError,match="NEW output"):
        runner.prepare(args)
    args.output_dir = str(tmp_path/"new"); args.updates = 3000
    with pytest.raises(ValueError,match="2000"):
        runner.prepare(args)


def test_manifest_checks_policy_hash_source_and_code(tmp_path,monkeypatch):
    monkeypatch.setattr(worker,"ROOT",tmp_path)
    source = tmp_path/"input.json"
    source.write_text("{}")
    code = tmp_path/"trainer.py"
    code.write_text("# verified trainer\n")
    identity = runner.file_identity(source)
    m = {"schema":"apr_projection_trial_v1","arms":deepcopy(ops.ARMS),"start":START,
         "updates":2000,"num_gpus":4,"seed":42,"lr_horizon":HORIZON,
         "rank_guard":worker.RANK_GUARD,"rank_period":worker.RANK_PERIOD,"cpu_threads":2,
         "torch_version":str(torch.__version__),"code":{"trainer.py":runner.file_identity(code)["sha256"]},
         **{k:identity for k in ("config","prompt_bank","reference_manifest")}}
    path = tmp_path/"manifest.json"
    def save():
        m["fingerprint"] = runner.fingerprint({k:v for k,v in m.items() if k != "fingerprint"})
        runner.save_json(path,m)
    save()
    assert worker.read_manifest(path) == m
    m["updates"] = 500
    runner.save_json(path,m)
    with pytest.raises(ValueError,match="fingerprint"):
        worker.read_manifest(path)
    m["updates"] = 2000
    m["arms"]["R"]["balance_weight"] = 0
    save()
    with pytest.raises(ValueError,match="protocol"):
        worker.read_manifest(path)
    m["arms"] = deepcopy(ops.ARMS)
    save()
    code.write_text("# modified trainer\n")
    with pytest.raises(ValueError,match="code changed"):
        worker.read_manifest(path)
    code.write_text("# verified trainer\n")
    source.write_text('{"changed":true}')
    with pytest.raises(ValueError,match="input changed"):
        worker.read_manifest(path)


def test_incomplete_evaluation_stops_before_next_arm(tmp_path,monkeypatch):
    m,events = mock_pipeline(tmp_path,monkeypatch)
    monkeypatch.setattr(runner,"prepare",lambda args:m)
    def collect(directory):
        raise RuntimeError("Native evaluation failed/skipped")
    monkeypatch.setattr(runner,"collect_evaluation",collect)
    with pytest.raises(RuntimeError,match="evaluation failed"):
        runner.main(["--reference-trial","unused","--output-dir",m["output_dir"]])
    assert events == ["train_A","eval_A"]
    summary = runner.load_json(Path(m["output_dir"])/"summary.json")
    assert not summary["complete"] and not summary["evaluations"]
    status = runner.load_json(Path(m["output_dir"])/"STATUS.json")
    assert status["state"] == "FAILED" and status["phase"] == "evaluate_A"


def test_changed_input_aborts_before_any_training(tmp_path,monkeypatch):
    m,events = mock_pipeline(tmp_path,monkeypatch)
    Path(m["checkpoint"]["path"]).write_text("changed")
    with pytest.raises(ValueError,match="input changed"):
        runner.execute(m)
    assert events == []


def test_apr_records_catch_nonfinite_wrong_weights_and_loss():
    policy = ops.ARMS["R"]
    terms = {"loss_prototype_diversity":.2,"loss_balance":.1,"lambda_orth":0.,"lambda_balance":.03}
    ops.check_apr_record({"loss_apr":.003},terms,policy)
    with pytest.raises(ValueError,match="coefficient changed"):
        ops.check_apr_record({"loss_apr":.003},{**terms,"lambda_balance":0.},policy)
    with pytest.raises(ValueError,match="nonfinite"):
        ops.check_apr_record({"loss_apr":.003},{**terms,"loss_prototype_diversity":float("nan")},policy)
    with pytest.raises(ValueError,match="loss differs"):
        ops.check_apr_record({"loss_apr":.023},terms,policy)


def _four_rank(rank,rendezvous,directory):
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group("gloo",init_method=Path(rendezvous).as_uri(),rank=rank,world_size=4)
    patch = pytest.MonkeyPatch()
    try:
        m,build,_ = experiment(Path(directory),patch,rank,distributed=True)
        for arm in ops.ARMS:
            t = build(arm)
            for update in range(2):
                t.iter = START+update; t.run_step()
            assert t.actual_updates == 2 and all(row["guard_pass"] for row in t.health_records)
            hashes = [None]*4
            dist.all_gather_object(hashes,state_digest(t.raw_model.state_dict()))
            assert len(set(hashes)) == 1
            t.close_streams()
            dist.barrier()
        Path(directory,f"rank{rank}.ok").write_text("A/P/R paired")
    finally:
        patch.undo(); dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(),reason="Gloo unavailable")
def test_four_rank_native_three_arm_integration(tmp_path):
    dirs(tmp_path)
    torch.multiprocessing.spawn(_four_rank,args=(str(tmp_path/"rendezvous"),str(tmp_path)),nprocs=4,join=True)
    assert all((tmp_path/f"rank{r}.ok").exists() for r in range(4))
