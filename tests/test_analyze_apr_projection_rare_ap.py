import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools.analyze_apr_projection_rare_ap import POLICIES, parse_args, run
from tools.compare_rare_pr_reports import file_identity, load_json, save_json
from test_rare_ap_concentration import reports


@pytest.fixture
def trial(tmp_path):
    source = tmp_path/"trial"
    source.mkdir()
    for name in ("A", "P", "annotations"):
        (source/(name+".json")).write_text("[]")
    manifest = {"schema": "apr_projection_extension_v1", "arms": POLICIES,
                "total_updates": 4000, "val_annotations": file_identity(source/"annotations.json")}
    manifest["fingerprint"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    a, p = reports()
    summary = {"complete": True, "manifest_fingerprint": manifest["fingerprint"], "arms": POLICIES,
               "total_updates_per_arm": 4000, "evaluations": {
                   arm: {"metrics": {"APr": r["official_apr"]}, "predictions": file_identity(source/(arm+".json"))}
                   for arm, r in zip(("A", "P"), (a, p))},
               "delta_P_minus_A": {"APr": p["official_apr"]-a["official_apr"]}}
    save_json(source/"manifest.json", manifest)
    save_json(source/"summary.json", summary)
    return parse_args(["--trial-dir", str(source), "--output-dir", str(tmp_path/"analysis")])


def fake_evaluator(calls):
    def evaluate(command, **kwargs):
        calls.append((command, kwargs))
        def arg(name):
            return command[command.index(name)+1]
        arm = Path(arg("--predictions")).stem
        r = reports()[0 if arm == "A" else 1]
        r.update(predictions=arg("--predictions"), prediction_bytes=Path(arg("--predictions")).stat().st_size,
                 annotations=arg("--annotations"))
        save_json(arg("--output"), r)
    return evaluate


def test_cpu_official_pipeline_preserves_sources_and_supports_authenticated_resume(trial, monkeypatch):
    original = {p: p.read_bytes() for p in Path(trial.trial_dir).iterdir()}
    calls = []
    monkeypatch.setattr(subprocess, "run", fake_evaluator(calls))
    result = run(trial)
    assert result["complete"] and len(calls) == 2
    assert result["delta_apr"] == pytest.approx(10/9)
    for (cmd, kw), arm in zip(calls, ("A", "P")):
        assert cmd[:2] == [sys.executable, "-u"]
        assert Path(cmd[2]).name == "report_lvis_rare_pr.py"
        assert kw["env"]["CUDA_VISIBLE_DEVICES"] == "" and kw["check"]
        assert cmd[cmd.index("--predictions")+1] == str(Path(trial.trial_dir)/(arm+".json"))
        assert cmd[cmd.index("--apr-tolerance")+1] == "0.0002"
        assert not any(x in " ".join(cmd) for x in ("--eval-only", "checkpoint", "train_net", "--dump-dir"))
    output = Path(trial.output_dir)
    assert load_json(output/"STATUS.json")["status"] == "COMPLETE"
    assert len((output/"per_class.csv").read_text().splitlines()) == 10
    trial.resume = True
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Should reuse both verified reports"))
    assert run(trial) == result
    for path, content in original.items():
        assert path.read_bytes() == content


@pytest.mark.parametrize("mutation", ["missing_P", "wrong_P_hash", "incomplete", "wrong_stage", "wrong_fingerprint", "wrong_policy", "wrong_delta", "bad_apr", "annotation_hash"])
def test_invalid_source_fails_before_any_cpu_evaluation(trial, monkeypatch, mutation):
    source = Path(trial.trial_dir)
    summary = load_json(source/"summary.json")
    if mutation == "missing_P":
        (source/"P.json").unlink()
    elif mutation == "wrong_P_hash":
        (source/"P.json").write_text("[0]")
    elif mutation == "annotation_hash":
        (source/"annotations.json").write_text("changed")
    elif mutation == "incomplete":
        summary["complete"] = False
    elif mutation == "wrong_stage":
        summary["total_updates_per_arm"] = 2000
    elif mutation == "wrong_fingerprint":
        summary["manifest_fingerprint"] = "bad"
    elif mutation == "wrong_policy":
        summary["arms"]["P"]["barrier_weight"] = 0
    elif mutation == "wrong_delta":
        summary["delta_P_minus_A"]["APr"] = .1
    else:
        summary["evaluations"]["A"]["metrics"]["APr"] = 101
    save_json(source/"summary.json", summary)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Must not evaluate"))
    with pytest.raises((ValueError, FileNotFoundError)):
        run(trial)
    assert not Path(trial.output_dir).exists()


def test_child_failure_then_resume_reuses_A_only(trial, monkeypatch):
    calls = []
    evaluate = fake_evaluator(calls)
    def fail_p(command, **kw):
        if Path(command[command.index("--predictions")+1]).stem == "P":
            raise subprocess.CalledProcessError(1, command)
        evaluate(command, **kw)
    monkeypatch.setattr(subprocess, "run", fail_p)
    with pytest.raises(subprocess.CalledProcessError):
        run(trial)
    output = Path(trial.output_dir)
    assert load_json(output/"STATUS.json")["status"] == "FAILED"
    assert not (output/"report.json").exists()
    trial.resume = True
    calls.clear()
    monkeypatch.setattr(subprocess, "run", evaluate)
    run(trial)
    assert len(calls) == 1 and Path(calls[0][0][calls[0][0].index("--predictions")+1]).stem == "P"


def test_tampered_cached_report_is_not_silently_reused(trial, monkeypatch):
    monkeypatch.setattr(subprocess, "run", fake_evaluator([]))
    run(trial)
    trial.resume = True
    with (Path(trial.output_dir)/"A_report.json").open("a") as stream:
        stream.write(" ")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Must not evaluate"))
    with pytest.raises(ValueError, match="Cached A"):
        run(trial)
    assert load_json(Path(trial.output_dir)/"STATUS.json")["status"] == "FAILED"


def test_no_overwrite_and_no_source_output_overlap(trial, monkeypatch):
    output = Path(trial.output_dir)
    output.mkdir()
    (output/"report.json").write_text("user file")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        run(trial)
    assert (output/"report.json").read_text() == "user file"
    trial.output_dir = trial.trial_dir
    with pytest.raises(ValueError, match="separate"):
        run(trial)


def test_source_changed_during_eval_never_creates_final_report(trial, monkeypatch):
    evaluate = fake_evaluator([])
    def mutate(cmd, **kw):
        evaluate(cmd, **kw)
        if Path(cmd[cmd.index("--predictions")+1]).stem == "P":
            Path(trial.trial_dir, "annotations.json").write_text("changed during evaluation")
    monkeypatch.setattr(subprocess, "run", mutate)
    with pytest.raises(ValueError, match="changed during analysis"):
        run(trial)
    assert not Path(trial.output_dir, "report.json").exists()


def test_direct_script_help_does_not_require_torch_or_lvis():
    script = Path(__file__).resolve().parents[1]/"tools/analyze_apr_projection_rare_ap.py"
    result = subprocess.run([sys.executable, "-S", str(script), "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--trial-dir" in result.stdout and "--resume" in result.stdout
