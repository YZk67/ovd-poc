import subprocess

import pytest

from tools.run_rare_stage_comparison import parse_args, run


@pytest.fixture
def args(tmp_path):
    for name in ("old", "new", "annotations"):
        (tmp_path / (name + ".json")).write_text("[]")
    return parse_args([
        "--old-predictions", str(tmp_path / "old.json"),
        "--new-predictions", str(tmp_path / "new.json"),
        "--annotations", str(tmp_path / "annotations.json"),
        "--expected-old-apr", "42.8843", "--expected-new-apr", "42.3031",
        "--output-dir", str(tmp_path / "output"),
    ])


def test_missing_earlier_predictions_prevents_all_work(args, tmp_path, monkeypatch):
    (tmp_path / "old.json").unlink()
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Must not start evaluation"))
    with pytest.raises(FileNotFoundError, match="NO evaluation or GPU inference started"):
        run(args)
    assert not (tmp_path / "output").exists()


def test_saved_inputs_cannot_be_overwritten_or_identical(args, tmp_path):
    args.new_predictions = args.old_predictions
    with pytest.raises(ValueError, match="different saved files"):
        run(args)
    args.new_predictions = str(tmp_path / "new.json")
    (tmp_path / "output").mkdir()
    (tmp_path / "output/new_report.json").write_text("preserve")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        run(args)
    assert (tmp_path / "output/new_report.json").read_text() == "preserve"


def test_pipeline_uses_same_python_cpu_all_curves_and_expected_apr(args, monkeypatch):
    import sys

    calls = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kw: calls.append((command, kw)))
    run(args)
    assert len(calls) == 3
    for command, kw in calls:
        assert command[:2] == [sys.executable, "-u"]
        assert kw["check"] and kw["env"]["CUDA_VISIBLE_DEVICES"] == ""
    for (command, _), apr in zip(calls[:2], ("42.8843", "42.3031")):
        assert "--all-curves" in command
        assert command[command.index("--expected-apr") + 1] == apr
    command = calls[-1][0]
    assert command[command.index("--top-declines") + 1] == "20"
    assert "--fill-missing-curves" not in command  # No redundant evaluator pass.


def test_failed_apr_check_or_evaluation_aborts_later_stages(args, monkeypatch):
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        run(args)
    assert len(calls) == 1


@pytest.mark.parametrize("field,value", [("expected_old_apr", float("nan")),
                                        ("expected_new_apr", 101), ("top_declines", 0)])
def test_invalid_options_fail_before_creating_output(args, field, value, tmp_path):
    setattr(args, field, value)
    with pytest.raises(ValueError):
        run(args)
    assert not (tmp_path / "output").exists()
