import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_validation_report_produces_required_outputs(tmp_path, tiny_checkpoint):
    ckpt_path, _device = tiny_checkpoint
    out_dir = tmp_path / "results"
    cmd = [sys.executable, "scripts/validation_report.py",
           "--checkpoint", ckpt_path, "--n-per-condition", "3",
           "--out-dir", str(out_dir)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stderr

    report = json.load(open(out_dir / "validation_report.json"))
    assert set(report["conditions"]) == {
        "noise=normal_geom=normal", "noise=harsh_geom=normal",
        "noise=normal_geom=drift", "noise=harsh_geom=drift",
    }
    for cond in report["conditions"].values():
        assert cond["n"] == 3
        for t in ("5.0", "4.0", "2.0", "1.0"):
            assert f"pass_rate@{t}px" in cond
        assert "mean_error_px" in cond and "median_error_px" in cond
        assert "worst_error_px" in cond
        assert "runtime_ms_mean" in cond and "runtime_ms_median" in cond

    assert "hardware" in report
    assert "python_version" in report
    assert "timing_method" in report
    assert report["failure_case"]["root_cause"]

    assert (out_dir / "validation_report.md").exists()
    assert (out_dir / "failure_case.png").stat().st_size > 0
