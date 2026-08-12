import csv
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_pair(tmp_path, name):
    rng = np.random.default_rng(hash(name) % (2**31))
    ref_path = tmp_path / f"{name}_ref.png"
    search_path = tmp_path / f"{name}_search.png"
    cv2.imwrite(str(ref_path), rng.integers(0, 255, (100, 100), dtype=np.uint8))
    cv2.imwrite(str(search_path), rng.integers(0, 255, (1000, 1000), dtype=np.uint8))
    return str(ref_path), str(search_path)


@pytest.mark.slow
def test_localize_single_pair_matches_predict_py_format(tmp_path, tiny_checkpoint):
    ckpt_path, _device = tiny_checkpoint
    ref_path, search_path = _write_pair(tmp_path, "single")
    cmd = [sys.executable, "localize.py", "--checkpoint", ckpt_path,
           "--reference", ref_path, "--search", search_path]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stderr
    parts = result.stdout.strip().split(",")
    assert len(parts) == 3
    float(parts[0]); float(parts[1]); float(parts[2])  # all parse as numbers


@pytest.mark.slow
def test_localize_xy_only_prints_bare_x_y(tmp_path, tiny_checkpoint):
    ckpt_path, _device = tiny_checkpoint
    ref_path, search_path = _write_pair(tmp_path, "xy_only")
    cmd = [sys.executable, "localize.py", "--checkpoint", ckpt_path,
           "--reference", ref_path, "--search", search_path, "--xy-only"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stderr
    parts = result.stdout.strip().split(",")
    assert len(parts) == 2, f"expected exactly 'x,y', got {result.stdout!r}"
    float(parts[0]); float(parts[1])


@pytest.mark.slow
def test_localize_batch_mode_writes_predictions_csv(tmp_path, tiny_checkpoint):
    ckpt_path, _device = tiny_checkpoint
    ref1, search1 = _write_pair(tmp_path, "a")
    ref2, search2 = _write_pair(tmp_path, "b")

    manifest_path = tmp_path / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "reference_path", "search_path"])
        writer.writeheader()
        writer.writerow({"id": 0, "reference_path": ref1, "search_path": search1})
        writer.writerow({"id": 1, "reference_path": ref2, "search_path": search2})

    out_path = tmp_path / "predictions.csv"
    cmd = [sys.executable, "localize.py", "--checkpoint", ckpt_path,
           "--manifest", str(manifest_path), "--output", str(out_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stderr

    rows = list(csv.DictReader(open(out_path)))
    assert len(rows) == 2
    assert set(rows[0]) == {"id", "reference_path", "search_path", "predicted_x",
                            "predicted_y", "confidence", "runtime_ms"}
    for row in rows:
        float(row["predicted_x"]); float(row["confidence"]); float(row["runtime_ms"])


@pytest.mark.slow
def test_localize_batch_mode_passes_through_ground_truth_and_metadata_columns(tmp_path, tiny_checkpoint):
    """Manifest columns beyond id/reference_path/search_path -- ground truth,
    generation metadata -- must survive into the output CSV alongside the
    new prediction columns, so predictions.csv is self-contained rather than
    needing a separate join back to the source manifest."""
    ckpt_path, _device = tiny_checkpoint
    ref1, search1 = _write_pair(tmp_path, "gt_a")

    manifest_path = tmp_path / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "reference_path", "search_path",
                                                "gt_x", "gt_y", "imaging_noise_profile"])
        writer.writeheader()
        writer.writerow({"id": 0, "reference_path": ref1, "search_path": search1,
                         "gt_x": 512.0, "gt_y": 488.0, "imaging_noise_profile": "harsh"})

    out_path = tmp_path / "predictions.csv"
    cmd = [sys.executable, "localize.py", "--checkpoint", ckpt_path,
           "--manifest", str(manifest_path), "--output", str(out_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stderr

    rows = list(csv.DictReader(open(out_path)))
    assert len(rows) == 1
    assert rows[0]["gt_x"] == "512.0"
    assert rows[0]["gt_y"] == "488.0"
    assert rows[0]["imaging_noise_profile"] == "harsh"
    assert "predicted_x" in rows[0]


def test_localize_requires_reference_and_search_together():
    cmd = [sys.executable, "localize.py", "--checkpoint", "x.pt", "--reference", "r.png"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode != 0
    assert "together" in (result.stdout + result.stderr)


def test_localize_requires_some_mode():
    cmd = [sys.executable, "localize.py", "--checkpoint", "x.pt"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode != 0


def test_localize_checkpoint_defaults_to_production_v3(monkeypatch):
    sys.path.insert(0, str(REPO_ROOT))
    import localize

    monkeypatch.setattr(sys, "argv", ["localize.py", "--reference", "r.png",
                                       "--search", "s.png"])
    args = localize.parse_args()
    assert args.checkpoint == str(REPO_ROOT / "checkpoints" / "production_v3" / "best.pt")
