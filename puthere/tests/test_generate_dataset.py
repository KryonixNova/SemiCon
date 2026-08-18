import csv
import subprocess
import sys
from pathlib import Path

import cv2
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_generate_dataset_writes_pngs_and_manifest(tmp_path):
    out_dir = tmp_path / "out"
    cmd = [sys.executable, "generate_dataset.py", "--num-samples", "3",
           "--split", "test", "--output-dir", str(out_dir),
           "--geometric-profile", "drift"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stderr

    manifest_path = out_dir / "manifest.csv"
    assert manifest_path.exists()
    rows = list(csv.DictReader(open(manifest_path)))
    assert len(rows) == 3
    expected_cols = {"id", "architecture", "reference_path", "search_path", "gt_x", "gt_y",
                     "canvas_seed", "crop_index", "jitter_profile",
                     "imaging_noise_profile", "geometric_profile",
                     "scale_ratio", "rotation_deg"}
    assert set(rows[0]) == expected_cols
    for row in rows:
        assert row["architecture"] == "dram"
        assert row["geometric_profile"] == "drift"
        ref_img = cv2.imread(row["reference_path"], cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(row["search_path"], cv2.IMREAD_GRAYSCALE)
        # 1000x1000, matching the spec's "100x close-up view" reference
        # format -- upscaled from the model's native 100x100 representation
        # (see generate_dataset.py's REFERENCE_HIRES_PX comment).
        assert ref_img.shape == (1000, 1000)
        assert search_img.shape == (1000, 1000)


def test_generate_dataset_rejects_seed_outside_split_range(tmp_path):
    out_dir = tmp_path / "out"
    cmd = [sys.executable, "generate_dataset.py", "--num-samples", "1",
           "--split", "test", "--seed", "0", "--output-dir", str(out_dir)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode != 0
    assert "outside" in (result.stdout + result.stderr)
