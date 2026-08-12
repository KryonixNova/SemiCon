import csv
import subprocess
import sys as _sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.localizer.geometry import SEARCH_PX

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_manifest(tmp_path, name, n):
    ref_dir = tmp_path / f"{name}_ref"
    search_dir = tmp_path / f"{name}_search"
    ref_dir.mkdir()
    search_dir.mkdir()
    manifest_path = tmp_path / f"{name}_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "reference_path", "search_path",
                                                "gt_x", "gt_y"])
        writer.writeheader()
        for i in range(n):
            rng = np.random.default_rng(hash((name, i)) % (2**31))
            ref_path = ref_dir / f"{i:05d}.png"
            search_path = search_dir / f"{i:05d}.png"
            cv2.imwrite(str(ref_path), rng.integers(0, 255, (100, 100), dtype=np.uint8))
            cv2.imwrite(str(search_path), rng.integers(0, 255, (SEARCH_PX, SEARCH_PX), dtype=np.uint8))
            writer.writerow({"id": i, "reference_path": str(ref_path),
                             "search_path": str(search_path),
                             "gt_x": 400.0 + i, "gt_y": 500.0 + i})
    return str(manifest_path)


@pytest.mark.slow
def test_finetune_on_manifest_runs_and_saves_checkpoint(tmp_path, tiny_checkpoint):
    ckpt_path, _device = tiny_checkpoint
    train_manifest = _write_manifest(tmp_path, "train", 6)
    val_manifest = _write_manifest(tmp_path, "val", 4)

    cmd = [_sys.executable, "scripts/finetune_on_manifest.py",
           "--run-name", "manifest_test", "--out-dir", str(tmp_path / "ckpts"),
           "--init-from", ckpt_path,
           "--train-manifest", train_manifest, "--val-manifest", val_manifest,
           "--max-steps", "2", "--val-every", "2", "--val-batches", "1",
           "--num-workers", "0", "--batch-size", "2"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stderr

    last_path = tmp_path / "ckpts" / "manifest_test" / "last.pt"
    assert last_path.exists()
    ckpt = torch.load(last_path, weights_only=False)
    assert ckpt["step"] == 2
    assert ckpt["data_source"] == "manifest"


@pytest.mark.slow
def test_finetune_on_manifest_best_metric_persisted_and_resets_on_change(tmp_path, tiny_checkpoint):
    ckpt_path, _device = tiny_checkpoint
    train_manifest = _write_manifest(tmp_path, "train", 6)
    val_manifest = _write_manifest(tmp_path, "val", 4)
    out_dir = tmp_path / "ckpts"

    cmd1 = [_sys.executable, "scripts/finetune_on_manifest.py",
            "--run-name", "metric_test", "--out-dir", str(out_dir),
            "--init-from", ckpt_path, "--train-manifest", train_manifest,
            "--val-manifest", val_manifest, "--max-steps", "2", "--val-every", "2",
            "--val-batches", "1", "--num-workers", "0", "--batch-size", "2",
            "--best-metric", "5"]
    r1 = subprocess.run(cmd1, capture_output=True, text=True, cwd=REPO_ROOT)
    assert r1.returncode == 0, r1.stderr
    ckpt1 = torch.load(out_dir / "metric_test" / "last.pt", weights_only=False)
    assert ckpt1["best_metric"] == 5

    cmd2 = [_sys.executable, "scripts/finetune_on_manifest.py",
            "--run-name", "metric_test", "--out-dir", str(out_dir),
            "--resume", "--train-manifest", train_manifest,
            "--val-manifest", val_manifest, "--max-steps", "4", "--val-every", "2",
            "--val-batches", "1", "--num-workers", "0", "--batch-size", "2",
            "--best-metric", "1"]
    r2 = subprocess.run(cmd2, capture_output=True, text=True, cwd=REPO_ROOT)
    assert r2.returncode == 0, r2.stderr
    assert "NOTE: best_metric changed" in r2.stdout
    ckpt2 = torch.load(out_dir / "metric_test" / "last.pt", weights_only=False)
    assert ckpt2["best_metric"] == 1


def test_finetune_on_manifest_requires_init_from_or_resume():
    cmd = [_sys.executable, "scripts/finetune_on_manifest.py",
           "--run-name", "x", "--train-manifest", "a.csv", "--val-manifest", "b.csv"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode != 0
    assert "init-from" in (result.stdout + result.stderr)


def test_finetune_on_manifest_rejects_resume_and_init_from_together():
    cmd = [_sys.executable, "scripts/finetune_on_manifest.py",
           "--run-name", "x", "--train-manifest", "a.csv", "--val-manifest", "b.csv",
           "--resume", "--init-from", "checkpoints/production_v3/best.pt"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode != 0
    assert "mutually exclusive" in (result.stdout + result.stderr)
