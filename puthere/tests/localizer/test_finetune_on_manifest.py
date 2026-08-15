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


def test_finetune_on_manifest_rejects_geometric_profile_without_noise_profile():
    cmd = [_sys.executable, "scripts/finetune_on_manifest.py",
           "--run-name", "x", "--train-manifest", "a.csv", "--val-manifest", "b.csv",
           "--init-from", "checkpoints/production_v3/best.pt",
           "--rehearsal-geometric-profile", "drift"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode != 0
    assert "rehearsal-noise-profile" in (result.stdout + result.stderr)


@pytest.mark.slow
def test_finetune_on_manifest_rehearsal_runs_and_saves_synthetic_metrics(tmp_path, tiny_checkpoint):
    ckpt_path, _device = tiny_checkpoint
    train_manifest = _write_manifest(tmp_path, "train", 6)
    val_manifest = _write_manifest(tmp_path, "val", 4)

    cmd = [_sys.executable, "scripts/finetune_on_manifest.py",
           "--run-name", "rehearsal_test", "--out-dir", str(tmp_path / "ckpts"),
           "--init-from", ckpt_path,
           "--train-manifest", train_manifest, "--val-manifest", val_manifest,
           "--rehearsal-noise-profile", "normal", "--rehearsal-geometric-profile", "normal",
           "--rehearsal-ratio", "0.5", "--rehearsal-floor-mean-px", "100000.0",
           "--max-steps", "2", "--val-every", "2", "--val-batches", "1",
           "--num-workers", "0", "--batch-size", "2"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stderr

    last_path = tmp_path / "ckpts" / "rehearsal_test" / "last.pt"
    assert last_path.exists()
    ckpt = torch.load(last_path, weights_only=False)
    assert ckpt["step"] == 2
    assert ckpt["rehearsal_metrics"] is not None
    assert "mean_error_px" in ckpt["rehearsal_metrics"]


@pytest.mark.slow
def test_finetune_on_manifest_rehearsal_floor_blocks_best_save(tmp_path, tiny_checkpoint):
    ckpt_path, _device = tiny_checkpoint
    train_manifest = _write_manifest(tmp_path, "train", 6)
    val_manifest = _write_manifest(tmp_path, "val", 4)

    cmd = [_sys.executable, "scripts/finetune_on_manifest.py",
           "--run-name", "rehearsal_floor_test", "--out-dir", str(tmp_path / "ckpts"),
           "--init-from", ckpt_path,
           "--train-manifest", train_manifest, "--val-manifest", val_manifest,
           "--rehearsal-noise-profile", "normal", "--rehearsal-geometric-profile", "normal",
           "--rehearsal-ratio", "0.5", "--rehearsal-floor-mean-px", "0.0",
           "--max-steps", "2", "--val-every", "2", "--val-batches", "1",
           "--num-workers", "0", "--batch-size", "2"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stderr

    best_path = tmp_path / "ckpts" / "rehearsal_floor_test" / "best.pt"
    # An impossible floor (0.0px) must never be cleared by a real model,
    # so best.pt is never created even though last.pt always saves.
    assert not best_path.exists()


# Unit tests for _cycle, train_batches, and should_save_best (pure functions)
import random

_sys.path.insert(0, str(REPO_ROOT))
from scripts.finetune_on_manifest import train_batches, should_save_best, _cycle


def test_cycle_repeats_the_iterable_forever():
    it = _cycle([1, 2, 3])
    drawn = [next(it) for _ in range(7)]
    assert drawn == [1, 2, 3, 1, 2, 3, 1]


def test_train_batches_ratio_zero_only_draws_manifest():
    gen = train_batches(["M"], synthetic_loader=["S"], rehearsal_ratio=0.0,
                        rng=random.Random(0))
    drawn = [next(gen) for _ in range(10)]
    assert drawn == ["M"] * 10


def test_train_batches_ratio_one_only_draws_synthetic():
    gen = train_batches(["M"], synthetic_loader=["S"], rehearsal_ratio=1.0,
                        rng=random.Random(0))
    drawn = [next(gen) for _ in range(10)]
    assert drawn == ["S"] * 10


def test_train_batches_no_synthetic_loader_only_draws_manifest():
    gen = train_batches(["M"], synthetic_loader=None, rehearsal_ratio=0.9,
                        rng=random.Random(0))
    drawn = [next(gen) for _ in range(10)]
    assert drawn == ["M"] * 10


def test_train_batches_mixed_ratio_draws_both_sources_close_to_ratio():
    gen = train_batches(["M"], synthetic_loader=["S"], rehearsal_ratio=0.5,
                        rng=random.Random(42))
    drawn = [next(gen) for _ in range(2000)]
    synthetic_frac = drawn.count("S") / len(drawn)
    assert 0.45 < synthetic_frac < 0.55


def test_should_save_best_no_rehearsal_only_checks_accuracy():
    assert should_save_best(candidate_acc=0.9, best_acc=0.8,
                            synthetic_mean_px=None, rehearsal_floor_mean_px=8.0) is True
    assert should_save_best(candidate_acc=0.7, best_acc=0.8,
                            synthetic_mean_px=None, rehearsal_floor_mean_px=8.0) is False


def test_should_save_best_blocks_when_synthetic_exceeds_floor():
    assert should_save_best(candidate_acc=0.99, best_acc=0.0,
                            synthetic_mean_px=41.06, rehearsal_floor_mean_px=8.0) is False


def test_should_save_best_allows_when_synthetic_clears_floor():
    assert should_save_best(candidate_acc=0.99, best_acc=0.0,
                            synthetic_mean_px=4.5, rehearsal_floor_mean_px=8.0) is True
