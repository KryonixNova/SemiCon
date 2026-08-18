"""Tests for generate_dataset.py's dataset-split CLI."""
import csv
import sys

import pytest

import dram_localization_pipeline as m
import generate_dataset


def test_generate_dataset_cli_writes_pngs_and_manifest(tmp_path, monkeypatch):
    out_dir = tmp_path / "output"
    argv = ["generate_dataset.py",
            "--num-samples", "2", "--split", "train",
            "--output-dir", str(out_dir), "--seed", "7"]
    monkeypatch.setattr(sys, "argv", argv)
    generate_dataset.main()

    split_dir = out_dir / "train"
    assert (split_dir / "reference" / "00000.png").exists()
    assert (split_dir / "reference" / "00001.png").exists()
    assert (split_dir / "search" / "00000.png").exists()
    assert (split_dir / "search" / "00001.png").exists()

    with open(split_dir / "manifest.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    expected_cols = {
        "id", "reference_path", "search_path", "gt_x", "gt_y",
        "gt_box_x", "gt_box_y", "gt_box_w", "gt_box_h", "preset",
        "zoom_ratio", "die_block_rows", "die_block_cols", "boundary_bias", "seed",
    }
    for row in rows:
        assert expected_cols <= set(row.keys())
        assert float(row["zoom_ratio"]) == 10.0
        assert row["preset"] in m.DRAM_PRESETS


def test_generate_dataset_cli_multiblock_uses_block_presets(tmp_path, monkeypatch):
    out_dir = tmp_path / "output"
    argv = ["generate_dataset.py",
            "--num-samples", "1", "--split", "train",
            "--output-dir", str(out_dir), "--seed", "3",
            "--presets", "dram_1x", "dram_dense",
            "--die-block-rows", "2", "--die-block-cols", "2",
            "--boundary-bias", "1.0"]
    monkeypatch.setattr(sys, "argv", argv)
    generate_dataset.main()

    with open(out_dir / "train" / "manifest.csv") as f:
        row = next(csv.DictReader(f))
    assert set(row["preset"].split("+")) <= {"dram_1x", "dram_dense"}
    assert row["die_block_rows"] == "2"
    assert row["die_block_cols"] == "2"
    assert row["boundary_bias"] == "1.0"


def test_reference_search_bbox_px_matches_true_center(tmp_path):
    sample = m.generate_sample(seed=1, tmp_dir=str(tmp_path), zoom_ratio=10)
    x0, y0, w, h = generate_dataset._reference_search_bbox_px(sample)
    cx, cy = sample.true_center_px
    assert x0 + w / 2 == pytest.approx(cx)
    assert y0 + h / 2 == pytest.approx(cy)
