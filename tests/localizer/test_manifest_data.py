import csv

import cv2
import numpy as np
import torch

from src.localizer.geometry import REF_DS_PX, SEARCH_PX
from src.localizer.manifest_data import ManifestDataset


def _write_manifest(tmp_path, n):
    ref_dir = tmp_path / "reference"
    search_dir = tmp_path / "search"
    ref_dir.mkdir()
    search_dir.mkdir()

    manifest_path = tmp_path / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "reference_path", "search_path",
                                                "gt_x", "gt_y"])
        writer.writeheader()
        for i in range(n):
            rng = np.random.default_rng(i)
            # A real AFB reference is rendered at REFERENCE_ZOOM_RATIO=10x,
            # i.e. 1000x1000 -- not already at the 100x100 training scale.
            ref_path = ref_dir / f"{i:05d}.png"
            search_path = search_dir / f"{i:05d}.png"
            cv2.imwrite(str(ref_path), rng.integers(0, 255, (1000, 1000), dtype=np.uint8))
            cv2.imwrite(str(search_path), rng.integers(0, 255, (SEARCH_PX, SEARCH_PX), dtype=np.uint8))
            writer.writerow({"id": i, "reference_path": str(ref_path),
                             "search_path": str(search_path),
                             "gt_x": 400.0 + i, "gt_y": 500.0 + i})
    return str(manifest_path)


def test_manifest_dataset_length_matches_manifest_row_count(tmp_path):
    manifest_path = _write_manifest(tmp_path, 5)
    ds = ManifestDataset(manifest_path)
    assert len(ds) == 5


def test_manifest_dataset_item_shapes_and_dtypes(tmp_path):
    manifest_path = _write_manifest(tmp_path, 3)
    ds = ManifestDataset(manifest_path)
    item = ds[0]

    assert item["reference_img"].shape == (1, REF_DS_PX, REF_DS_PX)
    assert item["search_img"].shape == (1, SEARCH_PX, SEARCH_PX)
    assert item["reference_img"].dtype == torch.float32
    assert item["search_img"].dtype == torch.float32
    assert torch.is_tensor(item["gt_x"]) and item["gt_x"].shape == ()
    assert torch.is_tensor(item["gt_y"]) and item["gt_y"].shape == ()


def test_manifest_dataset_gt_values_match_manifest(tmp_path):
    manifest_path = _write_manifest(tmp_path, 3)
    ds = ManifestDataset(manifest_path)
    item = ds[1]
    assert float(item["gt_x"]) == 401.0
    assert float(item["gt_y"]) == 501.0


def test_manifest_dataset_standardizes_images(tmp_path):
    """Standardized images should be roughly zero-mean, unit-std -- the same
    contract LocalizerDataset's on-the-fly samples already produce, so the
    training loop's targets/loss code needs no special-casing per data
    source."""
    manifest_path = _write_manifest(tmp_path, 1)
    ds = ManifestDataset(manifest_path)
    item = ds[0]
    assert abs(float(item["reference_img"].mean())) < 0.5
    assert abs(float(item["reference_img"].std()) - 1.0) < 0.5


def test_manifest_dataset_works_with_dataloader_batching(tmp_path):
    from torch.utils.data import DataLoader

    manifest_path = _write_manifest(tmp_path, 4)
    ds = ManifestDataset(manifest_path)
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    assert batch["reference_img"].shape == (2, 1, REF_DS_PX, REF_DS_PX)
    assert batch["search_img"].shape == (2, 1, SEARCH_PX, SEARCH_PX)
    assert batch["gt_x"].shape == (2,)
