import json
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader


class WaferDataset(Dataset):
    def __init__(self, root: str, split: str = "train"):
        self.root = Path(root)
        records   = json.loads((self.root / "metadata.json").read_text())
        self.items = [r for r in records if r["split"] == split]

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        rec  = self.items[idx]
        sp   = rec["split"]
        fid  = rec["id"]

        img  = cv2.imread(
            str(self.root / sp / "images" / f"{fid}.png"),
            cv2.IMREAD_GRAYSCALE
        ).astype(np.float32) / 255.0

        mask = cv2.imread(
            str(self.root / sp / "masks"  / f"{fid}.png"),
            cv2.IMREAD_GRAYSCALE
        ).astype(np.float32) / 255.0

        label = int(rec["has_defect"])

        return {
            "image":  torch.from_numpy(img).unsqueeze(0),   # (1, H, W)
            "mask":   torch.from_numpy(mask).unsqueeze(0),  # (1, H, W)
            "label":  torch.tensor(label, dtype=torch.long),
            "layout": rec["layout"],
            "defect": rec["defect"] or "none",
        }


# ── Quick test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    ds = WaferDataset("dataset", split="train")
    dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=4)

    batch = next(iter(dl))
    print("images:", batch["image"].shape)    # (16, 1, 512, 512)
    print("labels:", batch["label"])
    print("layouts:", batch["layout"])
