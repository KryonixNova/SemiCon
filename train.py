"""
Step 6 — Model Training
Trains a lightweight U-Net with a defect classification head on the
synthetic wafer dataset produced in Step 5.

Outputs:
  Desktop/checkpoints/best.pt   — best val-loss checkpoint
  Desktop/checkpoints/last.pt   — final epoch checkpoint
  Desktop/checkpoints/train_log.json
"""
import sys, os
_desktop = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _desktop)

import json, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from wafer_dataset import WaferDataset

# ── Config ────────────────────────────────────────────────────────────────
DESKTOP    = "/home/nihal/Desktop"
DATA_ROOT  = os.path.join(DESKTOP, "dataset")
CKPT_DIR   = os.path.join(DESKTOP, "checkpoints")
EPOCHS     = 30
BATCH      = 8
LR         = 3e-4
NUM_WORKERS= 4
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# Loss weights
W_SEG  = 1.0   # segmentation (dice + bce)
W_CLS  = 0.5   # defect classification


# ── U-Net ─────────────────────────────────────────────────────────────────
def _conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNetWafer(nn.Module):
    """
    Lightweight U-Net (1-channel in → 1-channel mask out) with a
    global-average-pool classification head for defect detection.
    """
    def __init__(self, base=32):
        super().__init__()
        b = base
        # Encoder
        self.enc1 = _conv_block(1,    b)
        self.enc2 = _conv_block(b,  2*b)
        self.enc3 = _conv_block(2*b,4*b)
        self.enc4 = _conv_block(4*b,8*b)
        self.pool = nn.MaxPool2d(2)
        # Bottleneck
        self.bot  = _conv_block(8*b, 16*b)
        # Decoder
        self.up4  = nn.ConvTranspose2d(16*b, 8*b, 2, stride=2)
        self.dec4 = _conv_block(16*b, 8*b)
        self.up3  = nn.ConvTranspose2d(8*b,  4*b, 2, stride=2)
        self.dec3 = _conv_block(8*b,  4*b)
        self.up2  = nn.ConvTranspose2d(4*b,  2*b, 2, stride=2)
        self.dec2 = _conv_block(4*b,  2*b)
        self.up1  = nn.ConvTranspose2d(2*b,  b,   2, stride=2)
        self.dec1 = _conv_block(2*b,  b)
        # Segmentation head
        self.seg_head = nn.Conv2d(b, 1, 1)
        # Classification head (on bottleneck features)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16*b, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bot(self.pool(e4))
        cls_logits = self.cls_head(b)
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        seg_logits = self.seg_head(d1)
        return seg_logits, cls_logits


# ── Losses ────────────────────────────────────────────────────────────────
def dice_loss(pred, target, eps=1e-6):
    pred   = torch.sigmoid(pred)
    inter  = (pred * target).sum(dim=(2, 3))
    union  = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1 - (2 * inter + eps) / (union + eps)


def seg_loss(logits, masks):
    bce  = F.binary_cross_entropy_with_logits(logits, masks, reduction="mean")
    dice = dice_loss(logits, masks).mean()
    return 0.5 * bce + 0.5 * dice


# ── Training loop ─────────────────────────────────────────────────────────
def run_epoch(model, loader, optim, device, train=True):
    model.train(train)
    tot_loss = tot_seg = tot_cls = 0.0
    correct  = total = 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            imgs   = batch["image"].to(device)
            masks  = batch["mask"].to(device)
            labels = batch["label"].to(device)

            seg_log, cls_log = model(imgs)
            ls = seg_loss(seg_log, masks)
            lc = F.cross_entropy(cls_log, labels)
            loss = W_SEG * ls + W_CLS * lc

            if train:
                optim.zero_grad()
                loss.backward()
                optim.step()

            bs = imgs.size(0)
            tot_loss += loss.item() * bs
            tot_seg  += ls.item()   * bs
            tot_cls  += lc.item()   * bs
            preds     = cls_log.argmax(1)
            correct  += (preds == labels).sum().item()
            total    += bs

    n = total
    return {
        "loss": tot_loss / n,
        "seg":  tot_seg  / n,
        "cls":  tot_cls  / n,
        "acc":  correct  / n,
    }


def main():
    Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)

    train_ds = WaferDataset(DATA_ROOT, split="train")
    val_ds   = WaferDataset(DATA_ROOT, split="val")
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

    model = UNetWafer(base=32).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel  : UNetWafer  ({n_params/1e6:.2f}M params)")
    print(f"Device : {DEVICE}")
    print(f"Train  : {len(train_ds)} samples   Val: {len(val_ds)} samples")
    print(f"Epochs : {EPOCHS}   Batch: {BATCH}   LR: {LR}\n")
    print(f"{'Ep':>3}  {'T-loss':>7} {'T-seg':>7} {'T-cls':>7} {'T-acc':>6}"
          f"  │  {'V-loss':>7} {'V-seg':>7} {'V-cls':>7} {'V-acc':>6}  {'time':>5}")
    print("─" * 90)

    log      = []
    best_val = float("inf")
    t0_total = time.time()

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        tr = run_epoch(model, train_dl, optim, DEVICE, train=True)
        va = run_epoch(model, val_dl,   optim, DEVICE, train=False)
        sched.step()
        elapsed = time.time() - t0

        print(f"{ep:>3}  {tr['loss']:>7.4f} {tr['seg']:>7.4f} {tr['cls']:>7.4f} "
              f"{tr['acc']*100:>5.1f}%"
              f"  │  {va['loss']:>7.4f} {va['seg']:>7.4f} {va['cls']:>7.4f} "
              f"{va['acc']*100:>5.1f}%  {elapsed:>4.0f}s")

        rec = {"epoch": ep, "train": tr, "val": va, "lr": sched.get_last_lr()[0]}
        log.append(rec)

        torch.save({"epoch": ep, "state": model.state_dict(),
                    "optim": optim.state_dict(), "val_loss": va["loss"]},
                   os.path.join(CKPT_DIR, "last.pt"))
        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save({"epoch": ep, "state": model.state_dict(),
                        "val_loss": va["loss"]},
                       os.path.join(CKPT_DIR, "best.pt"))
            print(f"     ↳ best val loss {best_val:.4f}  (saved)")

    total_time = time.time() - t0_total
    Path(CKPT_DIR, "train_log.json").write_text(json.dumps(log, indent=2))

    print(f"\n{'─'*50}")
    print(f"  Training complete  ({total_time/60:.1f} min)")
    print(f"  Best val loss : {best_val:.4f}")
    print(f"  Checkpoints   → {CKPT_DIR}/")
    print(f"\nStep 6 complete.\n")


if __name__ == "__main__":
    main()
