"""
Step 7 — Evaluation & Inference
Loads best.pt, runs on the test split, computes metrics,
and saves a visual report to Desktop/eval_report/.
"""
import sys, os
_desktop = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _desktop)

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
)

from wafer_dataset import WaferDataset
from train import UNetWafer

# ── Config ────────────────────────────────────────────────────────────────
DESKTOP   = "/home/nihal/Desktop"
DATA_ROOT = os.path.join(DESKTOP, "dataset")
CKPT      = os.path.join(DESKTOP, "checkpoints", "best.pt")
OUT_DIR   = os.path.join(DESKTOP, "eval_report")
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
BATCH     = 8


# ── Dice / IoU helpers ────────────────────────────────────────────────────
def dice_score(pred_bin, gt_bin, eps=1e-6):
    inter = (pred_bin & gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum()
    return (2 * inter + eps) / (union + eps)

def iou_score(pred_bin, gt_bin, eps=1e-6):
    inter = (pred_bin & gt_bin).sum()
    union = (pred_bin | gt_bin).sum()
    return (inter + eps) / (union + eps)


# ── Load model ────────────────────────────────────────────────────────────
model = UNetWafer(base=32).to(DEVICE)
ckpt  = torch.load(CKPT, map_location=DEVICE)
model.load_state_dict(ckpt["state"])
model.eval()
print(f"Loaded checkpoint  epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.4f}")

# ── DataLoader ────────────────────────────────────────────────────────────
test_ds = WaferDataset(DATA_ROOT, split="test")
test_dl = DataLoader(test_ds, batch_size=BATCH, shuffle=False,
                     num_workers=4, pin_memory=True)
print(f"Test set: {len(test_ds)} samples  |  device={DEVICE}\n")

# ── Inference ─────────────────────────────────────────────────────────────
all_labels, all_preds, all_probs = [], [], []
all_dice, all_iou = [], []
per_layout = defaultdict(lambda: {"labels": [], "preds": [], "dice": [], "iou": []})
samples_for_plot = []   # store a few for visual panel

with torch.no_grad():
    for batch in test_dl:
        imgs   = batch["image"].to(DEVICE)
        masks  = batch["mask"].to(DEVICE)
        labels = batch["label"]

        seg_log, cls_log = model(imgs)
        probs  = F.softmax(cls_log, dim=1)[:, 1].cpu().numpy()
        preds  = cls_log.argmax(1).cpu().numpy()
        seg_pr = torch.sigmoid(seg_log).cpu().numpy()

        for i in range(len(labels)):
            lbl = labels[i].item()
            pr  = preds[i]
            mgt = masks[i, 0].cpu().numpy() > 0.5
            mpr = seg_pr[i, 0] > 0.5
            dc  = dice_score(mpr, mgt)
            io  = iou_score(mpr, mgt)
            lt  = batch["layout"][i]
            dk  = batch["defect"][i]

            all_labels.append(lbl)
            all_preds.append(pr)
            all_probs.append(probs[i])
            all_dice.append(dc)
            all_iou.append(io)
            per_layout[lt]["labels"].append(lbl)
            per_layout[lt]["preds"].append(pr)
            per_layout[lt]["dice"].append(dc)
            per_layout[lt]["iou"].append(io)

            if len(samples_for_plot) < 16:
                samples_for_plot.append({
                    "img":    imgs[i, 0].cpu().numpy(),
                    "mask_gt":mgt.astype(np.float32),
                    "mask_pr":seg_pr[i, 0],
                    "label":  lbl,
                    "pred":   pr,
                    "prob":   probs[i],
                    "layout": lt,
                    "defect": dk,
                    "dice":   dc,
                    "iou":    io,
                })

# ── Metrics ───────────────────────────────────────────────────────────────
all_labels = np.array(all_labels)
all_preds  = np.array(all_preds)
all_probs  = np.array(all_probs)
all_dice   = np.array(all_dice)
all_iou    = np.array(all_iou)

acc    = (all_preds == all_labels).mean()
auc    = roc_auc_score(all_labels, all_probs)
ap     = average_precision_score(all_labels, all_probs)
cm     = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds,
                                target_names=["clean", "defect"])

print("=" * 58)
print(f"  Classification")
print(f"    Accuracy      : {acc*100:.1f}%")
print(f"    ROC-AUC       : {auc:.4f}")
print(f"    Avg Precision : {ap:.4f}")
print(f"\n  Segmentation")
print(f"    Dice (mean)   : {all_dice.mean():.4f}  ±{all_dice.std():.4f}")
print(f"    IoU  (mean)   : {all_iou.mean():.4f}  ±{all_iou.std():.4f}")
print(f"\n{report}")
print("  Confusion matrix  [rows=GT, cols=Pred]")
print(f"           clean  defect")
print(f"  clean    {cm[0,0]:>5}   {cm[0,1]:>5}")
print(f"  defect   {cm[1,0]:>5}   {cm[1,1]:>5}")
print("=" * 58)

# Per-layout breakdown
print(f"\n  Per-layout  (acc / dice / iou)")
for lt, d in sorted(per_layout.items()):
    la, lp = np.array(d["labels"]), np.array(d["preds"])
    la_acc  = (la == lp).mean()
    ld, li  = np.mean(d["dice"]), np.mean(d["iou"])
    print(f"    {lt:<8}: acc={la_acc*100:.0f}%  dice={ld:.3f}  iou={li:.3f}")

# ── Save JSON report ──────────────────────────────────────────────────────
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
metrics = {
    "accuracy":    float(acc),
    "roc_auc":     float(auc),
    "avg_precision": float(ap),
    "dice_mean":   float(all_dice.mean()),
    "dice_std":    float(all_dice.std()),
    "iou_mean":    float(all_iou.mean()),
    "iou_std":     float(all_iou.std()),
    "confusion_matrix": cm.tolist(),
    "per_layout":  {lt: {
        "accuracy": float((np.array(d["labels"]) == np.array(d["preds"])).mean()),
        "dice":     float(np.mean(d["dice"])),
        "iou":      float(np.mean(d["iou"])),
        "n":        len(d["labels"]),
    } for lt, d in per_layout.items()},
}
Path(OUT_DIR, "metrics.json").write_text(json.dumps(metrics, indent=2))


# ── Visual panel 1: sample predictions ───────────────────────────────────
n = len(samples_for_plot)
cols = 4
fig, axes = plt.subplots(n, cols, figsize=(cols * 3.5, n * 3.2), facecolor="#0C1018")
fig.suptitle("Step 7 — Test Set Predictions   (SEM image | GT mask | Pred mask | Overlay)",
             color="white", fontsize=11, fontfamily="monospace", fontweight="bold", y=1.001)

COL_TITLES = ["SEM Image", "GT Mask", "Pred Mask", "Overlay (pred)"]
for c, t in enumerate(COL_TITLES):
    axes[0][c].set_title(t, color="#7B9CFF", fontsize=8, fontfamily="monospace",
                         fontweight="bold", pad=5)

for row, s in enumerate(samples_for_plot):
    overlay = np.stack([s["img"]] * 3, axis=-1)
    pred_mask = s["mask_pr"] > 0.5
    overlay[pred_mask, 0] = np.clip(overlay[pred_mask, 0] + 0.4, 0, 1)

    imgs_row = [s["img"], s["mask_gt"], s["mask_pr"], overlay]
    cmaps    = ["gray", "gray", "plasma", None]
    for c, (im, cm_) in enumerate(zip(imgs_row, cmaps)):
        ax = axes[row][c]
        if cm_ is None:
            ax.imshow(np.clip(im, 0, 1))
        else:
            ax.imshow(im, cmap=cm_, vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#243047")

    correct = s["label"] == s["pred"]
    tag_col = "#A5D6A7" if correct else "#EF9A9A"
    gt_str  = "defect" if s["label"] else "clean"
    pr_str  = "defect" if s["pred"]  else "clean"
    axes[row][0].set_ylabel(
        f"{s['layout']}\nGT:{gt_str}  PR:{pr_str}",
        color=tag_col, fontsize=7.5, fontfamily="monospace", labelpad=4
    )
    axes[row][3].text(0.97, 0.03,
        f"dice={s['dice']:.2f}  iou={s['iou']:.2f}",
        transform=axes[row][3].transAxes, ha="right", va="bottom",
        color="#FFF176", fontsize=6.5, fontfamily="monospace"
    )

fig.tight_layout()
p1 = os.path.join(OUT_DIR, "predictions.png")
fig.savefig(p1, dpi=120, bbox_inches="tight", facecolor="#0C1018")
plt.close(fig)
print(f"\nPredictions panel → {p1}")


# ── Visual panel 2: confusion matrix + dice histogram ────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor="#0C1018")
fig.suptitle("Step 7 — Test Metrics Summary", color="white",
             fontsize=11, fontfamily="monospace", fontweight="bold")

# Confusion matrix heatmap
ax = axes[0]
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["clean", "defect"], color="white", fontfamily="monospace")
ax.set_yticklabels(["clean", "defect"], color="white", fontfamily="monospace")
ax.set_xlabel("Predicted", color="#7B9CFF", fontfamily="monospace")
ax.set_ylabel("Ground Truth", color="#7B9CFF", fontfamily="monospace")
ax.set_title(f"Confusion Matrix\nacc={acc*100:.1f}%  AUC={auc:.3f}",
             color="white", fontfamily="monospace", fontsize=9)
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white", fontsize=16, fontweight="bold")
ax.tick_params(colors="white")

# Dice histogram
ax = axes[1]
ax.hist(all_dice, bins=20, color="#7B9CFF", edgecolor="#0C1018", alpha=0.85)
ax.axvline(all_dice.mean(), color="#FFF176", linestyle="--", linewidth=1.5,
           label=f"mean={all_dice.mean():.3f}")
ax.set_facecolor("#141B27")
ax.set_title("Dice Score Distribution", color="white",
             fontfamily="monospace", fontsize=9)
ax.set_xlabel("Dice", color="#7B9CFF", fontfamily="monospace")
ax.set_ylabel("Count",  color="#7B9CFF", fontfamily="monospace")
ax.tick_params(colors="white")
ax.legend(facecolor="#0C1018", edgecolor="#243047", labelcolor="white",
          fontsize=8)

# Per-layout bar chart
ax = axes[2]
layouts = sorted(per_layout.keys())
accs  = [(np.array(per_layout[l]["labels"]) == np.array(per_layout[l]["preds"])).mean() * 100
         for l in layouts]
dices = [np.mean(per_layout[l]["dice"]) for l in layouts]
x = np.arange(len(layouts))
w = 0.35
ax.bar(x - w/2, accs,  w, label="Accuracy %", color="#A5D6A7", alpha=0.85)
ax.bar(x + w/2, [d*100 for d in dices], w, label="Dice ×100", color="#CE93D8", alpha=0.85)
ax.set_facecolor("#141B27")
ax.set_xticks(x); ax.set_xticklabels(layouts, color="white",
                                      fontfamily="monospace", fontsize=8)
ax.set_title("Per-Layout Performance", color="white",
             fontfamily="monospace", fontsize=9)
ax.tick_params(colors="white")
ax.legend(facecolor="#0C1018", edgecolor="#243047", labelcolor="white", fontsize=8)
ax.set_ylim(0, 105)

fig.tight_layout()
p2 = os.path.join(OUT_DIR, "metrics_summary.png")
fig.savefig(p2, dpi=130, bbox_inches="tight", facecolor="#0C1018")
plt.close(fig)
print(f"Metrics summary   → {p2}")
print(f"\nStep 7 complete.\n")
