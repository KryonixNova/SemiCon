#!/usr/bin/env python3
"""Train the Drift-Sense localizer.

Primary metric is acc@50px (region accuracy) for M0-M3: that is where the
headroom is, since the ZNCC baseline already reaches 1.0 px median error
whenever it picks the right region.

Example:
    python scripts/train.py --run-name m1_no_context --no-context --max-steps 4000
    python scripts/train.py --run-name m2_context --max-steps 40000
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.localizer.config import LocalizerConfig
from src.localizer.data import LocalizerDataset
from src.localizer.decode import decode
from src.localizer.losses import focal_heatmap_loss, offset_loss
from src.localizer.metrics import summarize
from src.localizer.model import DriftSenseLocalizer
from src.localizer.targets import build_targets


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--no-context", action="store_true", help="B2 ablation (M1)")
    p.add_argument("--max-steps", type=int, default=40000)
    # --batch-size / --lr / --hn-radius default to None, not a hardcoded
    # number: LocalizerConfig's own field defaults are the single source of
    # truth (e.g. hard_negative_radius_cells=24, selected by Task 17's
    # radius sweep -- acc@50px 0.875 vs 0.450 at radius 12). A hardcoded
    # argparse default here would silently win over the config's calibrated
    # value on every run that omits the flag. None means "no explicit
    # override"; see main() where these are only applied to `cfg` when set.
    p.add_argument("--batch-size", type=int, default=None,
                   help="override LocalizerConfig.batch_size (default: config's value)")
    p.add_argument("--lr", type=float, default=None,
                   help="override LocalizerConfig.lr (default: config's value)")
    # --lambda-hn is the one deliberate exception to the "config wins"
    # rule above: LocalizerConfig.lambda_hard_negative defaults to 0.5 (the
    # value selected once hard-negative training is on), but M0-M2 runs
    # must have it OFF by default. So this flag's own default of 0.0
    # intentionally overrides the config unconditionally, every run, unless
    # a run explicitly opts in via --lambda-hn.
    p.add_argument("--lambda-hn", type=float, default=0.0,
                   help="hard-negative weight; 0 until M3")
    p.add_argument("--hn-radius", type=int, default=None,
                   help="override LocalizerConfig.hard_negative_radius_cells "
                        "(default: config's value, 24)")
    p.add_argument("--jitter-profile", default="normal",
                   choices=["normal", "zero", "shifted"])
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--val-batches", type=int, default=40)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--out-dir", default="./checkpoints")
    return p.parse_args()


def evaluate(model, loader, cfg, device, max_batches):
    model.eval()
    px, py, gx, gy, conf = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            out = model.predict(batch["reference_img"].to(device),
                                batch["search_img"].to(device))
            px += out["x"].cpu().tolist()
            py += out["y"].cpu().tolist()
            conf += out["confidence"].cpu().tolist()
            gx += batch["gt_x"].tolist()
            gy += batch["gt_y"].tolist()
    model.train()
    m = summarize(px, py, gx, gy, conf)

    # Prediction-dispersion monitoring. Four separate corner-bias/collapse
    # bugs surfaced during this branch's development, and every one of them
    # was only ever caught by a human noticing an oddly round number (e.g.
    # every prediction landing on (50, 50)) -- shape-only unit tests cannot
    # see it. Surface it in every validation log instead: a collapsing model
    # shows up here as std_x/std_y shrinking toward 0 and n_distinct_preds
    # shrinking toward 1, well before acc@50px makes it obvious.
    m["std_px"] = float(np.std(px)) if px else 0.0
    m["std_py"] = float(np.std(py)) if py else 0.0
    m["n_distinct_preds"] = len(set(zip(px, py)))
    return m


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # LocalizerConfig()'s own field defaults are the source of truth; CLI
    # flags only override what a run explicitly sets (--lambda-hn is the
    # one deliberate exception -- see its help text in parse_args()).
    cfg = LocalizerConfig()
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.hn_radius is not None:
        cfg.hard_negative_radius_cells = args.hn_radius
    cfg.lambda_hard_negative = args.lambda_hn
    print(f"config: batch_size={cfg.batch_size} lr={cfg.lr} "
          f"hard_negative_radius_cells={cfg.hard_negative_radius_cells} "
          f"lambda_hard_negative={cfg.lambda_hard_negative}")

    model = DriftSenseLocalizer(cfg, use_context=not args.no_context).to(device)
    align = model.calibrate()
    print(f"calibrated align_offset = {align:.3f} px")

    train_loader = DataLoader(
        LocalizerDataset("train", cfg, args.jitter_profile),
        batch_size=cfg.batch_size, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(
        LocalizerDataset("val", cfg, args.jitter_profile),
        batch_size=cfg.batch_size, num_workers=2)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_steps)
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))

    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    best_acc, step, t0 = -1.0, 0, time.time()

    for batch in train_loader:
        if step >= args.max_steps:
            break
        ref = batch["reference_img"].to(device, non_blocking=True)
        srch = batch["search_img"].to(device, non_blocking=True)
        tgt = build_targets(batch["gt_x"], batch["gt_y"],
                            sigma_cells=cfg.heatmap_sigma_cells,
                            align_offset=align)
        hm_t = tgt["heatmap"].to(device)
        off_t = tgt["offset"].to(device)
        peak = tgt["peak_cell"].to(device)

        with torch.amp.autocast(device, enabled=(device == "cuda")):
            logits, offset = model(ref, srch)
            loss = focal_heatmap_loss(logits, hm_t, cfg.focal_alpha, cfg.focal_beta)
            loss = loss + cfg.lambda_offset * offset_loss(offset, off_t, peak)
            if cfg.lambda_hard_negative > 0:
                from src.localizer.losses import hard_negative_loss
                loss = loss + cfg.lambda_hard_negative * hard_negative_loss(
                    logits, peak, cfg.hard_negative_radius_cells)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1

        if step % 100 == 0:
            print(f"step {step:>6}  loss {float(loss):.4f}  "
                  f"{step / (time.time() - t0):.1f} it/s")

        if step % args.val_every == 0:
            m = evaluate(model, val_loader, cfg, device, args.val_batches)
            print(f"  [val] acc@50px {m['acc@50px']:.3f}  acc@5px {m['acc@5px']:.3f}  "
                  f"median {m['median_error_px']:.1f}px  AP {m['ap']:.3f}")
            print(f"  [val] std_x {m['std_px']:.2f}px  std_y {m['std_py']:.2f}px  "
                  f"distinct_preds {m['n_distinct_preds']}/{m['n']}")
            if m["acc@50px"] > best_acc:
                best_acc = m["acc@50px"]
                torch.save({"model": model.state_dict(), "config": cfg.as_dict(),
                            "align_offset": align, "step": step, "metrics": m},
                           os.path.join(run_dir, "best.pt"))
                print(f"  saved new best (acc@50px {best_acc:.3f})")

    print(f"done. best acc@50px = {best_acc:.3f}")


if __name__ == "__main__":
    main()
