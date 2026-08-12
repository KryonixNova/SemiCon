#!/usr/bin/env python3
"""Fine-tune the Drift-Sense localizer on a persisted, manifest-backed
dataset (e.g. AFB's independently-rendered generate_dataset.py output)
instead of drift-sense's own on-the-fly synthetic pipeline (train.py /
LocalizerDataset).

Always a fine-tune, never from-scratch: --init-from is required. Mirrors
train.py's loss/optimizer/safety-net behavior exactly; the only real
difference is the data source -- a finite ManifestDataset iterated
epoch-wise instead of an infinite canvas-seed-driven IterableDataset.

Example:
    python scripts/finetune_on_manifest.py --run-name production_v5 \
        --init-from checkpoints/production_v5/best.pt \
        --train-manifest /path/to/afb_finetune/train/manifest.csv \
        --val-manifest /path/to/afb_finetune/val/manifest.csv \
        --lambda-hn 0.5 --max-steps 2000
"""

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.localizer.config import LocalizerConfig
from src.localizer.losses import focal_heatmap_loss, offset_loss
from src.localizer.manifest_data import ManifestDataset
from src.localizer.metrics import summarize
from src.localizer.model import DriftSenseLocalizer
from src.localizer.targets import build_targets


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", required=True)
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--val-manifest", required=True)
    p.add_argument("--init-from", default=None,
                   help="warm-start weights from another run's checkpoint "
                        "(required unless --resume). Mutually exclusive with --resume.")
    p.add_argument("--resume", action="store_true",
                   help="resume THIS run's own last.pt instead of --init-from.")
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--steps-this-run", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lambda-hn", type=float, default=0.0)
    p.add_argument("--hn-radius", type=int, default=None)
    p.add_argument("--val-every", type=int, default=200)
    p.add_argument("--val-batches", type=int, default=40)
    p.add_argument("--best-metric", type=int, default=5, choices=[1, 3, 5],
                   help="which acc@Tpx tolerance (in px) decides checkpoint "
                        "'best' and the AP primary threshold. Also joins "
                        "--resume's best_metric-mismatch check.")
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--out-dir", default="./checkpoints")
    args = p.parse_args()
    if args.resume and args.init_from:
        p.error("--resume and --init-from are mutually exclusive")
    if not args.resume and not args.init_from:
        p.error("must give --init-from (or --resume to continue this run's own checkpoint)")
    return args


def evaluate(model, loader, cfg, device, max_batches, best_metric):
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
    m = summarize(px, py, gx, gy, conf, primary_tolerance=float(best_metric))
    m["std_px"] = float(np.std(px)) if px else 0.0
    m["std_py"] = float(np.std(py)) if py else 0.0
    m["n_distinct_preds"] = len(set(zip(px, py)))
    return m


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
          f"lambda_hard_negative={cfg.lambda_hard_negative} "
          f"train_manifest={args.train_manifest} val_manifest={args.val_manifest}")

    model = DriftSenseLocalizer(cfg).to(device)

    val_loader = DataLoader(ManifestDataset(args.val_manifest),
                            batch_size=cfg.batch_size, shuffle=False,
                            num_workers=min(2, args.num_workers))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_steps)
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))

    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    last_path = os.path.join(run_dir, "last.pt")
    best_path = os.path.join(run_dir, "best.pt")

    if args.resume:
        if not os.path.exists(last_path):
            raise SystemExit(f"--resume given but no checkpoint at {last_path}")
        ckpt = torch.load(last_path, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.align_offset.fill_(ckpt["align_offset"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        align = ckpt["align_offset"]
        step = ckpt["step"]
        best_acc = ckpt["best_acc"]
        # None = predates the best_metric field -- every checkpoint saved
        # before this feature existed reads this way.
        prev_best_metric = ckpt.get("best_metric", None)
        prev_metric_label = (f"acc@{prev_best_metric}px" if prev_best_metric is not None
                              else "acc@50px (legacy)")
        print(f"resumed {run_dir} at step {step}/{args.max_steps} "
              f"(best {prev_metric_label} so far: {best_acc:.3f})")
        if prev_best_metric != args.best_metric:
            print(f"  NOTE: best_metric changed ({prev_metric_label} -> "
                  f"acc@{args.best_metric}px) -- the old best_acc={best_acc:.3f} "
                  f"isn't a fair bar for the new criterion. Resetting "
                  f"best-so-far to -1.0 so checkpoints save normally under "
                  f"the new metric.")
            best_acc = -1.0
    else:
        if not os.path.exists(args.init_from):
            raise SystemExit(f"--init-from checkpoint not found: {args.init_from}")
        src_ckpt = torch.load(args.init_from, weights_only=False, map_location=device)
        model.load_state_dict(src_ckpt["model"])
        align = float(src_ckpt["align_offset"])
        model.align_offset.fill_(align)
        step, best_acc = 0, -1.0
        src_step = src_ckpt.get("step", "?")
        src_acc = src_ckpt.get("metrics", {}).get(f"acc@{args.best_metric}px", float("nan"))
        print(f"initialized weights from {args.init_from} "
              f"(its step={src_step}, acc@{args.best_metric}px={src_acc:.3f}); "
              f"starting this run fresh at step 0 with a new optimizer/scheduler")

    train_ds = ManifestDataset(args.train_manifest)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True)
    print(f"train dataset: {len(train_ds)} samples "
          f"({len(train_ds) // cfg.batch_size} steps/epoch)")

    stop_step = args.max_steps
    if args.steps_this_run is not None:
        stop_step = min(args.max_steps, step + args.steps_this_run)
    t0 = time.time()

    def train_batches():
        while True:
            for batch in train_loader:
                yield batch

    for batch in train_batches():
        if step >= stop_step:
            break
        ref = batch["reference_img"].to(device, non_blocking=True)
        srch = batch["search_img"].to(device, non_blocking=True)

        if not (torch.isfinite(ref).all() and torch.isfinite(srch).all()):
            print(f"  WARNING: non-finite input at step {step} -- skipping this batch")
            continue

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

        if not torch.isfinite(loss):
            print(f"  WARNING: non-finite loss at step {step} -- skipping optimizer step")
            continue

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1

        if step % 50 == 0:
            print(f"step {step:>6}  loss {float(loss):.4f}  "
                  f"{step / (time.time() - t0):.1f} it/s")

        if step % args.val_every == 0 or step == stop_step:
            m = evaluate(model, val_loader, cfg, device, args.val_batches, args.best_metric)
            print(f"  [val] acc@1px {m['acc@1px']:.3f}  acc@3px {m['acc@3px']:.3f}  "
                  f"acc@5px {m['acc@5px']:.3f}  median {m['median_error_px']:.1f}px  "
                  f"AP {m['ap']:.3f}")
            print(f"  [val] std_x {m['std_px']:.2f}px  std_y {m['std_py']:.2f}px  "
                  f"distinct_preds {m['n_distinct_preds']}/{m['n']}")

            collapsed = math.isnan(m["mean_error_px"]) or math.isnan(m["median_error_px"])
            if collapsed and os.path.exists(last_path):
                print(f"  WARNING: validation collapsed -- rolling back to {last_path}")
                recover = torch.load(last_path, weights_only=False, map_location=device)
                model.load_state_dict(recover["model"])
                model.align_offset.fill_(recover["align_offset"])
                opt.load_state_dict(recover["optimizer"])
                sched.load_state_dict(recover["scheduler"])
                scaler.load_state_dict(recover["scaler"])
                step = recover["step"]
            elif collapsed:
                print("  WARNING: validation collapsed but no prior checkpoint to roll back to")
            else:
                best_key = f"acc@{args.best_metric}px"
                if m[best_key] > best_acc:
                    best_acc = m[best_key]
                    torch.save({"model": model.state_dict(), "config": cfg.as_dict(),
                                "align_offset": align, "step": step, "metrics": m,
                                "data_source": "manifest", "train_manifest": args.train_manifest,
                                "val_manifest": args.val_manifest,
                                "best_metric": args.best_metric},
                               best_path)
                    print(f"  saved new best ({best_key} {best_acc:.3f})")

                torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                            "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                            "config": cfg.as_dict(), "align_offset": align,
                            "step": step, "best_acc": best_acc, "metrics": m,
                            "data_source": "manifest", "train_manifest": args.train_manifest,
                            "val_manifest": args.val_manifest,
                            "best_metric": args.best_metric},
                           last_path)

    print(f"stopped at step {step}/{args.max_steps}. "
          f"best acc@{args.best_metric}px so far = {best_acc:.3f}")
    if step < args.max_steps:
        print(f"  {args.max_steps - step} steps remaining -- re-run with the same "
              f"command plus --resume to continue.")
    else:
        print("  done -- reached the full --max-steps budget.")


if __name__ == "__main__":
    main()
