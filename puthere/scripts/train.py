#!/usr/bin/env python3
"""Train the Drift-Sense localizer.

The checkpoint-selection criterion is acc@Tpx, where T in {1, 3, 5} is set
via --best-metric (default 5). Region accuracy at this tolerance is where
the headroom is, since the ZNCC baseline already reaches 1.0 px median
error whenever it picks the right region.

Example:
    python scripts/train.py --run-name m1_no_context --no-context --max-steps 4000
    python scripts/train.py --run-name m2_context --max-steps 40000
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
from src.localizer.data import LocalizerDataset
from src.localizer.decode import decode
from src.localizer.losses import focal_heatmap_loss, offset_loss
from src.localizer.metrics import summarize
from src.localizer.model import DriftSenseLocalizer
from src.localizer.targets import build_targets
from src.presets import DRAM_PRESET_NAMES


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
    p.add_argument("--imaging-noise-profile", default="normal",
                   choices=["normal", "harsh"],
                   help="'harsh' widens every acquisition-noise and polygon-"
                        "distortion knob (dose, drift, astigmatism, vignette, "
                        "barrel distortion, charging streaks, speckle, salt-"
                        "and-pepper, CD bias, corner rounding) for training a "
                        "model more robust to noisier inputs than 'normal'.")
    p.add_argument("--geometric-profile", default="normal",
                   choices=["normal", "drift"],
                   help="'drift' adds ~1-2 degree rotation and 9:1-11:1 "
                        "scale-ratio jitter to reference crops during "
                        "training, for robustness to stage-drift/"
                        "calibration variation between the reference and "
                        "search captures (see src/localizer/data.py's "
                        "GEOMETRIC_PROFILES).")
    p.add_argument("--dram-presets", nargs="+", default=None,
                   choices=DRAM_PRESET_NAMES,
                   help="restrict every mat's DRAM preset draw to this "
                        "subset instead of all six (dram_1x, dram_dense, "
                        "dram_loose, dram_wide, dram_compact, dram_legacy) "
                        "-- e.g. fine-tuning on only the presets a prior "
                        "checkpoint tested weak on. Omit for the full pool.")
    p.add_argument("--best-metric", type=int, default=5, choices=[1, 3, 5],
                   help="which acc@Tpx tolerance (in px) decides checkpoint "
                        "'best' and the AP primary threshold. Also joins "
                        "--resume's profile-mismatch check: resuming under a "
                        "different --best-metric resets best-so-far, same "
                        "as a changed noise/geometric profile.")
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--val-batches", type=int, default=40)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--out-dir", default="./checkpoints")
    p.add_argument("--resume", action="store_true",
                   help="resume from <out-dir>/<run-name>/last.pt instead of "
                        "starting a fresh model. Requires that file to exist "
                        "(i.e. a previous run of this --run-name saved a checkpoint).")
    p.add_argument("--init-from", default=None,
                   help="warm-start model weights (and align_offset) from "
                        "another run's checkpoint, then start THIS run at "
                        "step=0 with a fresh optimizer/scheduler under this "
                        "invocation's own --max-steps/profiles. Unlike "
                        "--resume (which restores optimizer/scheduler/step "
                        "from this run's OWN last.pt), --init-from only "
                        "transplants weights -- for fine-tuning into a new "
                        "augmentation profile after a prior run's LR "
                        "schedule has already fully decayed. Mutually "
                        "exclusive with --resume.")
    p.add_argument("--steps-this-run", type=int, default=None,
                   help="train in a chunk: stop after this many ADDITIONAL steps "
                        "from wherever this invocation starts (0 if fresh, or the "
                        "resumed step count if --resume) and exit cleanly, rather "
                        "than running all the way to --max-steps in one sitting. "
                        "Re-run with --resume to continue the next chunk. Omit to "
                        "run straight through to --max-steps as before.")
    return p.parse_args()


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

    # Prediction-dispersion monitoring. Four separate corner-bias/collapse
    # bugs surfaced during this branch's development, and every one of them
    # was only ever caught by a human noticing an oddly round number (e.g.
    # every prediction landing on (50, 50)) -- shape-only unit tests cannot
    # see it. Surface it in every validation log instead: a collapsing model
    # shows up here as std_x/std_y shrinking toward 0 and n_distinct_preds
    # shrinking toward 1, well before the accuracy metrics make it obvious.
    m["std_px"] = float(np.std(px)) if px else 0.0
    m["std_py"] = float(np.std(py)) if py else 0.0
    m["n_distinct_preds"] = len(set(zip(px, py)))
    return m


def main():
    args = parse_args()
    # Normalize once so every comparison/save/DataLoader call below sees the
    # same canonical form regardless of the order flags were given in.
    args.dram_presets = sorted(args.dram_presets) if args.dram_presets else None
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
          f"lambda_hard_negative={cfg.lambda_hard_negative} "
          f"imaging_noise_profile={args.imaging_noise_profile} "
          f"geometric_profile={args.geometric_profile} "
          f"dram_presets={args.dram_presets or 'all'}")

    model = DriftSenseLocalizer(cfg, use_context=not args.no_context).to(device)

    val_loader = DataLoader(
        LocalizerDataset("val", cfg, args.jitter_profile,
                         imaging_noise_profile=args.imaging_noise_profile,
                         geometric_profile=args.geometric_profile,
                         dram_presets=args.dram_presets),
        batch_size=cfg.batch_size, num_workers=2)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    # T_max is the TOTAL step budget across every chunk of a resumed run, not
    # just this invocation's share -- the cosine schedule needs to know the
    # full horizon upfront to shape its curve correctly. Always pass the same
    # --max-steps on every resume of a given run.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_steps)
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))

    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    last_path = os.path.join(run_dir, "last.pt")
    best_path = os.path.join(run_dir, "best.pt")

    if args.resume and args.init_from:
        raise SystemExit(
            "--resume and --init-from are mutually exclusive -- --resume "
            "continues THIS run's own last.pt (full optimizer/scheduler/"
            "step state); --init-from warm-starts fresh from a DIFFERENT "
            "checkpoint's weights only. Pick one.")

    if args.resume:
        if not os.path.exists(last_path):
            raise SystemExit(f"--resume given but no checkpoint at {last_path} -- "
                              f"run without --resume first to start this run-name fresh.")
        ckpt = torch.load(last_path, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.align_offset.fill_(ckpt["align_offset"])  # redundant with load_state_dict
        opt.load_state_dict(ckpt["optimizer"])           # (buffer is persistent) but
        sched.load_state_dict(ckpt["scheduler"])          # explicit for clarity, matching
        scaler.load_state_dict(ckpt["scaler"])             # every other script's pattern.
        align = ckpt["align_offset"]
        step = ckpt["step"]
        best_acc = ckpt["best_acc"]
        # Checkpoints saved before this field existed are always "normal"
        # (the only profile that existed then) -- safe default for old runs.
        prev_profile = ckpt.get("imaging_noise_profile", "normal")
        prev_geom = ckpt.get("geometric_profile", "normal")
        prev_presets = ckpt.get("dram_presets", None)
        # None = predates the best_metric field (every production_v2-v5
        # checkpoint) -- labeled distinctly from a real 1/3/5 value below.
        prev_best_metric = ckpt.get("best_metric", None)
        prev_metric_label = (f"acc@{prev_best_metric}px" if prev_best_metric is not None
                              else "acc@50px (legacy)")
        print(f"resumed {run_dir} at step {step}/{args.max_steps} "
              f"(best {prev_metric_label} so far: {best_acc:.3f}, trained under "
              f"imaging_noise_profile={prev_profile}, "
              f"geometric_profile={prev_geom}, "
              f"dram_presets={prev_presets or 'all'})")
        if (prev_profile != args.imaging_noise_profile or prev_geom != args.geometric_profile
                or prev_presets != args.dram_presets or prev_best_metric != args.best_metric):
            print(f"  NOTE: profile changed (imaging_noise_profile {prev_profile} -> "
                  f"{args.imaging_noise_profile}, geometric_profile {prev_geom} -> "
                  f"{args.geometric_profile}, dram_presets {prev_presets or 'all'} -> "
                  f"{args.dram_presets or 'all'}, best_metric {prev_metric_label} -> "
                  f"acc@{args.best_metric}px) -- validation difficulty and/or "
                  f"selection criterion just changed, so the old "
                  f"best_acc={best_acc:.3f} isn't a fair bar for the new "
                  f"conditions. Resetting best-so-far to -1.0 so checkpoints "
                  f"save normally under the new conditions.")
            best_acc = -1.0
    elif args.init_from:
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
    else:
        align = model.calibrate()
        print(f"calibrated align_offset = {align:.3f} px")
        step, best_acc = 0, -1.0

    # Shift the train split's starting canvas seed forward by roughly how
    # many canvases prior chunks of this run already consumed, so a
    # --resume'd chunk explores fresh territory in the seed range instead of
    # re-walking the same few hundred canvases every chunk starts from.
    train_seed_offset = (step * cfg.batch_size) // cfg.crops_per_canvas
    train_loader = DataLoader(
        LocalizerDataset("train", cfg, args.jitter_profile,
                         seed_offset=train_seed_offset,
                         imaging_noise_profile=args.imaging_noise_profile,
                         geometric_profile=args.geometric_profile,
                         dram_presets=args.dram_presets),
        batch_size=cfg.batch_size, num_workers=args.num_workers, pin_memory=True)

    stop_step = args.max_steps
    if args.steps_this_run is not None:
        stop_step = min(args.max_steps, step + args.steps_this_run)
    t0 = time.time()

    for batch in train_loader:
        if step >= stop_step:
            break
        ref = batch["reference_img"].to(device, non_blocking=True)
        srch = batch["search_img"].to(device, non_blocking=True)

        # Defends against a rare upstream numerical-overflow edge case in the
        # synthetic imaging pipeline (src/sem_imaging.py's barrel-distortion/
        # vignette math, seen to occasionally emit inf/nan under harsh-profile
        # noise) that would otherwise silently poison BatchNorm's running
        # mean/var: those buffers update *during the forward pass itself*, in
        # train mode, regardless of what backward/optimizer-step guards exist
        # afterward. A single corrupted input can therefore leave training
        # loss looking perfectly normal for thousands of steps while eval
        # mode (which uses the frozen running stats) collapses to NaN the
        # next time it runs. Skipping the batch here -- before it ever
        # reaches model() -- is the only point that actually prevents that.
        if not (torch.isfinite(ref).all() and torch.isfinite(srch).all()):
            print(f"  WARNING: non-finite input at step {step} -- skipping this "
                  f"batch (not counted toward --steps-this-run/--max-steps)")
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

        # Second-layer guard: even with finite inputs, a still-developing
        # model or an extreme-but-finite input can occasionally produce a
        # non-finite loss. scaler.step() alone would only skip the optimizer
        # update here (GradScaler's own inf/nan check) -- but by this point
        # the forward pass (and any BatchNorm running-stat update) has
        # already happened, so this is a belt-and-suspenders check, not the
        # primary defense (that's the input check above).
        if not torch.isfinite(loss):
            print(f"  WARNING: non-finite loss at step {step} -- skipping "
                  f"optimizer step (not counted toward --steps-this-run/--max-steps)")
            continue

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1

        if step % 100 == 0:
            print(f"step {step:>6}  loss {float(loss):.4f}  "
                  f"{step / (time.time() - t0):.1f} it/s")

        if step % args.val_every == 0 or step == stop_step:
            m = evaluate(model, val_loader, cfg, device, args.val_batches, args.best_metric)
            print(f"  [val] acc@1px {m['acc@1px']:.3f}  acc@3px {m['acc@3px']:.3f}  "
                  f"acc@5px {m['acc@5px']:.3f}  median {m['median_error_px']:.1f}px  "
                  f"AP {m['ap']:.3f}")
            print(f"  [val] std_x {m['std_px']:.2f}px  std_y {m['std_py']:.2f}px  "
                  f"distinct_preds {m['n_distinct_preds']}/{m['n']}")

            # A validation this bad (nan predictions) means some forward pass
            # since the last checkpoint corrupted BatchNorm's running mean/var
            # -- the input/loss guards above only stop WEIGHT corruption via
            # backward()/optimizer.step(), not this, since running-stat
            # buffers update during forward() itself regardless of what
            # happens afterward. Rather than chase every possible upstream
            # numerical trigger, detect the collapse and roll back to the
            # last saved state: last_path still reflects the PREVIOUS
            # validation's (known-good) state, since it hasn't been
            # overwritten yet this cycle.
            collapsed = math.isnan(m["mean_error_px"]) or math.isnan(m["median_error_px"])
            if collapsed and os.path.exists(last_path):
                print(f"  WARNING: validation collapsed (nan predictions) -- rolling back "
                      f"model/optimizer/scheduler/scaler to the last known-good checkpoint "
                      f"at {last_path} and continuing training from there.")
                recover = torch.load(last_path, weights_only=False, map_location=device)
                model.load_state_dict(recover["model"])
                model.align_offset.fill_(recover["align_offset"])
                opt.load_state_dict(recover["optimizer"])
                sched.load_state_dict(recover["scheduler"])
                scaler.load_state_dict(recover["scaler"])
                step = recover["step"]
            elif collapsed:
                print("  WARNING: validation collapsed (nan predictions) but no prior "
                      "checkpoint exists to roll back to -- continuing training as-is; "
                      "if this doesn't self-correct, stop and investigate.")
            else:
                best_key = f"acc@{args.best_metric}px"
                if m[best_key] > best_acc:
                    best_acc = m[best_key]
                    torch.save({"model": model.state_dict(), "config": cfg.as_dict(),
                                "align_offset": align, "step": step, "metrics": m,
                                "imaging_noise_profile": args.imaging_noise_profile,
                                "geometric_profile": args.geometric_profile,
                                "dram_presets": args.dram_presets,
                                "best_metric": args.best_metric},
                               best_path)
                    print(f"  saved new best ({best_key} {best_acc:.3f})")

                # Full resumable state, overwritten every validation regardless
                # of whether this step was a new best -- best.pt above is a
                # lightweight inference-only snapshot; this one carries
                # everything needed to continue training exactly where it left
                # off via --resume.
                torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                            "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                            "config": cfg.as_dict(), "align_offset": align,
                            "step": step, "best_acc": best_acc, "metrics": m,
                            "imaging_noise_profile": args.imaging_noise_profile,
                            "geometric_profile": args.geometric_profile,
                            "dram_presets": args.dram_presets,
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
