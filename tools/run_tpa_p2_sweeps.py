#!/usr/bin/env python
"""Launch small P2 TPA sweeps for LVIS experiments.

The script prints commands by default. Pass --run to execute them sequentially.
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_CONFIG = "lami_dino/configs/dino_convnext_large_4scale_12ep_lvis.py"
DEFAULT_ANCHOR_INIT = "/root/lami_convnext_large_12ep_lvis_20260419_132029/model_0042599.pth"


def _tag(value: float) -> str:
    return f"{int(round(value * 1000)):03d}"


def _format_cmd(cmd):
    return " ".join(shlex.quote(part) for part in cmd)


def _train_net_base(args):
    return [
        sys.executable,
        "tools/train_net.py",
        "--num-gpus",
        str(args.num_gpus),
        "--config-file",
        args.config,
    ]


def _tau_eval_cmd(args, tau: float):
    out_dir = Path(args.output_root) / "p2_tau_sweep" / f"tau{_tag(tau)}"
    return _train_net_base(args) + [
        "--eval-only",
        f"train.init_checkpoint={args.tau_checkpoint}",
        f"train.output_dir={out_dir}",
        f"dataloader.evaluator.output_dir={out_dir}",
        f"model.classifier.tpa_tau={tau:g}",
    ]


def _anchor_train_cmd(args, anchor: float):
    out_dir = Path(args.output_root) / f"p2_anchor{_tag(anchor)}_ep1_lr1e5"
    return _train_net_base(args) + [
        f"train.output_dir={out_dir}",
        f"dataloader.evaluator.output_dir={out_dir}",
        f"train.init_checkpoint={args.anchor_init_checkpoint}",
        f"train.max_iter={args.max_iter}",
        f"train.eval_period={args.eval_period}",
        f"train.checkpointer.period={args.checkpointer_period}",
        f"train.log_period={args.log_period}",
        f"optimizer.lr={args.lr:g}",
        f"model.classifier.tpa_novel_anchor_weight={anchor:g}",
    ]


def _run_or_print(commands, *, run: bool, cuda_visible_devices: str = ""):
    env = os.environ.copy()
    prefix = ""
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        prefix = f"CUDA_VISIBLE_DEVICES={shlex.quote(cuda_visible_devices)} "

    for cmd in commands:
        print(prefix + _format_cmd(cmd), flush=True)
        if run:
            subprocess.run(cmd, check=True, env=env)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("tau", "anchor", "all"), default="all")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument("--output-root", default="/root")
    parser.add_argument("--run", action="store_true", help="Execute commands instead of only printing them.")

    parser.add_argument("--tau-checkpoint", default="", help="P2 checkpoint to evaluate for tau sweep.")
    parser.add_argument(
        "--tau-values",
        type=float,
        nargs="+",
        default=[0.03, 0.05, 0.07, 0.10, 0.15],
    )

    parser.add_argument("--anchor-init-checkpoint", default=DEFAULT_ANCHOR_INIT)
    parser.add_argument(
        "--anchor-values",
        type=float,
        nargs="+",
        default=[0.0, 0.02, 0.10],
    )
    parser.add_argument("--max-iter", type=int, default=7100)
    parser.add_argument("--eval-period", type=int, default=7100)
    parser.add_argument("--checkpointer-period", type=int, default=7100)
    parser.add_argument("--log-period", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1.5e-5)
    return parser.parse_args()


def main():
    args = parse_args()
    commands = []

    if args.phase in ("tau", "all"):
        if not args.tau_checkpoint:
            raise SystemExit("--tau-checkpoint is required for tau sweep.")
        commands.extend(_tau_eval_cmd(args, tau) for tau in args.tau_values)

    if args.phase in ("anchor", "all"):
        commands.extend(_anchor_train_cmd(args, anchor) for anchor in args.anchor_values)

    _run_or_print(commands, run=args.run, cuda_visible_devices=args.cuda_visible_devices)


if __name__ == "__main__":
    main()
