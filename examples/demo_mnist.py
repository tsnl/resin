"""MNIST MLP demo with Resin (WGPU) or PyTorch backends."""

# pyright: reportMissingImports=false
# pyright: reportImplicitRelativeImport=false
# pyright: reportUnusedCallResult=false
# pyright: reportAny=false

# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "resin",
#   "resin-rt-pybind",
#   "numpy",
#   "torch",
# ]
#
# [tool.uv.sources]
# resin = { path = "..", editable = true }
# resin-rt-pybind = { path = "../crates/resin-rt-pybind", editable = true }
# ///

import argparse
import sys
from typing import assert_never

from demo_mnist_common import TrainConfig
from demo_mnist_pytorch import run_pytorch
from demo_mnist_resin import run_resin


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("resin", "pytorch"),
        default="resin",
        help="Training runtime (default: resin).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=28)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--optimizer",
        choices=("adam", "adamw", "sgd"),
        default="adam",
        help="PyTorch only (default: adam). Resin always uses SGD.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch only: auto, cpu, mps, or cuda.",
    )
    parser.add_argument(
        "--match-resin-loss",
        action="store_true",
        help="PyTorch only: softmax in forward, then log(probs). "
        + "Default uses log-softmax on logits.",
    )
    parser.add_argument("--benchmark-steps", type=int, default=0)
    parser.add_argument("--benchmark-warmup", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"backend={args.backend}", file=sys.stderr)
    config = TrainConfig(
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_classes=args.num_classes,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        steps=args.steps,
        eval_interval=args.eval_interval,
        seed=args.seed,
    )

    match args.backend:
        case "resin":
            run_resin(
                config,
                benchmark_steps=args.benchmark_steps,
                benchmark_warmup=args.benchmark_warmup,
            )
        case "pytorch":
            run_pytorch(
                config,
                optimizer_name=args.optimizer,
                device_name=args.device,
                match_resin_loss=args.match_resin_loss,
                benchmark_steps=args.benchmark_steps,
                benchmark_warmup=args.benchmark_warmup,
            )
        case _:
            assert_never(args.backend)


if __name__ == "__main__":
    main()