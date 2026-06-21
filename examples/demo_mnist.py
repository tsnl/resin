"""MNIST MLP demo with Resin (WGPU) or PyTorch backends."""

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
from collections.abc import Sequence
from typing import assert_never, cast

from demo_mnist_common import DemoMnistCli, TrainConfig
from demo_mnist_pytorch import run_pytorch
from demo_mnist_resin import run_resin


def _require_int(
    value: object,
    parser: argparse.ArgumentParser,
    *,
    name: str,
) -> int:
    if isinstance(value, int):
        return value
    parser.error(f"invalid {name}: {value!r}")


def _require_float(
    value: object,
    parser: argparse.ArgumentParser,
    *,
    name: str,
) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    parser.error(f"invalid {name}: {value!r}")


def _require_str(
    value: object,
    parser: argparse.ArgumentParser,
    *,
    name: str,
) -> str:
    if isinstance(value, str):
        return value
    parser.error(f"invalid {name}: {value!r}")


def _require_bool(
    value: object,
    parser: argparse.ArgumentParser,
    *,
    name: str,
) -> bool:
    if isinstance(value, bool):
        return value
    parser.error(f"invalid {name}: {value!r}")


def _require_choice[T: str](
    value: object,
    choices: tuple[T, ...],
    parser: argparse.ArgumentParser,
    *,
    name: str,
) -> T:
    if isinstance(value, str) and value in choices:
        return cast(T, value)
    parser.error(f"invalid {name}: {value!r}")


def parse_cli(argv: Sequence[str] | None = None) -> DemoMnistCli:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        "--backend",
        choices=("resin", "pytorch"),
        default="resin",
        help="Training runtime (default: resin).",
    )
    _ = parser.add_argument("--batch-size", type=int, default=64)
    _ = parser.add_argument("--img-size", type=int, default=28)
    _ = parser.add_argument("--num-classes", type=int, default=10)
    _ = parser.add_argument("--hidden-size", type=int, default=128)
    _ = parser.add_argument("--learning-rate", type=float, default=5e-3)
    _ = parser.add_argument("--steps", type=int, default=100_000)
    _ = parser.add_argument("--eval-interval", type=int, default=100)
    _ = parser.add_argument("--seed", type=int, default=0)
    _ = parser.add_argument(
        "--optimizer",
        choices=("adam", "adamw", "sgd"),
        default="adam",
        help="PyTorch only (default: adam). Resin always uses SGD.",
    )
    _ = parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch only: auto, cpu, mps, or cuda.",
    )
    _ = parser.add_argument(
        "--match-resin-loss",
        action="store_true",
        help="PyTorch only: softmax in forward, then log(probs). "
        + "Default uses log-softmax on logits.",
    )
    _ = parser.add_argument("--benchmark-steps", type=int, default=0)
    _ = parser.add_argument("--benchmark-warmup", type=int, default=20)
    args: dict[str, object] = vars(
        parser.parse_args(list(argv) if argv is not None else None)
    )

    return DemoMnistCli(
        backend=_require_choice(
            args["backend"], ("resin", "pytorch"), parser, name="backend"
        ),
        train=TrainConfig(
            batch_size=_require_int(args["batch_size"], parser, name="batch_size"),
            img_size=_require_int(args["img_size"], parser, name="img_size"),
            num_classes=_require_int(args["num_classes"], parser, name="num_classes"),
            hidden_size=_require_int(args["hidden_size"], parser, name="hidden_size"),
            learning_rate=_require_float(
                args["learning_rate"], parser, name="learning_rate"
            ),
            steps=_require_int(args["steps"], parser, name="steps"),
            eval_interval=_require_int(
                args["eval_interval"], parser, name="eval_interval"
            ),
            seed=_require_int(args["seed"], parser, name="seed"),
        ),
        optimizer=_require_choice(
            args["optimizer"], ("adam", "adamw", "sgd"), parser, name="optimizer"
        ),
        device=_require_str(args["device"], parser, name="device"),
        match_resin_loss=_require_bool(
            args["match_resin_loss"], parser, name="match_resin_loss"
        ),
        benchmark_steps=_require_int(
            args["benchmark_steps"], parser, name="benchmark_steps"
        ),
        benchmark_warmup=_require_int(
            args["benchmark_warmup"], parser, name="benchmark_warmup"
        ),
    )


def main() -> None:
    cli = parse_cli()
    print(f"backend={cli.backend}", file=sys.stderr)

    match cli.backend:
        case "resin":
            run_resin(
                cli.train,
                benchmark_steps=cli.benchmark_steps,
                benchmark_warmup=cli.benchmark_warmup,
            )
        case "pytorch":
            run_pytorch(
                cli.train,
                optimizer_name=cli.optimizer,
                device_name=cli.device,
                match_resin_loss=cli.match_resin_loss,
                benchmark_steps=cli.benchmark_steps,
                benchmark_warmup=cli.benchmark_warmup,
            )
        case _:
            assert_never(cli.backend)


if __name__ == "__main__":
    main()
