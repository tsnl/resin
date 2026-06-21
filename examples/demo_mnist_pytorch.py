# pyright: reportMissingImports=false
# pyright: reportImplicitRelativeImport=false
# pyright: reportAny=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownParameterType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportExplicitAny=false
# pyright: reportUntypedBaseClass=false
# pyright: reportUntypedFunctionDecorator=false
# pyright: reportUnusedCallResult=false
# pyright: reportImplicitStringConcatenation=false
# pyright: reportUnannotatedClassAttribute=false
"""PyTorch backend for the MNIST MLP demo."""

import random
import struct
import sys
import time

from demo_mnist_common import (
    ClassificationMetrics,
    TrainConfig,
    argmax,
    iter_test_batches,
    metrics_from_confusion,
    next_batch,
    print_eval_metrics,
    update_confusion,
)
from resin.dataset import MnistDataLoader, MnistDataset


def run_pytorch(
    config: TrainConfig,
    *,
    optimizer_name: str,
    device_name: str,
    match_resin_loss: bool,
    benchmark_steps: int,
    benchmark_warmup: int,
) -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def bytes_to_image_tensor(
        image_bytes: bytes,
        *,
        batch_size: int,
        image_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        values = struct.unpack(f"<{batch_size * image_size}f", image_bytes)
        return torch.tensor(values, dtype=torch.float32, device=device).reshape(
            batch_size,
            image_size,
        )

    def bytes_to_label_tensor(
        label_bytes: bytes,
        *,
        batch_size: int,
        num_classes: int,
        device: torch.device,
    ) -> torch.Tensor:
        values = struct.unpack(f"<{batch_size * num_classes}f", label_bytes)
        return torch.tensor(values, dtype=torch.float32, device=device).reshape(
            batch_size,
            num_classes,
        )

    def resolve_device(device_name: str) -> torch.device:
        if device_name == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device_name)

    def sync_device(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    class TorchMnistMlp(nn.Module):
        def __init__(
            self, *, input_size: int, hidden_size: int, output_size: int
        ) -> None:
            super().__init__()
            self.l1 = nn.Linear(input_size, hidden_size)
            self.l2 = nn.Linear(hidden_size, hidden_size)
            self.l3 = nn.Linear(hidden_size, output_size)
            self._init_params()

        def _init_params(self) -> None:
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.uniform_(module.weight, -0.1, 0.1)
                    if module.bias is not None:
                        nn.init.uniform_(module.bias, -0.1, 0.1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.l1(x))
            x = F.relu(self.l2(x))
            return F.softmax(self.l3(x), dim=-1)

        def logits(self, x: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.l1(x))
            x = F.relu(self.l2(x))
            return self.l3(x)

    def cross_entropy_mean(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return -(labels * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()

    def cross_entropy_mean_from_logits(
        logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        return -(labels * log_probs).sum(dim=-1).mean()

    def build_optimizer(
        model: TorchMnistMlp,
    ) -> torch.optim.Optimizer:
        params = model.parameters()
        if optimizer_name == "adam":
            return torch.optim.Adam(params, lr=config.learning_rate)
        if optimizer_name == "adamw":
            return torch.optim.AdamW(params, lr=config.learning_rate)
        if optimizer_name == "sgd":
            return torch.optim.SGD(params, lr=config.learning_rate)
        raise ValueError(f"unsupported optimizer: {optimizer_name}")

    @torch.no_grad()
    def evaluate_test_set(
        model: TorchMnistMlp,
        *,
        test_dataset: MnistDataset,
        device: torch.device,
    ) -> ClassificationMetrics:
        model.eval()
        confusion = [[0] * config.num_classes for _ in range(config.num_classes)]
        for image_bytes, labels in iter_test_batches(test_dataset, config.batch_size):
            images = bytes_to_image_tensor(
                image_bytes,
                batch_size=config.batch_size,
                image_size=test_dataset.image_size,
                device=device,
            )
            probs_flat = model(images).detach().cpu().reshape(-1).tolist()
            for sample_index, true_label in enumerate(labels):
                offset = sample_index * config.num_classes
                predicted_label = argmax(
                    probs_flat[offset : offset + config.num_classes],
                    config.num_classes,
                )
                update_confusion(confusion, true_label, predicted_label)
        model.train()
        return metrics_from_confusion(confusion)

    use_logits_loss = not match_resin_loss

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = resolve_device(device_name)

    train_dataset = MnistDataset.load("train")
    test_dataset = MnistDataset.load("test")
    data_loader = MnistDataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
        drop_last=True,
    )
    epoch_batches = data_loader.iter_epochs()

    model = TorchMnistMlp(
        input_size=config.img_size**2,
        hidden_size=config.hidden_size,
        output_size=config.num_classes,
    ).to(device)
    optimizer = build_optimizer(model)

    def train_step(image_bytes: bytes, label_bytes: bytes) -> float:
        images = bytes_to_image_tensor(
            image_bytes,
            batch_size=config.batch_size,
            image_size=config.img_size * config.img_size,
            device=device,
        )
        labels = bytes_to_label_tensor(
            label_bytes,
            batch_size=config.batch_size,
            num_classes=config.num_classes,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        if use_logits_loss:
            loss = cross_entropy_mean_from_logits(model.logits(images), labels)
        else:
            loss = cross_entropy_mean(model(images), labels)
        loss.backward()
        optimizer.step()
        return float(loss.detach().cpu())

    if benchmark_steps > 0:
        batch_iter = next(epoch_batches)
        timings: list[float] = []
        for step in range(benchmark_warmup + benchmark_steps):
            (image_bytes, label_bytes), batch_iter = next_batch(
                batch_iter, epoch_batches
            )
            sync_device(device)
            start = time.perf_counter()
            train_step(image_bytes, label_bytes)
            sync_device(device)
            elapsed = time.perf_counter() - start
            if step >= benchmark_warmup:
                timings.append(elapsed)
        mean_step_s = sum(timings) / len(timings)
        print(
            f"pytorch benchmark ({optimizer_name}, {device.type}): "
            f"mean_train_step={mean_step_s * 1e3:.3f} ms "
            f"({1.0 / mean_step_s:.1f} steps/s)",
            file=sys.stderr,
        )
        return

    step = 0
    batch_iter = next(epoch_batches)
    while step < config.steps:
        (image_bytes, label_bytes), batch_iter = next_batch(batch_iter, epoch_batches)
        loss = train_step(image_bytes, label_bytes)

        if step % config.eval_interval == 0 or step == config.steps - 1:
            print(f"step {step}: loss={loss:.6f}", file=sys.stderr)
            metrics = evaluate_test_set(
                model,
                test_dataset=test_dataset,
                device=device,
            )
            print_eval_metrics(step, metrics)
        step += 1