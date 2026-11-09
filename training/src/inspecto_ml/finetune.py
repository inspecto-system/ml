"""Fine-tuning of pretrained networks on CPU (ML-04 CNN, ML-05 ViT; ADR-027).

Two phases: first only the classifier is trained on frozen features, then the last blocks are
unfrozen with a smaller learning rate. After every epoch the model is evaluated on the validation
part; the best epoch by macro-F1 is kept. Augmentation: random crop, horizontal flip, brightness.
"""

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from inspecto_ml.metrics import evaluate

CROP = 224
BRIGHTNESS = 0.2


@dataclass(frozen=True)
class Recipe:
    """How one network is fine-tuned: which parameters learn in each phase."""

    timm_name: str
    head: tuple[str, ...]
    last_blocks: tuple[str, ...]
    head_epochs: int = 1
    tune_epochs: int = 2
    batch_size: int = 32
    head_lr: float = 1e-3
    tune_lr: float = 1e-4
    label_smoothing: float = 0.1
    crop: int = CROP


RECIPES: dict[str, Recipe] = {
    "efficientnet_b0": Recipe(
        "efficientnet_b0.ra_in1k", ("classifier",), ("blocks.6", "conv_head", "bn2", "classifier")
    ),
    "convnext_tiny": Recipe("convnext_tiny.fb_in1k", ("head",), ("stages.3", "head")),
    "deit_small": Recipe(
        "deit_small_patch16_224.fb_in1k", ("head",), ("blocks.10", "blocks.11", "norm", "head")
    ),
}


@dataclass(frozen=True)
class Normalisation:
    """Mean and standard deviation of the pretrained network's input."""

    mean: tuple[float, ...]
    std: tuple[float, ...]

    def apply(self, images: torch.Tensor) -> torch.Tensor:
        """``(B, 3, H, W)`` float in 0..1 → normalised."""
        mean = torch.tensor(self.mean).view(1, 3, 1, 1)
        std = torch.tensor(self.std).view(1, 3, 1, 1)
        return (images - mean) / std


def set_trainable(model: nn.Module, prefixes: Sequence[str]) -> int:
    """Only parameters whose name starts with a prefix learn; returns their number."""
    count = 0
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(tuple(prefixes))
        count += parameter.numel() if parameter.requires_grad else 0
    return count


def augment(batch: np.ndarray, rng: np.random.Generator, crop: int = CROP) -> torch.Tensor:
    """Random crop, flip and brightness: ``(B, S, S, 3)`` uint8 → ``(B, 3, crop, crop)`` float."""
    side = batch.shape[1]
    out = np.empty((len(batch), crop, crop, 3), dtype=np.float32)
    for index, image in enumerate(batch):
        top, left = rng.integers(0, side - crop + 1, size=2)
        piece = image[top : top + crop, left : left + crop].astype(np.float32) / 255
        if rng.random() < 0.5:  # noqa: PLR2004 — a coin toss
            piece = piece[:, ::-1]
        out[index] = np.clip(piece * rng.uniform(1 - BRIGHTNESS, 1 + BRIGHTNESS), 0, 1)
    return torch.from_numpy(out).permute(0, 3, 1, 2)


def as_input(batch: np.ndarray, crop: int = CROP) -> torch.Tensor:
    """Centre crop of ``(B, S, S, 3)`` uint8 → ``(B, 3, crop, crop)`` float in 0..1."""
    side = batch.shape[1]
    start = (side - crop) // 2
    piece = batch[:, start : start + crop, start : start + crop].astype(np.float32) / 255
    return torch.from_numpy(np.ascontiguousarray(piece)).permute(0, 3, 1, 2)


@torch.inference_mode()
def predict(
    model: nn.Module,
    images: np.ndarray,
    normalisation: Normalisation,
    batch_size: int = 64,
    crop: int = CROP,
) -> np.ndarray:
    """Class probabilities ``(N, C)`` of the images."""
    model.eval()
    parts = [
        torch.softmax(
            model(normalisation.apply(as_input(images[start : start + batch_size], crop))), 1
        ).numpy()
        for start in range(0, len(images), batch_size)
    ]
    return np.concatenate(parts)


@dataclass
class TrainingData:
    """Training images ``(N, S, S, 3)`` with class indices, and validation images with targets."""

    images: np.ndarray
    labels: np.ndarray
    val_images: np.ndarray
    val_targets: list[str]
    classes: list[str]


@dataclass
class History:
    """What happened in every epoch."""

    epochs: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = 0
    best_val_macro_f1: float = -1.0


Progress = Callable[[str], None]


@dataclass
class Trainer:
    """Two-phase fine-tuning; the model ends with the weights of the best validation epoch."""

    model: nn.Module
    data: TrainingData
    recipe: Recipe
    normalisation: Normalisation
    progress: Progress
    seed: int = 2026

    def run(self) -> History:
        """Train both phases and keep the best epoch by validation macro-F1."""
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        loss = nn.CrossEntropyLoss(label_smoothing=self.recipe.label_smoothing)
        history = History()
        best_state = copy.deepcopy(self.model.state_dict())
        phases = [
            (self.recipe.head, self.recipe.head_lr, self.recipe.head_epochs),
            (self.recipe.last_blocks, self.recipe.tune_lr, self.recipe.tune_epochs),
        ]
        for prefixes, lr, epochs in phases:
            trainable = set_trainable(self.model, prefixes)
            learning = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(learning, lr=lr)
            for _ in range(epochs):
                mean_loss = self._epoch(optimizer, loss, rng)
                probabilities = predict(
                    self.model, self.data.val_images, self.normalisation, crop=self.recipe.crop
                )
                score = float(
                    evaluate(self.data.val_targets, probabilities, self.data.classes)["macro_f1"]
                )
                epoch = len(history.epochs) + 1
                history.epochs.append(
                    {
                        "epoch": epoch,
                        "trainable": trainable,
                        "lr": lr,
                        "loss": mean_loss,
                        "val_macro_f1": score,
                    }
                )
                self.progress(f"epoch {epoch}: loss {mean_loss:.4f}, val macro-F1 {score:.4f}")
                if score > history.best_val_macro_f1:
                    history.best_epoch, history.best_val_macro_f1 = epoch, score
                    best_state = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(best_state)
        return history

    def _epoch(
        self, optimizer: torch.optim.Optimizer, loss: nn.Module, rng: np.random.Generator
    ) -> float:
        self.model.train()
        order = rng.permutation(len(self.data.images))
        total = 0.0
        size = self.recipe.batch_size
        for start in range(0, len(order), size):
            rows = np.sort(order[start : start + size])  # sorted reads are faster on a memmap
            inputs = self.normalisation.apply(
                augment(np.asarray(self.data.images[rows]), rng, self.recipe.crop)
            )
            targets = torch.from_numpy(self.data.labels[rows]).long()
            optimizer.zero_grad()
            value = loss(self.model(inputs), targets)
            value.backward()
            optimizer.step()
            total += float(value) * len(rows)
        return total / len(order)
