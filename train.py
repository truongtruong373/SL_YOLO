from __future__ import annotations

import argparse
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

if __package__:
    from .config_utils import (
        DEFAULT_CONFIG_PATH,
        get_section,
        load_config,
        resolve_config_path,
    )
    from .dataset import VISDRONE_CLASS_NAMES, create_dataloader
    from .load_pretrained import build_detection_model
    from .loss import YOLODetectionLoss, YOLOLossConfig
else:
    from config_utils import (
        DEFAULT_CONFIG_PATH,
        get_section,
        load_config,
        resolve_config_path,
    )
    from dataset import VISDRONE_CLASS_NAMES, create_dataloader
    from load_pretrained import build_detection_model
    from loss import YOLODetectionLoss, YOLOLossConfig


TRAINING_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = TRAINING_DIR / "data"
DEFAULT_PRETRAINED_PATH = TRAINING_DIR / "yolo_state_dict.pt"
DEFAULT_OUTPUT_DIR = TRAINING_DIR / "runs" / "visdrone_full"


@dataclass
class TrainConfig:
    train_images: Path = DEFAULT_DATA_DIR / "VisDrone2019-DET-train" / "images"
    train_labels: Path = DEFAULT_DATA_DIR / "VisDrone2019-DET-train" / "annotations"
    val_images: Path = DEFAULT_DATA_DIR / "VisDrone2019-DET-val" / "images"
    val_labels: Path = DEFAULT_DATA_DIR / "VisDrone2019-DET-val" / "annotations"
    pretrained_path: Path = DEFAULT_PRETRAINED_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR

    num_classes: int = len(VISDRONE_CLASS_NAMES)
    image_size: int = 640
    epochs: int = 50
    batch_size: int = 16
    num_workers: int = 4

    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-5
    weight_decay: float = 5e-4
    gradient_clip_norm: float = 10.0

    use_augmentation: bool = True
    use_amp: bool = True
    seed: int = 42
    log_interval: int = 20

    reg_max: int = 16
    strides: tuple[int, ...] = (8, 16, 32)
    box_gain: float = 7.5
    cls_gain: float = 0.5
    dfl_gain: float = 1.5
    assigner_topk: int = 10
    assigner_alpha: float = 0.5
    assigner_beta: float = 6.0
    scale_loss_by_batch: bool = True


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_targets_to_device(
    targets: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device=device, non_blocking=True)
        for key, value in targets.items()
    }


def train_one_epoch(
    model: nn.Module,
    criterion: YOLODetectionLoss,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    gradient_clip_norm: float,
    log_interval: int,
    epoch: int,
) -> float:
    model.train()

    accumulated_loss = 0.0
    processed_images = 0

    for batch_index, (images, targets) in enumerate(dataloader, start=1):
        images = images.to(device=device, non_blocking=True)
        targets = move_targets_to_device(targets, device)
        current_batch_size = images.shape[0]

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            predictions = model(images)
            total_loss, loss_items = criterion(predictions, targets)

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f"Loss không hữu hạn tại epoch={epoch}, "
                f"batch={batch_index}: {total_loss.detach().item()}"
            )

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)

        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                (
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
                max_norm=gradient_clip_norm,
            )

        scaler.step(optimizer)
        scaler.update()

        accumulated_loss += total_loss.detach().item()
        processed_images += current_batch_size

        if batch_index % log_interval == 0:
            mean_loss = accumulated_loss / processed_images
            print(
                f"Epoch {epoch:03d} | "
                f"batch {batch_index:04d}/{len(dataloader):04d} | "
                f"loss/image {mean_loss:.6f} | "
                f"box {loss_items['box_loss'].item():.4f} | "
                f"cls {loss_items['cls_loss'].item():.4f} | "
                f"dfl {loss_items['dfl_loss'].item():.4f} | "
                f"foreground "
                f"{int(loss_items['num_foreground'].item())}"
            )

    if processed_images == 0:
        raise RuntimeError("Train DataLoader không có ảnh.")

    return accumulated_loss / processed_images


@torch.no_grad()
def validate(
    model: nn.Module,
    criterion: YOLODetectionLoss,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.eval()

    accumulated_loss = 0.0
    processed_images = 0

    for images, targets in dataloader:
        images = images.to(device=device, non_blocking=True)
        targets = move_targets_to_device(targets, device)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            predictions = model(images)
            total_loss, _ = criterion(predictions, targets)

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                "Validation loss chứa NaN hoặc Inf."
            )

        accumulated_loss += total_loss.item()
        processed_images += images.shape[0]

    if processed_images == 0:
        raise RuntimeError("Validation DataLoader không có ảnh.")

    return accumulated_loss / processed_images


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_validation_loss: float,
    config: TrainConfig,
) -> None:
    """Lưu checkpoint qua file tạm để tránh checkpoint bị ghi dở."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_validation_loss": best_validation_loss,
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config).items()
            },
        },
        temporary_path,
    )
    temporary_path.replace(path)


def train(config: TrainConfig) -> None:
    set_random_seed(config.seed)
    device = choose_device()
    amp_enabled = config.use_amp and device.type == "cuda"

    config.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"AMP: {amp_enabled}")

    print("Training mode: full model from pretrained")

    model, load_report = build_detection_model(
        checkpoint_path=config.pretrained_path,
        mode="backbone_neck",
        freeze_loaded=False,
        strict=True,
        num_classes=config.num_classes,
    )
    model = model.to(device)
    print(load_report)

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("Model không còn parameter nào để train.")

    total_parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    trainable_parameter_count = sum(
        parameter.numel() for parameter in trainable_parameters
    )
    print(
        f"Trainable parameters: {trainable_parameter_count:,}/"
        f"{total_parameter_count:,}"
    )

    criterion = YOLODetectionLoss(
        YOLOLossConfig(
            num_classes=config.num_classes,
            reg_max=config.reg_max,
            strides=config.strides,
            box_gain=config.box_gain,
            cls_gain=config.cls_gain,
            dfl_gain=config.dfl_gain,
            assigner_topk=config.assigner_topk,
            assigner_alpha=config.assigner_alpha,
            assigner_beta=config.assigner_beta,
            scale_loss_by_batch=config.scale_loss_by_batch,
        )
    ).to(device)

    train_loader = create_dataloader(
        images_dir=config.train_images,
        labels_dir=config.train_labels,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_classes=config.num_classes,
        augment=config.use_augmentation,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=False,
        annotation_format="visdrone",
        strict_labels=True,
    )
    validation_loader = create_dataloader(
        images_dir=config.val_images,
        labels_dir=config.val_labels,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_classes=config.num_classes,
        augment=False,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
        annotation_format="visdrone",
        strict_labels=True,
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    def cosine_learning_rate(epoch_index: int) -> float:
        if config.epochs <= 1:
            return 1.0

        epoch_index = min(epoch_index, config.epochs - 1)
        minimum_ratio = (
            config.min_learning_rate / config.learning_rate
        )
        cosine = (
            1.0
            + math.cos(
                math.pi * epoch_index / (config.epochs - 1)
            )
        ) / 2.0
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=cosine_learning_rate,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    best_validation_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            criterion=criterion,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=amp_enabled,
            gradient_clip_norm=config.gradient_clip_norm,
            log_interval=config.log_interval,
            epoch=epoch,
        )
        validation_loss = validate(
            model=model,
            criterion=criterion,
            dataloader=validation_loader,
            device=device,
            use_amp=amp_enabled,
        )

        current_learning_rate = optimizer.param_groups[0]["lr"]
        is_best = validation_loss < best_validation_loss
        if is_best:
            best_validation_loss = validation_loss

        scheduler.step()

        save_checkpoint(
            path=config.output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_validation_loss=best_validation_loss,
            config=config,
        )

        if is_best:
            save_checkpoint(
                path=config.output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_validation_loss=best_validation_loss,
                config=config,
            )

        print(
            f"Epoch {epoch:03d}/{config.epochs:03d} hoàn tất | "
            f"train loss/image {train_loss:.6f} | "
            f"val loss/image {validation_loss:.6f} | "
            f"lr {current_learning_rate:.8f} | "
            f"best val {best_validation_loss:.6f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune YOLO11n trên VisDrone DET theo config YAML."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Đường dẫn tới config.yaml.",
    )
    return parser.parse_args()


def validate_config(config: TrainConfig) -> None:
    if config.num_classes != len(VISDRONE_CLASS_NAMES):
        raise ValueError("VisDrone DET phải có num_classes=10.")
    if config.image_size <= 0 or config.image_size % 32 != 0:
        raise ValueError(
            "image_size phải lớn hơn 0 và chia hết cho 32."
        )
    if config.epochs <= 0:
        raise ValueError("epochs phải lớn hơn 0.")
    if config.batch_size <= 0:
        raise ValueError("batch_size phải lớn hơn 0.")
    if config.num_workers < 0:
        raise ValueError("workers không được âm.")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate phải lớn hơn 0.")
    if config.min_learning_rate < 0:
        raise ValueError("min_learning_rate không được âm.")
    if config.min_learning_rate > config.learning_rate:
        raise ValueError(
            "min_learning_rate không được lớn hơn learning_rate."
        )
    if config.weight_decay < 0:
        raise ValueError("weight_decay không được âm.")
    if config.gradient_clip_norm < 0:
        raise ValueError("gradient_clip_norm không được âm.")
    if config.log_interval <= 0:
        raise ValueError("log_interval phải lớn hơn 0.")
    if config.reg_max != 16:
        raise ValueError(
            "Kiến trúc Detect hiện tại yêu cầu train.loss.reg_max=16."
        )
    if config.strides != (8, 16, 32):
        raise ValueError(
            "Kiến trúc Detect hiện tại yêu cầu "
            "train.loss.strides=[8, 16, 32]."
        )


if __name__ == "__main__":
    args = parse_args()
    raw_config, config_dir = load_config(args.config)
    train_section = get_section(raw_config, "train")
    loss_section = train_section.get("loss")
    if not isinstance(loss_section, dict):
        raise ValueError("config.yaml phải chứa section 'train.loss'.")

    data_dir = resolve_config_path(
        train_section["data_dir"], config_dir, "train.data_dir"
    )
    pretrained_path = resolve_config_path(
        train_section["pretrained"], config_dir, "train.pretrained"
    )
    configured_output_dir = resolve_config_path(
        train_section.get("output_dir"),
        config_dir,
        "train.output_dir",
        allow_none=True,
    )
    output_dir = configured_output_dir or (
        config_dir / "runs" / "visdrone_full"
    )

    training_config = TrainConfig(
        train_images=data_dir / "VisDrone2019-DET-train" / "images",
        train_labels=data_dir / "VisDrone2019-DET-train" / "annotations",
        val_images=data_dir / "VisDrone2019-DET-val" / "images",
        val_labels=data_dir / "VisDrone2019-DET-val" / "annotations",
        pretrained_path=pretrained_path,
        output_dir=output_dir,
        num_classes=int(train_section["num_classes"]),
        image_size=int(train_section["image_size"]),
        epochs=int(train_section["epochs"]),
        batch_size=int(train_section["batch_size"]),
        num_workers=int(train_section["workers"]),
        learning_rate=float(train_section["learning_rate"]),
        min_learning_rate=float(train_section["min_learning_rate"]),
        weight_decay=float(train_section["weight_decay"]),
        gradient_clip_norm=float(train_section["gradient_clip_norm"]),
        use_augmentation=bool(train_section["augmentation"]),
        use_amp=bool(train_section["amp"]),
        seed=int(train_section["seed"]),
        log_interval=int(train_section["log_interval"]),
        reg_max=int(loss_section["reg_max"]),
        strides=tuple(int(value) for value in loss_section["strides"]),
        box_gain=float(loss_section["box_gain"]),
        cls_gain=float(loss_section["cls_gain"]),
        dfl_gain=float(loss_section["dfl_gain"]),
        assigner_topk=int(loss_section["assigner_topk"]),
        assigner_alpha=float(loss_section["assigner_alpha"]),
        assigner_beta=float(loss_section["assigner_beta"]),
        scale_loss_by_batch=bool(loss_section["scale_loss_by_batch"]),
    )
    print(f"Config: {args.config.expanduser().resolve()}")
    validate_config(training_config)
    train(training_config)
