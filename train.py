from __future__ import annotations

import argparse
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

if __package__:
    from .config_utils import (
        DEFAULT_CONFIG_PATH,
        get_section,
        load_config,
        resolve_config_path,
    )
    from .dataset import VISDRONE_CLASS_NAMES, create_dataloader
    from .evaluation import (
        DetectionMetricAccumulator,
        ValidationResult,
        append_evaluation_files,
        initialize_evaluation_files,
        xywh_to_xyxy,
    )
    from .load_pretrained import build_detection_model
    from .loss import YOLODetectionLoss, YOLOLossConfig
    from .predict_image import decode_predictions, postprocess
else:
    from config_utils import (
        DEFAULT_CONFIG_PATH,
        get_section,
        load_config,
        resolve_config_path,
    )
    from dataset import VISDRONE_CLASS_NAMES, create_dataloader
    from evaluation import (
        DetectionMetricAccumulator,
        ValidationResult,
        append_evaluation_files,
        initialize_evaluation_files,
        xywh_to_xyxy,
    )
    from load_pretrained import build_detection_model
    from loss import YOLODetectionLoss, YOLOLossConfig
    from predict_image import decode_predictions, postprocess


TRAINING_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = TRAINING_DIR / "data"
DEFAULT_PRETRAINED_PATH = TRAINING_DIR / "yolo_state_dict.pt"
DEFAULT_OUTPUT_DIR = TRAINING_DIR / "runs" / "visdrone_full"


DataMode = Literal["full", "devices"]
DeviceOrder = Literal["sequential", "shuffle", "rotate"]
BestMetric = Literal["val_loss", "f1", "map50", "map50_95"]


@dataclass(frozen=True)
class DeviceConfig:
    id: int
    name: str
    path: Path
    enabled: bool = True

    @property
    def images_dir(self) -> Path:
        return self.path / "images"

    @property
    def annotations_dir(self) -> Path:
        return self.path / "annotations"


@dataclass(frozen=True)
class TrainEpochResult:
    loss_sum: float
    image_count: int

    @property
    def mean_loss(self) -> float:
        return self.loss_sum / self.image_count


@dataclass
class TrainConfig:
    train_images: Path = DEFAULT_DATA_DIR / "VisDrone2019-DET-train" / "images"
    train_labels: Path = DEFAULT_DATA_DIR / "VisDrone2019-DET-train" / "annotations"
    val_images: Path = DEFAULT_DATA_DIR / "VisDrone2019-DET-val" / "images"
    val_labels: Path = DEFAULT_DATA_DIR / "VisDrone2019-DET-val" / "annotations"
    pretrained_path: Path = DEFAULT_PRETRAINED_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR

    data_mode: DataMode = "full"
    rounds: int = 50
    device_order: DeviceOrder = "sequential"
    local_epochs: int = 1
    devices: tuple[DeviceConfig, ...] = ()

    num_classes: int = len(VISDRONE_CLASS_NAMES)
    image_size: int = 640
    batch_size: int = 16
    num_workers: int = 4

    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-5
    weight_decay: float = 5e-4
    gradient_clip_norm: float = 10.0

    use_augmentation: bool = True
    use_amp: bool = True
    seed: int = 42

    evaluation_confidence_threshold: float = 0.001
    evaluation_nms_iou_threshold: float = 0.7
    evaluation_f1_confidence_threshold: float = 0.25
    evaluation_max_detections: int = 300
    evaluation_interval: int = 1
    best_metric: BestMetric = "map50_95"

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
    progress_label: str,
) -> TrainEpochResult:
    model.train()

    accumulated_loss = 0.0
    processed_images = 0

    progress_bar = tqdm(
        dataloader,
        desc=progress_label,
        unit="batch",
        dynamic_ncols=True,
    )
    for batch_index, (images, targets) in enumerate(progress_bar, start=1):
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
                f"Loss không hữu hạn tại {progress_label}, "
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

        progress_bar.set_postfix(
            {
                "loss/img": f"{accumulated_loss / processed_images:.4f}",
                "box": f"{loss_items['box_loss'].item():.3f}",
                "cls": f"{loss_items['cls_loss'].item():.3f}",
                "dfl": f"{loss_items['dfl_loss'].item():.3f}",
                "fg": int(loss_items["num_foreground"].item()),
            },
            refresh=False,
        )

    if processed_images == 0:
        raise RuntimeError("Train DataLoader không có ảnh.")

    return TrainEpochResult(
        loss_sum=accumulated_loss,
        image_count=processed_images,
    )


@torch.no_grad()
def validate(
    model: nn.Module,
    criterion: YOLODetectionLoss,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool,
    confidence_threshold: float,
    nms_iou_threshold: float,
    f1_confidence_threshold: float,
    max_detections: int,
) -> ValidationResult:
    model.eval()

    accumulated_loss = 0.0
    processed_images = 0
    metric_accumulator = DetectionMetricAccumulator(
        class_names=VISDRONE_CLASS_NAMES,
        f1_confidence_threshold=f1_confidence_threshold,
    )

    progress_bar = tqdm(
        dataloader,
        desc="Validation",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    )
    for images, targets in progress_bar:
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
        progress_bar.set_postfix(
            {"loss/img": f"{accumulated_loss / processed_images:.4f}"},
            refresh=False,
        )

        decoded_boxes, class_scores = decode_predictions(
            predictions,
            reg_max=criterion.config.reg_max,
            strides=criterion.config.strides,
        )
        image_height, image_width = images.shape[-2:]
        for image_index in range(images.shape[0]):
            detections = postprocess(
                boxes=decoded_boxes[image_index : image_index + 1],
                class_scores=class_scores[image_index : image_index + 1],
                confidence_threshold=confidence_threshold,
                iou_threshold=nms_iou_threshold,
                max_detections=max_detections,
            )
            target_mask = targets["batch_idx"] == image_index
            target_boxes = xywh_to_xyxy(
                targets["bboxes"][target_mask],
                width=image_width,
                height=image_height,
            )
            metric_accumulator.update(
                detections=detections,
                target_boxes=target_boxes,
                target_classes=targets["cls"][target_mask],
            )

    if processed_images == 0:
        raise RuntimeError("Validation DataLoader không có ảnh.")

    return ValidationResult(
        loss=accumulated_loss / processed_images,
        metrics=metric_accumulator.compute(),
    )


def create_training_dataloader(
    config: TrainConfig,
    images_dir: Path,
    annotations_dir: Path,
) -> torch.utils.data.DataLoader:
    return create_dataloader(
        images_dir=images_dir,
        labels_dir=annotations_dir,
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


def enabled_devices(config: TrainConfig) -> list[DeviceConfig]:
    return [device for device in config.devices if device.enabled]


def devices_for_round(
    devices: list[DeviceConfig],
    order: DeviceOrder,
    round_index: int,
    seed: int,
) -> list[DeviceConfig]:
    """Trả về thứ tự device, lấy danh sách trong config làm thứ tự gốc."""
    ordered_devices = list(devices)

    if order == "sequential":
        pass
    elif order == "shuffle":
        random.Random(seed + round_index).shuffle(ordered_devices)
    elif order == "rotate":
        if not ordered_devices:
            return ordered_devices
        offset = (round_index - 1) % len(ordered_devices)
        ordered_devices = (
            ordered_devices[offset:] + ordered_devices[:offset]
        )
    else:
        raise ValueError(f"Chiến lược thứ tự device không hợp lệ: {order}")

    return ordered_devices


def should_evaluate_round(
    round_index: int,
    total_rounds: int,
    interval: int,
) -> bool:
    """Evaluate theo chu kỳ và luôn evaluate round cuối cùng."""
    return round_index % interval == 0 or round_index == total_rounds


def serialize_checkpoint_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            key: serialize_checkpoint_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [serialize_checkpoint_value(item) for item in value]
    return value


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    round_index: int,
    device_order: tuple[int, ...],
    best_validation_loss: float,
    best_metric_name: BestMetric,
    best_metric_value: float,
    config: TrainConfig,
) -> None:
    """Lưu checkpoint qua file tạm để tránh checkpoint bị ghi dở."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    torch.save(
        {
            # Giữ key epoch để các công cụ đọc checkpoint cũ vẫn tương thích.
            "epoch": round_index,
            "round": round_index,
            "device_order": device_order,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_validation_loss": best_validation_loss,
            "best_metric_name": best_metric_name,
            "best_metric_value": best_metric_value,
            "config": serialize_checkpoint_value(asdict(config)),
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
    print("Model training mode: full model from pretrained")
    print(f"Training data mode: {config.data_mode}")

    selected_devices = enabled_devices(config)
    if config.data_mode == "devices":
        configured_order = ", ".join(
            str(device_config.id) for device_config in selected_devices
        )
        print(
            f"Device order strategy: {config.device_order} | "
            f"configured order: [{configured_order}] | "
            f"local epochs: {config.local_epochs}"
        )

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

    train_loader = None
    if config.data_mode == "full":
        train_loader = create_training_dataloader(
            config=config,
            images_dir=config.train_images,
            annotations_dir=config.train_labels,
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

    def cosine_learning_rate(round_index: int) -> float:
        if config.rounds <= 1:
            return 1.0

        round_index = min(round_index, config.rounds - 1)
        minimum_ratio = (
            config.min_learning_rate / config.learning_rate
        )
        cosine = (
            1.0
            + math.cos(
                math.pi * round_index / (config.rounds - 1)
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

    initialize_evaluation_files(config.output_dir)
    best_validation_loss = float("inf")
    best_metric_value = (
        float("inf") if config.best_metric == "val_loss" else float("-inf")
    )

    for round_index in range(1, config.rounds + 1):
        round_loss_sum = 0.0
        round_image_count = 0
        current_device_order: tuple[int, ...] = ()

        if config.data_mode == "full":
            if train_loader is None:
                raise RuntimeError("Full train DataLoader chưa được khởi tạo.")
            result = train_one_epoch(
                model=model,
                criterion=criterion,
                dataloader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                use_amp=amp_enabled,
                gradient_clip_norm=config.gradient_clip_norm,
                progress_label=(
                    f"Round {round_index:03d}/{config.rounds:03d}"
                ),
            )
            round_loss_sum += result.loss_sum
            round_image_count += result.image_count
        else:
            round_devices = devices_for_round(
                devices=selected_devices,
                order=config.device_order,
                round_index=round_index,
                seed=config.seed,
            )
            current_device_order = tuple(
                device_config.id for device_config in round_devices
            )
            print(
                f"Round {round_index:03d}/{config.rounds:03d} | "
                f"device order: {list(current_device_order)}"
            )

            for device_position, device_config in enumerate(
                round_devices,
                start=1,
            ):
                device_loader = create_training_dataloader(
                    config=config,
                    images_dir=device_config.images_dir,
                    annotations_dir=device_config.annotations_dir,
                )

                for local_epoch in range(1, config.local_epochs + 1):
                    progress_label = (
                        f"Round {round_index:03d}/{config.rounds:03d} | "
                        f"{device_config.name}[id={device_config.id}] "
                        f"({device_position}/{len(round_devices)}) | "
                        f"local epoch {local_epoch}/{config.local_epochs}"
                    )
                    result = train_one_epoch(
                        model=model,
                        criterion=criterion,
                        dataloader=device_loader,
                        optimizer=optimizer,
                        scaler=scaler,
                        device=device,
                        use_amp=amp_enabled,
                        gradient_clip_norm=config.gradient_clip_norm,
                        progress_label=progress_label,
                    )
                    round_loss_sum += result.loss_sum
                    round_image_count += result.image_count
                    print(
                        f"{progress_label} hoàn tất | "
                        f"loss/image {result.mean_loss:.6f}"
                    )

                del device_loader

        if round_image_count == 0:
            raise RuntimeError("Round không xử lý ảnh train nào.")
        train_loss = round_loss_sum / round_image_count
        current_learning_rate = optimizer.param_groups[0]["lr"]
        evaluate_this_round = should_evaluate_round(
            round_index=round_index,
            total_rounds=config.rounds,
            interval=config.evaluation_interval,
        )
        validation_result: ValidationResult | None = None
        is_best = False

        if evaluate_this_round:
            validation_result = validate(
                model=model,
                criterion=criterion,
                dataloader=validation_loader,
                device=device,
                use_amp=amp_enabled,
                confidence_threshold=(
                    config.evaluation_confidence_threshold
                ),
                nms_iou_threshold=config.evaluation_nms_iou_threshold,
                f1_confidence_threshold=(
                    config.evaluation_f1_confidence_threshold
                ),
                max_detections=config.evaluation_max_detections,
            )
            if validation_result.loss < best_validation_loss:
                best_validation_loss = validation_result.loss

            metric_values = {
                "val_loss": validation_result.loss,
                "f1": validation_result.f1,
                "map50": validation_result.map50,
                "map50_95": validation_result.map50_95,
            }
            current_metric_value = metric_values[config.best_metric]
            is_best = (
                current_metric_value < best_metric_value
                if config.best_metric == "val_loss"
                else current_metric_value > best_metric_value
            )
            if is_best:
                best_metric_value = current_metric_value

        scheduler.step()

        save_checkpoint(
            path=config.output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            round_index=round_index,
            device_order=current_device_order,
            best_validation_loss=best_validation_loss,
            best_metric_name=config.best_metric,
            best_metric_value=best_metric_value,
            config=config,
        )

        if is_best:
            save_checkpoint(
                path=config.output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                round_index=round_index,
                device_order=current_device_order,
                best_validation_loss=best_validation_loss,
                best_metric_name=config.best_metric,
                best_metric_value=best_metric_value,
                config=config,
            )

        if validation_result is not None:
            append_evaluation_files(
                output_dir=config.output_dir,
                round_index=round_index,
                train_loss=train_loss,
                result=validation_result,
                learning_rate=current_learning_rate,
                is_best=is_best,
            )
            print(
                f"Round {round_index:03d}/{config.rounds:03d} hoàn tất | "
                f"train loss/image {train_loss:.6f} | "
                f"val loss/image {validation_result.loss:.6f} | "
                f"P {validation_result.precision:.4f} | "
                f"R {validation_result.recall:.4f} | "
                f"F1 {validation_result.f1:.4f} | "
                f"mAP50 {validation_result.map50:.4f} | "
                f"mAP50-95 {validation_result.map50_95:.4f} | "
                f"lr {current_learning_rate:.8f} | "
                f"best {config.best_metric} {best_metric_value:.6f}"
            )
        else:
            print(
                f"Round {round_index:03d}/{config.rounds:03d} hoàn tất | "
                f"train loss/image {train_loss:.6f} | "
                f"lr {current_learning_rate:.8f} | "
                f"bỏ qua evaluation (interval="
                f"{config.evaluation_interval})"
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


def parse_device_configs(
    train_section: dict[str, Any],
    config_dir: Path,
) -> tuple[DeviceOrder, int, tuple[DeviceConfig, ...]]:
    devices_section = train_section.get("devices")
    if devices_section is None:
        return "sequential", 1, ()
    if not isinstance(devices_section, dict):
        raise ValueError("Config 'train.devices' phải là một mapping YAML.")

    order = cast(
        DeviceOrder,
        str(devices_section.get("order", "sequential")),
    )
    local_epochs = int(devices_section.get("local_epochs", 1))
    raw_items = devices_section.get("items", [])
    if not isinstance(raw_items, list):
        raise ValueError("Config 'train.devices.items' phải là một danh sách.")

    devices: list[DeviceConfig] = []
    for item_index, raw_device in enumerate(raw_items, start=1):
        field_prefix = f"train.devices.items[{item_index}]"
        if not isinstance(raw_device, dict):
            raise ValueError(f"Config '{field_prefix}' phải là một mapping.")

        try:
            device_id = int(raw_device["id"])
            device_name = str(raw_device["name"])
            raw_path = raw_device["path"]
        except KeyError as error:
            raise ValueError(
                f"Config '{field_prefix}' thiếu field {error.args[0]!r}."
            ) from error

        enabled = raw_device.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(
                f"Config '{field_prefix}.enabled' phải là true hoặc false."
            )

        device_path = resolve_config_path(
            raw_path,
            config_dir,
            f"{field_prefix}.path",
        )
        if device_path is None:
            raise ValueError(f"Config '{field_prefix}.path' không được trống.")

        devices.append(
            DeviceConfig(
                id=device_id,
                name=device_name,
                path=device_path,
                enabled=enabled,
            )
        )

    return order, local_epochs, tuple(devices)


def build_train_config(
    train_section: dict[str, Any],
    config_dir: Path,
) -> TrainConfig:
    loss_section = train_section.get("loss")
    if not isinstance(loss_section, dict):
        raise ValueError("config.yaml phải chứa section 'train.loss'.")
    evaluation_section = train_section.get("evaluation", {})
    if not isinstance(evaluation_section, dict):
        raise ValueError("Config 'train.evaluation' phải là một mapping YAML.")

    data_dir = resolve_config_path(
        train_section["data_dir"], config_dir, "train.data_dir"
    )
    pretrained_path = resolve_config_path(
        train_section["pretrained"], config_dir, "train.pretrained"
    )
    if data_dir is None or pretrained_path is None:
        raise ValueError("data_dir và pretrained không được để trống.")

    data_mode = cast(
        DataMode,
        str(train_section.get("data_mode", "full")),
    )
    configured_output_dir = resolve_config_path(
        train_section.get("output_dir"),
        config_dir,
        "train.output_dir",
        allow_none=True,
    )
    default_run_name = (
        "visdrone_devices" if data_mode == "devices" else "visdrone_full"
    )
    output_dir = configured_output_dir or (
        config_dir / "runs" / default_run_name
    )

    raw_rounds = train_section.get("rounds", train_section.get("epochs"))
    if raw_rounds is None:
        raise ValueError("Config 'train.rounds' không được để trống.")

    device_order, local_epochs, devices = parse_device_configs(
        train_section=train_section,
        config_dir=config_dir,
    )

    return TrainConfig(
        train_images=data_dir / "VisDrone2019-DET-train" / "images",
        train_labels=data_dir / "VisDrone2019-DET-train" / "annotations",
        val_images=data_dir / "VisDrone2019-DET-val" / "images",
        val_labels=data_dir / "VisDrone2019-DET-val" / "annotations",
        pretrained_path=pretrained_path,
        output_dir=output_dir,
        data_mode=data_mode,
        rounds=int(raw_rounds),
        device_order=device_order,
        local_epochs=local_epochs,
        devices=devices,
        num_classes=int(train_section["num_classes"]),
        image_size=int(train_section["image_size"]),
        batch_size=int(train_section["batch_size"]),
        num_workers=int(train_section["workers"]),
        learning_rate=float(train_section["learning_rate"]),
        min_learning_rate=float(train_section["min_learning_rate"]),
        weight_decay=float(train_section["weight_decay"]),
        gradient_clip_norm=float(train_section["gradient_clip_norm"]),
        use_augmentation=bool(train_section["augmentation"]),
        use_amp=bool(train_section["amp"]),
        seed=int(train_section["seed"]),
        evaluation_confidence_threshold=float(
            evaluation_section.get("confidence_threshold", 0.001)
        ),
        evaluation_nms_iou_threshold=float(
            evaluation_section.get("nms_iou_threshold", 0.7)
        ),
        evaluation_f1_confidence_threshold=float(
            evaluation_section.get("f1_confidence_threshold", 0.25)
        ),
        evaluation_max_detections=int(
            evaluation_section.get("max_detections", 300)
        ),
        evaluation_interval=int(evaluation_section.get("interval", 1)),
        best_metric=cast(
            BestMetric,
            str(evaluation_section.get("best_metric", "map50_95")),
        ),
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


def validate_config(config: TrainConfig) -> None:
    if config.data_mode not in ("full", "devices"):
        raise ValueError("train.data_mode phải là 'full' hoặc 'devices'.")
    if config.rounds <= 0:
        raise ValueError("rounds phải lớn hơn 0.")
    if config.device_order not in ("sequential", "shuffle", "rotate"):
        raise ValueError(
            "train.devices.order phải là 'sequential', 'shuffle' hoặc "
            "'rotate'."
        )
    if config.local_epochs <= 0:
        raise ValueError("train.devices.local_epochs phải lớn hơn 0.")

    device_ids = [device.id for device in config.devices]
    if len(device_ids) != len(set(device_ids)):
        raise ValueError("Device ID trong config không được trùng nhau.")
    device_names = [device.name for device in config.devices]
    if len(device_names) != len(set(device_names)):
        raise ValueError("Device name trong config không được trùng nhau.")
    for device in config.devices:
        if device.id <= 0:
            raise ValueError("Device ID phải lớn hơn 0.")
        if not device.name.strip():
            raise ValueError("Device name không được để trống.")

    if config.data_mode == "devices":
        selected_devices = enabled_devices(config)
        if not selected_devices:
            raise ValueError(
                "Chế độ devices yêu cầu ít nhất một device được bật."
            )
        for device in selected_devices:
            if not device.images_dir.is_dir():
                raise FileNotFoundError(
                    f"Không tìm thấy thư mục ảnh của {device.name}: "
                    f"{device.images_dir}"
                )
            if not device.annotations_dir.is_dir():
                raise FileNotFoundError(
                    f"Không tìm thấy annotation của {device.name}: "
                    f"{device.annotations_dir}"
                )

    if config.num_classes != len(VISDRONE_CLASS_NAMES):
        raise ValueError("VisDrone DET phải có num_classes=10.")
    if config.image_size <= 0 or config.image_size % 32 != 0:
        raise ValueError(
            "image_size phải lớn hơn 0 và chia hết cho 32."
        )
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
    if not 0.0 <= config.evaluation_confidence_threshold <= 1.0:
        raise ValueError(
            "train.evaluation.confidence_threshold phải nằm trong [0, 1]."
        )
    if not 0.0 <= config.evaluation_nms_iou_threshold <= 1.0:
        raise ValueError(
            "train.evaluation.nms_iou_threshold phải nằm trong [0, 1]."
        )
    if not 0.0 <= config.evaluation_f1_confidence_threshold <= 1.0:
        raise ValueError(
            "train.evaluation.f1_confidence_threshold phải nằm trong [0, 1]."
        )
    if config.evaluation_max_detections <= 0:
        raise ValueError("train.evaluation.max_detections phải lớn hơn 0.")
    if config.evaluation_interval <= 0:
        raise ValueError("train.evaluation.interval phải lớn hơn 0.")
    if config.best_metric not in ("val_loss", "f1", "map50", "map50_95"):
        raise ValueError(
            "train.evaluation.best_metric phải là 'val_loss', 'f1', "
            "'map50' hoặc 'map50_95'."
        )
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
    training_config = build_train_config(train_section, config_dir)
    print(f"Config: {args.config.expanduser().resolve()}")
    validate_config(training_config)
    train(training_config)
