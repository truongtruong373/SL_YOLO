from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

if __package__:
    from .model import MyYOLODetectionModel
else:
    from model import MyYOLODetectionModel


LoadMode = Literal[
    "backbone",
    "backbone_neck_partial",
    "backbone_neck",
]

DEFAULT_NUM_CLASSES = 10
DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent / "yolo_state_dict.pt"
)

# Block 23 là Detect head và không được load vì checkpoint COCO có 80 lớp,
# trong khi detection head của dataset đích có số lớp khác COCO.
MODE_LAST_BLOCK: dict[str, int] = {
    "backbone": 10,
    "backbone_neck_partial": 16,
    "backbone_neck": 22,
}


@dataclass(frozen=True)
class PretrainedLoadReport:
    checkpoint_path: Path
    mode: str
    selected_blocks: tuple[int, ...]
    loaded_tensor_count: int
    frozen_parameter_count: int

    def __str__(self) -> str:
        block_range = (
            f"{self.selected_blocks[0]}-{self.selected_blocks[-1]}"
        )
        return (
            f"Đã load {self.loaded_tensor_count} tensor từ "
            f"'{self.checkpoint_path}' với mode='{self.mode}' "
            f"(block {block_range}); "
            f"đã freeze {self.frozen_parameter_count} parameter."
        )


def _remove_common_prefix(key: str) -> str:
    """Chuẩn hóa key từ checkpoint DDP/torch.compile nếu có."""
    prefixes = ("module.", "_orig_mod.")
    changed = True

    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True

    return key


def _extract_state_dict(
    checkpoint: object,
) -> dict[str, torch.Tensor]:
    """
    Lấy state_dict từ checkpoint thuần hoặc checkpoint dạng wrapper.

    File hiện tại được tạo bởi ``torch.save(model.state_dict(), path)``,
    nhưng phần xử lý wrapper giúp hàm dùng được với checkpoint training.
    """
    candidate = checkpoint

    if isinstance(checkpoint, Mapping):
        for wrapper_key in ("state_dict", "model_state_dict"):
            wrapped = checkpoint.get(wrapper_key)
            if isinstance(wrapped, Mapping):
                candidate = wrapped
                break

    if not isinstance(candidate, Mapping):
        raise TypeError(
            "Checkpoint phải là state_dict hoặc chứa key "
            "'state_dict'/'model_state_dict'."
        )

    state_dict: dict[str, torch.Tensor] = {}
    for raw_key, value in candidate.items():
        if not isinstance(raw_key, str):
            continue
        if not isinstance(value, torch.Tensor):
            continue
        state_dict[_remove_common_prefix(raw_key)] = value

    if not state_dict:
        raise ValueError("Không tìm thấy tensor nào trong checkpoint.")

    return state_dict


def _get_selected_blocks(mode: LoadMode) -> tuple[int, ...]:
    if mode not in MODE_LAST_BLOCK:
        valid_modes = ", ".join(MODE_LAST_BLOCK)
        raise ValueError(
            f"Mode '{mode}' không hợp lệ. Chọn một trong: {valid_modes}."
        )

    return tuple(range(MODE_LAST_BLOCK[mode] + 1))


def _key_belongs_to_blocks(
    key: str,
    selected_blocks: tuple[int, ...],
) -> bool:
    return any(
        key.startswith(f"model.{block_index}.")
        for block_index in selected_blocks
    )


def load_yolo11n_pretrained(
    model: nn.Module,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    mode: LoadMode = "backbone_neck",
    freeze_loaded: bool = False,
    strict: bool = True,
) -> PretrainedLoadReport:
    """
    Load một phần trọng số YOLO11n Detect vào model detection đích.

    Args:
        model:
            ``MyYOLODetectionModel`` hoặc model có state_dict tương thích.
        checkpoint_path:
            File ``yolo_state_dict.pt`` pretrained trên COCO.
        mode:
            - ``backbone``: load block 0-10.
            - ``backbone_neck_partial``: load block 0-16.
            - ``backbone_neck``: load block 0-22.
        freeze_loaded:
            Nếu True, các parameter vừa load sẽ không được cập nhật gradient.
        strict:
            Nếu True, báo lỗi khi key hoặc shape của vùng được chọn không khớp.

    Detection head block 23 luôn giữ nguyên để học các lớp của dataset đích.
    """
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint: {checkpoint_path}"
        )

    selected_blocks = _get_selected_blocks(mode)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    source_state = _extract_state_dict(checkpoint)
    target_state = model.state_dict()

    selected_source = {
        key: value
        for key, value in source_state.items()
        if _key_belongs_to_blocks(key, selected_blocks)
    }
    selected_target_keys = {
        key
        for key in target_state
        if _key_belongs_to_blocks(key, selected_blocks)
    }

    compatible_state: dict[str, torch.Tensor] = {}
    unexpected_keys: list[str] = []
    shape_mismatches: list[str] = []

    for key, source_tensor in selected_source.items():
        target_tensor = target_state.get(key)
        if target_tensor is None:
            unexpected_keys.append(key)
            continue

        if source_tensor.shape != target_tensor.shape:
            shape_mismatches.append(
                f"{key}: checkpoint={tuple(source_tensor.shape)}, "
                f"model={tuple(target_tensor.shape)}"
            )
            continue

        compatible_state[key] = source_tensor

    missing_keys = sorted(
        selected_target_keys.difference(compatible_state)
    )

    if strict and (
        missing_keys
        or unexpected_keys
        or shape_mismatches
    ):
        problems: list[str] = []

        if missing_keys:
            problems.append(
                "Thiếu key trong checkpoint:\n  - "
                + "\n  - ".join(missing_keys)
            )

        if unexpected_keys:
            problems.append(
                "Key checkpoint không có trong model:\n  - "
                + "\n  - ".join(sorted(unexpected_keys))
            )

        if shape_mismatches:
            problems.append(
                "Tensor sai shape:\n  - "
                + "\n  - ".join(shape_mismatches)
            )

        raise RuntimeError(
            "Không thể load pretrained theo chế độ strict:\n"
            + "\n".join(problems)
        )

    if not compatible_state:
        raise RuntimeError(
            "Không có tensor tương thích nào để load. "
            "Hãy kiểm tra checkpoint và kiến trúc model."
        )

    model.load_state_dict(compatible_state, strict=False)

    frozen_parameter_count = 0
    if freeze_loaded:
        loaded_keys = set(compatible_state)
        for parameter_name, parameter in model.named_parameters():
            if parameter_name in loaded_keys:
                parameter.requires_grad_(False)
                frozen_parameter_count += parameter.numel()

    return PretrainedLoadReport(
        checkpoint_path=checkpoint_path,
        mode=mode,
        selected_blocks=selected_blocks,
        loaded_tensor_count=len(compatible_state),
        frozen_parameter_count=frozen_parameter_count,
    )


def build_detection_model(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    mode: LoadMode = "backbone_neck",
    freeze_loaded: bool = False,
    strict: bool = True,
    num_classes: int = DEFAULT_NUM_CLASSES,
) -> tuple[MyYOLODetectionModel, PretrainedLoadReport]:
    """Khởi tạo model cho dataset đích và load phần pretrained được chọn."""
    if num_classes <= 0:
        raise ValueError("num_classes phải lớn hơn 0.")
    model = MyYOLODetectionModel(nc=num_classes)
    report = load_yolo11n_pretrained(
        model=model,
        checkpoint_path=checkpoint_path,
        mode=mode,
        freeze_loaded=freeze_loaded,
        strict=strict,
    )
    return model, report


# Giữ alias để code cũ không bị lỗi import.
build_ppe_model = build_detection_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load pretrained YOLO11n Detect cho model đích."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Đường dẫn tới yolo_state_dict.pt.",
    )
    parser.add_argument(
        "--mode",
        choices=tuple(MODE_LAST_BLOCK),
        default="backbone_neck",
        help="Phần model cần load pretrained.",
    )
    parser.add_argument(
        "--freeze-loaded",
        action="store_true",
        help="Freeze các parameter được load.",
    )
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Bỏ qua key hoặc tensor không tương thích.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    detection_model, load_report = build_detection_model(
        checkpoint_path=args.checkpoint,
        mode=args.mode,
        freeze_loaded=args.freeze_loaded,
        strict=not args.non_strict,
    )
    print(load_report)
    print(
        "Số parameter có thể train:",
        sum(
            parameter.numel()
            for parameter in detection_model.parameters()
            if parameter.requires_grad
        ),
    )
