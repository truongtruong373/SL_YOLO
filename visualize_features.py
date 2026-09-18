from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import cv2

# Tránh Matplotlib thử ghi cache vào home trong môi trường server/container.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "sl_yolo_matplotlib"),
)
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config_utils import (
    DEFAULT_CONFIG_PATH,
    get_section,
    load_config,
    resolve_config_path,
)
from predict_image import choose_device, load_trained_model, preprocess_image


DEFAULT_LAYERS = (3, 4, 6, 10, 13, 16, 19, 22)


def normalize_map(values: np.ndarray) -> np.ndarray:
    """Chuẩn hóa robust về [0, 1] để outlier không làm tối ảnh."""
    lower, upper = np.percentile(values, (1.0, 99.0))
    if upper <= lower:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lower) / (upper - lower), 0.0, 1.0).astype(
        np.float32,
        copy=False,
    )


def select_informative_channels(
    activation: torch.Tensor,
    channel_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chọn channel có độ lệch chuẩn không gian lớn nhất."""
    flattened = activation.flatten(start_dim=1)
    scores = flattened.std(dim=1, unbiased=False)
    selected_count = min(channel_count, activation.shape[0])
    indices = scores.topk(selected_count).indices
    return indices, activation[indices]


def capture_activations(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    layer_indices: tuple[int, ...],
) -> dict[int, torch.Tensor]:
    if len(layer_indices) != len(set(layer_indices)):
        raise ValueError("Danh sách layer không được chứa phần tử trùng nhau.")

    model_layers = getattr(model, "model", None)
    if not isinstance(model_layers, torch.nn.Sequential):
        raise TypeError("Model không có thuộc tính 'model' dạng nn.Sequential.")

    for layer_index in layer_indices:
        if not 0 <= layer_index < len(model_layers):
            raise ValueError(
                f"Layer {layer_index} nằm ngoài phạm vi "
                f"[0, {len(model_layers) - 1}]."
            )

    activations: dict[int, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(layer_index: int):
        def hook(
            _module: torch.nn.Module,
            _inputs: tuple[Any, ...],
            output: Any,
        ) -> None:
            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    f"Output layer {layer_index} không phải Tensor."
                )
            if output.ndim != 4:
                raise ValueError(
                    f"Output layer {layer_index} phải có shape [B,C,H,W], "
                    f"nhận được {tuple(output.shape)}."
                )
            activations[layer_index] = output[0].detach().float().cpu()

        return hook

    try:
        for layer_index in layer_indices:
            handles.append(
                model_layers[layer_index].register_forward_hook(
                    make_hook(layer_index)
                )
            )
        with torch.inference_mode():
            model(image_tensor)
    finally:
        for handle in handles:
            handle.remove()

    missing = set(layer_indices) - set(activations)
    if missing:
        raise RuntimeError(f"Không capture được các layer: {sorted(missing)}")
    return activations


def save_layer_comparison(
    image_rgb: np.ndarray,
    activation: torch.Tensor,
    layer_index: int,
    output_path: Path,
    colormap: str,
) -> None:
    aggregate = activation.abs().mean(dim=0).numpy()
    normalized = normalize_map(aggregate)
    resized = cv2.resize(
        normalized,
        (image_rgb.shape[1], image_rgb.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Model input")
    axes[1].imshow(normalized, cmap=colormap, vmin=0.0, vmax=1.0)
    axes[1].set_title(
        f"Layer {layer_index}: mean(|activation|)\n"
        f"shape={tuple(activation.shape)}"
    )
    axes[2].imshow(image_rgb)
    axes[2].imshow(resized, cmap=colormap, alpha=0.5, vmin=0.0, vmax=1.0)
    axes[2].set_title("Activation overlay")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_channel_grid(
    activation: torch.Tensor,
    layer_index: int,
    channel_count: int,
    output_path: Path,
    colormap: str,
) -> None:
    indices, channels = select_informative_channels(
        activation,
        channel_count=channel_count,
    )
    columns = min(4, channels.shape[0])
    rows = math.ceil(channels.shape[0] / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.2 * columns, 3.2 * rows),
        squeeze=False,
    )
    for plot_index, axis in enumerate(axes.flat):
        axis.axis("off")
        if plot_index >= channels.shape[0]:
            continue
        channel = normalize_map(channels[plot_index].numpy())
        channel_index = int(indices[plot_index].item())
        axis.imshow(channel, cmap=colormap, vmin=0.0, vmax=1.0)
        axis.set_title(f"channel {channel_index}")
    figure.suptitle(
        f"Layer {layer_index} — channels có biến thiên không gian lớn nhất"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_overview(
    image_rgb: np.ndarray,
    activations: dict[int, torch.Tensor],
    output_path: Path,
    colormap: str,
) -> None:
    panel_count = len(activations) + 1
    columns = 3
    rows = math.ceil(panel_count / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5 * columns, 4.5 * rows),
        squeeze=False,
    )
    flat_axes = axes.flat
    first_axis = next(flat_axes)
    first_axis.imshow(image_rgb)
    first_axis.set_title("Đầu vào", fontsize=18)
    first_axis.axis("off")

    for axis, (layer_index, activation) in zip(
        flat_axes,
        activations.items(),
        strict=False,
    ):
        aggregate = normalize_map(activation.abs().mean(dim=0).numpy())
        axis.imshow(aggregate, cmap=colormap, vmin=0.0, vmax=1.0)
        axis.set_title(f"Khối {layer_index}", fontsize=18)
        axis.axis("off")
    for axis in flat_axes:
        axis.axis("off")

    # figure.suptitle("So sánh input và mean absolute feature maps")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def visualize_features(
    checkpoint_path: Path,
    image_path: Path,
    output_dir: Path,
    image_size: int,
    layer_indices: tuple[int, ...],
    channels_per_layer: int,
    colormap: str,
    device: torch.device,
) -> None:
    model = load_trained_model(checkpoint_path, device)
    image_tensor, _, _, _ = preprocess_image(
        image_path=image_path,
        image_size=image_size,
        device=device,
    )
    image_rgb = (
        image_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
    )
    activations = capture_activations(model, image_tensor, layer_indices)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_overview(
        image_rgb=image_rgb,
        activations=activations,
        output_path=output_dir / "overview.png",
        colormap=colormap,
    )

    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "image": str(image_path),
        "image_size": image_size,
        "layers": {},
    }
    for layer_index, activation in activations.items():
        save_layer_comparison(
            image_rgb=image_rgb,
            activation=activation,
            layer_index=layer_index,
            output_path=output_dir / f"layer_{layer_index:02d}_comparison.png",
            colormap=colormap,
        )
        save_channel_grid(
            activation=activation,
            layer_index=layer_index,
            channel_count=channels_per_layer,
            output_path=output_dir / f"layer_{layer_index:02d}_channels.png",
            colormap=colormap,
        )
        summary["layers"][str(layer_index)] = {
            "shape": list(activation.shape),
            "min": float(activation.min()),
            "max": float(activation.max()),
            "mean": float(activation.mean()),
            "std": float(activation.std(unbiased=False)),
        }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vẽ feature map của các layer để đánh giá privacy."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Đường dẫn tới config.yaml.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_config, config_dir = load_config(args.config)
    section = get_section(raw_config, "visualize_features")
    checkpoint_path = resolve_config_path(
        section["checkpoint"], config_dir, "visualize_features.checkpoint"
    )
    image_path = resolve_config_path(
        section["image"], config_dir, "visualize_features.image"
    )
    output_dir = resolve_config_path(
        section["output_dir"], config_dir, "visualize_features.output_dir"
    )
    if checkpoint_path is None or image_path is None or output_dir is None:
        raise ValueError("Các đường dẫn visualize_features không được trống.")

    image_size = int(section.get("image_size", 640))
    layer_indices = tuple(
        int(value) for value in section.get("layers", DEFAULT_LAYERS)
    )
    channels_per_layer = int(section.get("channels_per_layer", 16))
    colormap = str(section.get("colormap", "magma"))
    if image_size <= 0 or image_size % 32 != 0:
        raise ValueError("image_size phải lớn hơn 0 và chia hết cho 32.")
    if not layer_indices:
        raise ValueError("visualize_features.layers không được để trống.")
    if channels_per_layer <= 0:
        raise ValueError("channels_per_layer phải lớn hơn 0.")
    if colormap not in matplotlib.colormaps:
        raise ValueError(f"Matplotlib colormap không hợp lệ: {colormap}")

    device = choose_device(str(section.get("device", "auto")))
    print(f"Device: {device}")
    visualize_features(
        checkpoint_path=checkpoint_path,
        image_path=image_path,
        output_dir=output_dir,
        image_size=image_size,
        layer_indices=layer_indices,
        channels_per_layer=channels_per_layer,
        colormap=colormap,
        device=device,
    )
    print(f"Đã lưu feature visualization tại: {output_dir}")


if __name__ == "__main__":
    main()
