from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "sl_yolo_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

plt.rcParams.update(
    {
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
    }
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import Attention, MyYOLODetectionModel  # noqa: E402


CUT_LAYERS = {
    "Cut A": 3,
    "Cut B": 10,
    "Cut C": 16,
    "Cut D": 22,
}
FLOPS_COLOR = "#2563eb"
COMMUNICATION_COLOR = "#f59e0b"


def cut_elements_per_image(image_size: int) -> dict[str, int]:
    spatial_8 = image_size // 8
    spatial_16 = image_size // 16
    spatial_32 = image_size // 32
    return {
        "Cut A": 64 * spatial_8 * spatial_8,
        # Skip connections khiến Cut B phải truyền x4, x6 và x10.
        "Cut B": (
            128 * spatial_8 * spatial_8
            + 128 * spatial_16 * spatial_16
            + 256 * spatial_32 * spatial_32
        ),
        # Cut C truyền x16, x13, x10; Cut D truyền x16, x19, x22.
        "Cut C": (
            64 * spatial_8 * spatial_8
            + 128 * spatial_16 * spatial_16
            + 256 * spatial_32 * spatial_32
        ),
        "Cut D": (
            64 * spatial_8 * spatial_8
            + 128 * spatial_16 * spatial_16
            + 256 * spatial_32 * spatial_32
        ),
    }


def profile_client_training_gflops(image_size: int) -> dict[str, float]:
    """Ước lượng forward + backward FLOPs cho một ảnh trước mỗi Cut."""
    model = MyYOLODetectionModel(nc=10).eval()
    layer_macs = {index: 0 for index in range(len(model.model))}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_conv_hook(layer_index: int):
        def hook(
            module: torch.nn.Conv2d,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            batch, output_channels, height, width = output.shape
            kernel_height, kernel_width = module.kernel_size
            macs = (
                batch
                * output_channels
                * height
                * width
                * (module.in_channels // module.groups)
                * kernel_height
                * kernel_width
            )
            layer_macs[layer_index] += macs

        return hook

    def make_attention_hook(layer_index: int):
        def hook(
            module: Attention,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            batch, _, height, width = output.shape
            positions = height * width
            # QK^T và V·attention, chưa tính các Conv vì đã có hook riêng.
            macs = (
                batch
                * module.num_heads
                * positions
                * positions
                * (module.key_dim + module.head_dim)
            )
            layer_macs[layer_index] += macs

        return hook

    for layer_index, layer in enumerate(model.model):
        for module in layer.modules():
            if isinstance(module, torch.nn.Conv2d):
                handles.append(
                    module.register_forward_hook(make_conv_hook(layer_index))
                )
            elif isinstance(module, Attention):
                handles.append(
                    module.register_forward_hook(
                        make_attention_hook(layer_index)
                    )
                )

    try:
        with torch.inference_mode():
            model(torch.zeros(1, 3, image_size, image_size))
    finally:
        for handle in handles:
            handle.remove()

    return {
        cut_name: (
            # 1 MAC = 2 FLOPs; backward xấp xỉ 2 lần forward.
            3
            * 2
            * sum(layer_macs[index] for index in range(cut_layer + 1))
            / 1e9
        )
        for cut_name, cut_layer in CUT_LAYERS.items()
    }


def add_value_labels(
    axis: plt.Axes,
    bars: plt.Container,
    values: np.ndarray,
    unit: str,
) -> None:
    axis.bar_label(
        bars,
        labels=[f"{value:.1f}\n{unit}" for value in values],
        padding=7,
        fontsize=14,
        fontweight="bold",
    )


def plot_costs(
    batch_size: int,
    image_size: int,
    output_path: Path,
) -> None:
    client_gflops = profile_client_training_gflops(image_size)
    cut_elements = cut_elements_per_image(image_size)
    cuts = tuple(CUT_LAYERS)

    flops_values = np.asarray(
        [client_gflops[cut] for cut in cuts],
        dtype=np.float64,
    )
    # FP32: activation forward (4 byte) + gradient backward (4 byte).
    communication_values = np.asarray(
        [cut_elements[cut] * batch_size * 8 / 1e6 for cut in cuts],
        dtype=np.float64,
    )
    x_positions = np.arange(len(cuts))

    figure, flops_axis = plt.subplots(figsize=(14, 8))
    communication_axis = flops_axis.twinx()
    bar_width = 0.36
    flops_bars = flops_axis.bar(
        x_positions - bar_width / 2,
        flops_values,
        width=bar_width,
        color=FLOPS_COLOR,
        label="Chi phí tính toán",
    )
    communication_bars = communication_axis.bar(
        x_positions + bar_width / 2,
        communication_values,
        width=bar_width,
        color=COMMUNICATION_COLOR,
        label=f"Chi phí truyền thông",
    )

    add_value_labels(flops_axis, flops_bars, flops_values, "GFLOPs")
    add_value_labels(
        communication_axis,
        communication_bars,
        communication_values,
        "MB",
    )

    flops_axis.set_ylabel("Chi phí tính toán tại máy khách (GFLOPs)")
    communication_axis.set_ylabel("Chi phí truyền thông (MB)")
    flops_axis.set_xlabel("Điểm Cut")
    flops_axis.set_xticks(x_positions, cuts)
    flops_axis.set_ylim(0, flops_values.max() * 1.2)
    communication_axis.set_ylim(0, communication_values.max() * 1.2)
    flops_axis.grid(axis="y", linestyle="--", alpha=0.35)

    handles = [flops_bars, communication_bars]
    labels = [handle.get_label() for handle in handles]
    flops_axis.legend(
        handles,
        labels,
        loc="upper center",
        ncols=2,
        frameon=True,
    )
    # flops_axis.set_title(
    #     f"Chi phí tính toán và truyền dữ liệu tại các điểm Cut — "
    #     f"input {image_size}×{image_size}\n"
    #     "Training FLOPs ≈ 3× forward FLOPs (forward + backward)"
    # )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vẽ FLOPs phía client và chi phí truyền tại các Cut."
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/cut_compute_and_communication_bs8_960.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size phải lớn hơn 0.")
    if args.image_size <= 0 or args.image_size % 32 != 0:
        raise ValueError("image-size phải lớn hơn 0 và chia hết cho 32.")

    output_path = args.output.expanduser().resolve()
    plot_costs(
        batch_size=args.batch_size,
        image_size=args.image_size,
        output_path=output_path,
    )
    print(f"Đã lưu biểu đồ: {output_path}")


if __name__ == "__main__":
    main()
