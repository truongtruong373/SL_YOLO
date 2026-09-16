from __future__ import annotations

import argparse
import json
import os
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


plt.rcParams.update(
    {
        "font.size": 19,
        "axes.titlesize": 24,
        "axes.labelsize": 20,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 18,
    }
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = (
    PROJECT_ROOT / "data/VisDrone2019-DET-train-iid-8/summary.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "runs/iid_device_statistics.png"
BAR_COLOR = "#4C78A8"
LINE_COLOR = "#E45756"


def load_device_statistics(
    summary_path: Path,
) -> tuple[str, list[str], np.ndarray, np.ndarray]:
    with summary_path.open("r", encoding="utf-8") as summary_file:
        summary = json.load(summary_file)

    devices = summary.get("devices")
    if not isinstance(devices, list) or not devices:
        raise ValueError(f"Không có thống kê device trong {summary_path}")

    names = [f"Máy\nkhách {index}" for index in range(1, len(devices) + 1)]
    mean_object_counts = np.asarray(
        [float(device["mean_object_count"]) for device in devices],
        dtype=np.float64,
    )
    sample_counts = np.asarray(
        [int(device["sample_count"]) for device in devices],
        dtype=np.int64,
    )
    strategy = summary.get("strategy")
    if not strategy:
        directory_name = summary_path.parent.name.lower()
        strategy = "non-IID" if "non-iid" in directory_name else "IID"
    return str(strategy), names, mean_object_counts, sample_counts


def plot_device_statistics(summary_path: Path, output_path: Path) -> None:
    strategy, device_names, mean_object_counts, sample_counts = (
        load_device_statistics(summary_path)
    )
    positions = np.arange(len(device_names))

    figure, object_axis = plt.subplots(figsize=(12, 8))
    sample_axis = object_axis.twinx()

    bars = object_axis.bar(
        positions,
        mean_object_counts,
        width=0.48,
        color=BAR_COLOR,
        alpha=0.88,
        label="Số đối tượng trung bình/ảnh",
        zorder=2,
    )
    (sample_line,) = sample_axis.plot(
        positions,
        sample_counts,
        color=LINE_COLOR,
        linewidth=2.5,
        marker="o",
        markersize=7,
        label="Số lượng mẫu",
        zorder=3,
    )

    object_axis.bar_label(
        bars,
        labels=[f"{value:.1f}" for value in mean_object_counts],
        padding=4,
        fontsize=16,
        color=BAR_COLOR,
    )
    for x_position, sample_count in zip(positions, sample_counts, strict=True):
        sample_axis.annotate(
            f"{sample_count:,}",
            (x_position, sample_count),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=16,
            color=LINE_COLOR,
            fontweight="bold",
        )

    object_axis.tick_params(axis="y", labelcolor=BAR_COLOR)
    sample_axis.tick_params(axis="y", labelcolor=LINE_COLOR)
    object_axis.set_xticks(positions, device_names)
    object_axis.set_ylim(0, max(mean_object_counts) * 1.22)

    sample_min = int(sample_counts.min())
    sample_max = int(sample_counts.max())
    sample_padding = max(1, int((sample_max - sample_min) * 0.35))
    sample_axis.set_ylim(sample_min - sample_padding, sample_max + 3 * sample_padding)

    object_axis.grid(axis="y", linestyle="--", alpha=0.3, zorder=1)
    # object_axis.set_title(f"Phân bố dữ liệu {strategy.upper()} trên các thiết bị")
    object_axis.legend(
        [bars, sample_line],
        [bars.get_label(), sample_line.get_label()],
        loc="upper center",
        ncols=2,
        frameon=True,
    )

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Vẽ số đối tượng trung bình và số lượng mẫu của từng device."
        )
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_device_statistics(
        args.summary.expanduser().resolve(),
        args.output.expanduser().resolve(),
    )
    print(f"Đã lưu biểu đồ tại: {args.output}")


if __name__ == "__main__":
    main()
