from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parent / "runs" / "label_distribution.png"
)
SPLITS = (
    ("VisDrone2019-DET-train", "Train", "#2563EB"),
    ("VisDrone2019-DET-val", "Validation", "#F97316"),
)


def load_labels(labels_dir: Path) -> list[int]:
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục label: {labels_dir}")

    label_paths = sorted(labels_dir.rglob("*.txt"))
    if not label_paths:
        raise RuntimeError(f"Không tìm thấy file label trong: {labels_dir}")

    labels: list[int] = []
    for label_path in label_paths:
        content = label_path.read_text(encoding="utf-8").strip()
        try:
            label = int(content)
        except ValueError as error:
            raise ValueError(
                f"{label_path}: label phải là một số nguyên."
            ) from error
        if label < 0:
            raise ValueError(f"{label_path}: label không được âm.")
        labels.append(label)

    return labels


def draw_distribution(
    axis: plt.Axes,
    labels: list[int],
    split_name: str,
    color: str,
) -> None:
    distribution = Counter(labels)
    x_values = sorted(distribution)
    frequencies = [distribution[label] for label in x_values]

    axis.bar(
        x_values,
        frequencies,
        width=1.0,
        color=color,
        edgecolor=color,
        linewidth=0,
        alpha=0.9,
    )
    axis.set_title(split_name, loc="left", fontsize=14, fontweight="bold")
    axis.set_xlim(left=0, right=max(x_values) * 1.02)
    axis.set_ylim(bottom=0)
    axis.set_ylabel("Số lượng ảnh")
    axis.grid(axis="y", color="#CBD5E1", linewidth=0.8, alpha=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)

    summary = (
        f"Số ảnh: {len(labels):,}\n"
        f"Label duy nhất: {len(distribution):,}\n"
        f"Khoảng: {min(labels)}–{max(labels)}\n"
        f"Trung bình: {mean(labels):.1f}\n"
        f"Trung vị: {median(labels):.1f}"
    )
    axis.text(
        0.985,
        0.95,
        summary,
        transform=axis.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=9,
        linespacing=1.4,
        bbox={
            "boxstyle": "round,pad=0.6",
            "facecolor": "white",
            "edgecolor": "#CBD5E1",
            "alpha": 0.95,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vẽ phân phối train/validation theo label số đối tượng."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    figure, axes = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True)
    figure.suptitle(
        "Phân phối ảnh theo số lượng đối tượng",
        fontsize=18,
        fontweight="bold",
    )

    for axis, (split_dir, split_name, color) in zip(axes, SPLITS):
        labels = load_labels(data_dir / split_dir / "labels")
        draw_distribution(axis, labels, split_name, color)

    axes[-1].set_xlabel("Label (số đối tượng hợp lệ trong ảnh)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Đã lưu biểu đồ: {output_path}")


if __name__ == "__main__":
    main()
