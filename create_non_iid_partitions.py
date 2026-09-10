from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median


IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
DEFAULT_TRAIN_DIR = (
    Path(__file__).resolve().parent / "data" / "VisDrone2019-DET-train"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "data"
    / "VisDrone2019-DET-train-non-iid-8"
)


@dataclass(frozen=True)
class Sample:
    source_id: str
    image_path: Path
    annotation_path: Path
    label_path: Path
    object_count: int


@dataclass(frozen=True)
class SourceGroup:
    source_id: str
    samples: tuple[Sample, ...]
    mean_object_count: float


def read_object_count(label_path: Path) -> int:
    content = label_path.read_text(encoding="utf-8").strip()
    try:
        object_count = int(content)
    except ValueError as error:
        raise ValueError(
            f"{label_path}: label phải chứa đúng một số nguyên."
        ) from error
    if object_count < 0:
        raise ValueError(f"{label_path}: label không được âm.")
    return object_count


def source_id_from_path(image_path: Path) -> str:
    source_id, separator, _ = image_path.stem.partition("_")
    if not separator or not source_id:
        raise ValueError(
            f"{image_path}: không thể lấy source ID từ tên ảnh."
        )
    return source_id


def load_source_groups(train_dir: Path) -> list[SourceGroup]:
    images_dir = train_dir / "images"
    annotations_dir = train_dir / "annotations"
    labels_dir = train_dir / "labels"
    for directory in (images_dir, annotations_dir, labels_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Không tìm thấy thư mục: {directory}")

    grouped_samples: dict[str, list[Sample]] = {}
    image_paths = sorted(
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise RuntimeError(f"Không tìm thấy ảnh trong: {images_dir}")

    for image_path in image_paths:
        relative_path = image_path.relative_to(images_dir)
        annotation_path = (annotations_dir / relative_path).with_suffix(".txt")
        label_path = (labels_dir / relative_path).with_suffix(".txt")
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Thiếu annotation: {annotation_path}")
        if not label_path.is_file():
            raise FileNotFoundError(f"Thiếu label: {label_path}")

        source_id = source_id_from_path(image_path)
        sample = Sample(
            source_id=source_id,
            image_path=image_path,
            annotation_path=annotation_path,
            label_path=label_path,
            object_count=read_object_count(label_path),
        )
        grouped_samples.setdefault(source_id, []).append(sample)

    source_groups = [
        SourceGroup(
            source_id=source_id,
            samples=tuple(samples),
            mean_object_count=mean(
                sample.object_count for sample in samples
            ),
        )
        for source_id, samples in grouped_samples.items()
    ]
    return sorted(
        source_groups,
        key=lambda group: (group.mean_object_count, group.source_id),
    )


def partition_contiguous_sources(
    source_groups: list[SourceGroup],
    device_count: int,
) -> list[list[SourceGroup]]:
    """Minimize image-count imbalance while preserving source-score order."""
    source_count = len(source_groups)
    if device_count <= 0:
        raise ValueError("device_count phải lớn hơn 0.")
    if source_count < device_count:
        raise ValueError(
            f"Chỉ có {source_count} nguồn, không đủ cho {device_count} thiết bị."
        )

    prefix_image_counts = [0]
    for group in source_groups:
        prefix_image_counts.append(
            prefix_image_counts[-1] + len(group.samples)
        )

    target = prefix_image_counts[-1] / device_count
    infinity = float("inf")
    costs = [
        [infinity] * (source_count + 1) for _ in range(device_count + 1)
    ]
    previous_cut = [
        [-1] * (source_count + 1) for _ in range(device_count + 1)
    ]
    costs[0][0] = 0.0

    for partition_count in range(1, device_count + 1):
        for end in range(partition_count, source_count + 1):
            for start in range(partition_count - 1, end):
                image_count = (
                    prefix_image_counts[end] - prefix_image_counts[start]
                )
                candidate_cost = (
                    costs[partition_count - 1][start]
                    + (image_count - target) ** 2
                )
                if candidate_cost < costs[partition_count][end]:
                    costs[partition_count][end] = candidate_cost
                    previous_cut[partition_count][end] = start

    boundaries = [source_count]
    end = source_count
    for partition_count in range(device_count, 0, -1):
        end = previous_cut[partition_count][end]
        if end < 0:
            raise RuntimeError("Không thể tìm được cách chia dữ liệu.")
        boundaries.append(end)
    boundaries.reverse()

    return [
        source_groups[start:end]
        for start, end in zip(boundaries, boundaries[1:])
    ]


def create_relative_symlink(target: Path, link_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(target, start=link_path.parent)
    link_path.symlink_to(relative_target)


def write_partitions(
    train_dir: Path,
    output_dir: Path,
    partitions: list[list[SourceGroup]],
) -> list[dict[str, object]]:
    if output_dir.exists():
        raise FileExistsError(
            f"Thư mục đầu ra đã tồn tại: {output_dir}. Hãy xóa hoặc chọn "
            "--output-dir khác nếu muốn tạo lại."
        )

    temporary_dir = output_dir.with_name(output_dir.name + ".tmp")
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True)

    summary: list[dict[str, object]] = []
    manifest_path = temporary_dir / "manifest.csv"
    try:
        with manifest_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "device_id",
                    "source_id",
                    "image_path",
                    "annotation_path",
                    "label_path",
                    "object_count",
                ]
            )

            for device_index, source_groups in enumerate(partitions, start=1):
                device_id = f"device_{device_index:02d}"
                device_dir = temporary_dir / device_id
                device_dir.mkdir(parents=True)
                samples = [
                    sample
                    for group in source_groups
                    for sample in group.samples
                ]
                object_counts = [sample.object_count for sample in samples]

                (device_dir / "sources.txt").write_text(
                    "".join(f"{group.source_id}\n" for group in source_groups),
                    encoding="utf-8",
                )
                for sample in samples:
                    image_link = device_dir / "images" / sample.image_path.name
                    annotation_link = (
                        device_dir / "annotations" / sample.annotation_path.name
                    )
                    label_link = device_dir / "labels" / sample.label_path.name
                    create_relative_symlink(sample.image_path, image_link)
                    create_relative_symlink(sample.annotation_path, annotation_link)
                    create_relative_symlink(sample.label_path, label_link)
                    writer.writerow(
                        [
                            device_id,
                            sample.source_id,
                            sample.image_path.relative_to(train_dir),
                            sample.annotation_path.relative_to(train_dir),
                            sample.label_path.relative_to(train_dir),
                            sample.object_count,
                        ]
                    )

                summary.append(
                    {
                        "device_id": device_id,
                        "source_count": len(source_groups),
                        "sample_count": len(samples),
                        "min_object_count": min(object_counts),
                        "max_object_count": max(object_counts),
                        "mean_object_count": round(mean(object_counts), 4),
                        "median_object_count": float(median(object_counts)),
                        "min_source_mean": round(
                            source_groups[0].mean_object_count, 4
                        ),
                        "max_source_mean": round(
                            source_groups[-1].mean_object_count, 4
                        ),
                    }
                )

        metadata = {
            "source_definition": "filename prefix before the first underscore",
            "ordering_metric": "mean object count per source",
            "partition_method": (
                "contiguous dynamic-programming cuts minimizing squared "
                "sample-count deviation"
            ),
            "device_count": len(partitions),
            "total_sources": sum(len(partition) for partition in partitions),
            "total_samples": sum(
                int(device["sample_count"]) for device in summary
            ),
            "devices": summary,
        }
        (temporary_dir / "summary.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_dir.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Chia tập train thành các thiết bị non-IID theo số đối tượng, "
            "không tách ảnh cùng nguồn."
        )
    )
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--devices", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_dir = args.train_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    source_groups = load_source_groups(train_dir)
    partitions = partition_contiguous_sources(source_groups, args.devices)
    summary = write_partitions(train_dir, output_dir, partitions)

    print(f"Đã tạo dữ liệu tại: {output_dir}")
    print("device,sources,samples,mean,median,min,max")
    for device in summary:
        print(
            f"{device['device_id']},{device['source_count']},"
            f"{device['sample_count']},{device['mean_object_count']},"
            f"{device['median_object_count']},{device['min_object_count']},"
            f"{device['max_object_count']}"
        )


if __name__ == "__main__":
    main()
