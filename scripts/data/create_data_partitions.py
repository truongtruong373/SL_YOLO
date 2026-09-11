from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from collections import Counter, defaultdict
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
PROJECT_ROOT = Path(__file__).resolve().parents[2]
VISDRONE_CLASS_COUNT = 10
DEFAULT_TRAIN_DIR = (
    PROJECT_ROOT / "data" / "VisDrone2019-DET-train"
)


@dataclass(frozen=True)
class Sample:
    source_id: str
    image_path: Path
    annotation_path: Path
    label_path: Path
    object_count: int
    class_counts: tuple[int, ...]


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


def read_class_counts(annotation_path: Path) -> tuple[int, ...]:
    """Đếm category hợp lệ và ánh xạ category 1..10 về class 0..9."""
    counts = [0] * VISDRONE_CLASS_COUNT
    with annotation_path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            fields = [field.strip() for field in line.split(",")]
            while fields and not fields[-1]:
                fields.pop()
            if not fields:
                continue
            if len(fields) < 6:
                raise ValueError(
                    f"{annotation_path}:{line_number}: annotation VisDrone "
                    "phải có ít nhất 6 giá trị."
                )
            try:
                raw_category = float(fields[5])
            except ValueError as error:
                raise ValueError(
                    f"{annotation_path}:{line_number}: category_id không hợp lệ."
                ) from error
            category_id = int(raw_category)
            if raw_category != category_id:
                raise ValueError(
                    f"{annotation_path}:{line_number}: category_id phải là "
                    "số nguyên."
                )
            if category_id in (0, 11):
                continue
            if not 1 <= category_id <= VISDRONE_CLASS_COUNT:
                raise ValueError(
                    f"{annotation_path}:{line_number}: category_id={category_id} "
                    "không thuộc VisDrone DET."
                )
            counts[category_id - 1] += 1
    return tuple(counts)


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
        object_count = read_object_count(label_path)
        class_counts = read_class_counts(annotation_path)
        if sum(class_counts) != object_count:
            raise ValueError(
                f"Label và annotation không khớp tại {relative_path}: "
                f"label={object_count}, annotation={sum(class_counts)}. "
                "Hãy chạy lại generate_non_iid_labels.py --overwrite."
            )
        sample = Sample(
            source_id=source_id,
            image_path=image_path,
            annotation_path=annotation_path,
            label_path=label_path,
            object_count=object_count,
            class_counts=class_counts,
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


def partition_iid_samples(
    samples: list[Sample],
    device_count: int,
    seed: int,
) -> list[list[Sample]]:
    """Chia IID, cân bằng kích thước, object-count và phân phối class.

    Ảnh được phân tầng theo đúng tổng số object. Mỗi device nhận quota bằng
    nhau trong từng tầng; phần dư được đưa tới các device đang có ít ảnh
    nhất. Trong quota đó, ảnh được đưa tới device đang thiếu nhất các class
    có trong ảnh.
    """
    if device_count <= 0:
        raise ValueError("device_count phải lớn hơn 0.")
    if len(samples) < device_count:
        raise ValueError(
            f"Chỉ có {len(samples)} ảnh, không đủ cho {device_count} thiết bị."
        )

    rng = random.Random(seed)
    total_class_counts = [
        sum(sample.class_counts[class_id] for sample in samples)
        for class_id in range(VISDRONE_CLASS_COUNT)
    ]
    target_class_counts = [
        count / device_count for count in total_class_counts
    ]
    rarity_weights = [
        1.0 / count if count else 0.0 for count in total_class_counts
    ]

    strata: dict[int, list[Sample]] = defaultdict(list)
    for sample in samples:
        strata[sample.object_count].append(sample)

    partitions: list[list[Sample]] = [[] for _ in range(device_count)]
    device_class_counts = [
        [0] * VISDRONE_CLASS_COUNT for _ in range(device_count)
    ]

    # Tầng đông object được xử lý trước vì có nhiều tín hiệu class hơn.
    for object_count in sorted(strata, reverse=True):
        stratum = strata[object_count]
        rng.shuffle(stratum)
        stratum.sort(
            key=lambda sample: sum(
                count * rarity_weights[class_id]
                for class_id, count in enumerate(sample.class_counts)
            ),
            reverse=True,
        )

        base_quota, remainder = divmod(len(stratum), device_count)
        device_tiebreakers = list(range(device_count))
        rng.shuffle(device_tiebreakers)
        tie_rank = {
            device_id: rank
            for rank, device_id in enumerate(device_tiebreakers)
        }
        remainder_devices = set(
            sorted(
                range(device_count),
                key=lambda device_id: (
                    len(partitions[device_id]), tie_rank[device_id]
                ),
            )[:remainder]
        )
        quotas = [
            base_quota + int(device_id in remainder_devices)
            for device_id in range(device_count)
        ]
        assigned_in_stratum = [0] * device_count

        for sample in stratum:
            available_devices = [
                device_id
                for device_id in range(device_count)
                if assigned_in_stratum[device_id] < quotas[device_id]
            ]

            def class_deficit_score(device_id: int) -> float:
                return sum(
                    sample.class_counts[class_id]
                    * (
                        target_class_counts[class_id]
                        - device_class_counts[device_id][class_id]
                    )
                    / max(target_class_counts[class_id], 1.0)
                    for class_id in range(VISDRONE_CLASS_COUNT)
                )

            selected_device = max(
                available_devices,
                key=lambda device_id: (
                    class_deficit_score(device_id),
                    -tie_rank[device_id],
                ),
            )
            partitions[selected_device].append(sample)
            assigned_in_stratum[selected_device] += 1
            for class_id, count in enumerate(sample.class_counts):
                device_class_counts[selected_device][class_id] += count

    return partitions


def create_relative_symlink(target: Path, link_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(target, start=link_path.parent)
    link_path.symlink_to(relative_target)


def write_partitions(
    train_dir: Path,
    output_dir: Path,
    partitions: list[list[Sample]],
    strategy: str,
    seed: int,
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
                    *(
                        f"class_{class_id}_count"
                        for class_id in range(VISDRONE_CLASS_COUNT)
                    ),
                ]
            )

            for device_index, samples in enumerate(partitions, start=1):
                device_id = f"device_{device_index:02d}"
                device_dir = temporary_dir / device_id
                device_dir.mkdir(parents=True)
                object_counts = [sample.object_count for sample in samples]
                source_ids = sorted({sample.source_id for sample in samples})
                source_means = [
                    mean(
                        sample.object_count
                        for sample in samples
                        if sample.source_id == source_id
                    )
                    for source_id in source_ids
                ]
                class_counts = [
                    sum(sample.class_counts[class_id] for sample in samples)
                    for class_id in range(VISDRONE_CLASS_COUNT)
                ]

                (device_dir / "sources.txt").write_text(
                    "".join(f"{source_id}\n" for source_id in source_ids),
                    encoding="utf-8",
                )
                for sample in samples:
                    image_relative_path = sample.image_path.relative_to(
                        train_dir / "images"
                    )
                    image_link = device_dir / "images" / image_relative_path
                    annotation_link = (
                        device_dir / "annotations" / image_relative_path
                    ).with_suffix(".txt")
                    label_link = (
                        device_dir / "labels" / image_relative_path
                    ).with_suffix(".txt")
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
                            *sample.class_counts,
                        ]
                    )

                summary.append(
                    {
                        "device_id": device_id,
                        "source_count": len(source_ids),
                        "sample_count": len(samples),
                        "min_object_count": min(object_counts),
                        "max_object_count": max(object_counts),
                        "mean_object_count": round(mean(object_counts), 4),
                        "median_object_count": float(median(object_counts)),
                        "min_source_mean": round(min(source_means), 4),
                        "max_source_mean": round(max(source_means), 4),
                        "object_count_histogram": dict(
                            sorted(Counter(object_counts).items())
                        ),
                        "class_object_counts": class_counts,
                        "class_object_proportions": [
                            round(count / sum(class_counts), 6)
                            if sum(class_counts) else 0.0
                            for count in class_counts
                        ],
                    }
                )

        all_samples = [sample for partition in partitions for sample in partition]
        metadata = {
            "source_definition": "filename prefix before the first underscore",
            "strategy": strategy,
            "seed": seed if strategy == "iid" else None,
            "partition_method": (
                "exact object-count stratification with class-deficit balancing"
                if strategy == "iid"
                else "contiguous source groups ordered by mean object count; "
                "dynamic-programming cuts minimize sample-count deviation"
            ),
            "device_count": len(partitions),
            "total_sources": len({sample.source_id for sample in all_samples}),
            "total_samples": sum(
                int(device["sample_count"]) for device in summary
            ),
            "total_class_object_counts": [
                sum(sample.class_counts[class_id] for sample in all_samples)
                for class_id in range(VISDRONE_CLASS_COUNT)
            ],
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
            "Chia tập train thành các thiết bị IID hoặc non-IID theo số "
            "đối tượng và phân phối class."
        )
    )
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument(
        "--strategy", choices=("iid", "non-iid"), default="non-iid"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Mặc định: data/VisDrone2019-DET-train-<strategy>-<devices>.",
    )
    parser.add_argument("--devices", type=int, default=8)
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed dùng khi chia IID."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_dir = args.train_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else PROJECT_ROOT
        / "data"
        / f"VisDrone2019-DET-train-{args.strategy}-{args.devices}"
    )
    source_groups = load_source_groups(train_dir)
    if args.strategy == "iid":
        samples = [
            sample for group in source_groups for sample in group.samples
        ]
        partitions = partition_iid_samples(samples, args.devices, args.seed)
    else:
        grouped_partitions = partition_contiguous_sources(
            source_groups, args.devices
        )
        partitions = [
            [sample for group in groups for sample in group.samples]
            for groups in grouped_partitions
        ]
    summary = write_partitions(
        train_dir, output_dir, partitions, args.strategy, args.seed
    )

    print(f"Đã tạo dữ liệu tại: {output_dir}")
    print(f"Strategy: {args.strategy}")
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
