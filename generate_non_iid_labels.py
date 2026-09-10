from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
VISDRONE_CLASS_COUNT = 10
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_SPLITS = (
    "VisDrone2019-DET-train",
    "VisDrone2019-DET-val",
)


@dataclass(frozen=True)
class SplitReport:
    split: str
    sample_count: int
    object_count_distribution: tuple[tuple[int, int], ...]


def parse_visdrone_categories(annotation_path: Path) -> Counter[int]:
    """Count valid VisDrone categories and map category 1..10 to class 0..9."""
    category_counts: Counter[int] = Counter()

    with annotation_path.open("r", encoding="utf-8-sig") as annotation_file:
        for line_number, line in enumerate(annotation_file, start=1):
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
                    f"{annotation_path}:{line_number}: category_id "
                    "không hợp lệ."
                ) from error

            category_id = int(raw_category)
            if raw_category != category_id:
                raise ValueError(
                    f"{annotation_path}:{line_number}: category_id "
                    "phải là số nguyên."
                )
            if category_id in (0, 11):
                continue
            if not 1 <= category_id <= VISDRONE_CLASS_COUNT:
                raise ValueError(
                    f"{annotation_path}:{line_number}: category_id="
                    f"{category_id} không thuộc VisDrone DET."
                )

            category_counts[category_id - 1] += 1

    return category_counts


def object_count(annotation_path: Path) -> int:
    """Return the total number of valid VisDrone objects in an image."""
    return sum(parse_visdrone_categories(annotation_path).values())


def write_label(label_path: Path, count: int, overwrite: bool) -> None:
    content = f"{count}\n"
    if label_path.exists():
        current_content = label_path.read_text(encoding="utf-8")
        if current_content == content:
            return
        if not overwrite:
            raise FileExistsError(
                f"Label đã tồn tại và có nội dung khác: {label_path}. "
                "Dùng --overwrite để ghi đè."
            )

    label_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = label_path.with_suffix(label_path.suffix + ".tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(label_path)


def generate_split_labels(
    data_dir: Path,
    split: str,
    overwrite: bool = False,
) -> SplitReport:
    split_dir = data_dir / split
    images_dir = split_dir / "images"
    annotations_dir = split_dir / "annotations"
    labels_dir = split_dir / "labels"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {images_dir}")
    if not annotations_dir.is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục annotation: {annotations_dir}"
        )

    image_paths = sorted(
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise RuntimeError(f"Không tìm thấy ảnh hợp lệ trong: {images_dir}")

    object_count_distribution: Counter[int] = Counter()

    for image_path in image_paths:
        relative_path = image_path.relative_to(images_dir)
        annotation_path = (annotations_dir / relative_path).with_suffix(".txt")
        label_path = (labels_dir / relative_path).with_suffix(".txt")
        if not annotation_path.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy annotation cho ảnh {image_path}: "
                f"{annotation_path}"
            )

        count = object_count(annotation_path)
        write_label(label_path, count, overwrite=overwrite)
        object_count_distribution[count] += 1

    return SplitReport(
        split=split,
        sample_count=len(image_paths),
        object_count_distribution=tuple(
            sorted(object_count_distribution.items())
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tạo label phân chia non-IID từ tổng số đối tượng hợp lệ "
            "trong annotation VisDrone."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Thư mục data chứa các split VisDrone.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=DEFAULT_SPLITS,
        help="Tên các split cần xử lý.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ghi đè label đã tồn tại nếu nội dung thay đổi.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()

    for split in args.splits:
        report = generate_split_labels(
            data_dir=data_dir,
            split=split,
            overwrite=args.overwrite,
        )
        distribution = ", ".join(
            f"{object_count}:{image_count}"
            for object_count, image_count in report.object_count_distribution
        )
        print(
            f"{report.split}: {report.sample_count} labels, "
            f"object-count distribution=[{distribution}]"
        )


if __name__ == "__main__":
    main()
