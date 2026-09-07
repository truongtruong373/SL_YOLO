from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def _normalize_image_size(
    image_size: int | Sequence[int],
) -> tuple[int, int]:
    """Trả về kích thước ảnh theo thứ tự (height, width)."""
    if isinstance(image_size, int):
        height = width = image_size
    else:
        if len(image_size) != 2:
            raise ValueError(
                "image_size phải là int hoặc một cặp (height, width)."
            )
        height, width = int(image_size[0]), int(image_size[1])

    if height <= 0 or width <= 0:
        raise ValueError("Kích thước ảnh phải lớn hơn 0.")

    if height % 32 != 0 or width % 32 != 0:
        raise ValueError(
            "Chiều cao và chiều rộng phải chia hết cho stride lớn nhất (32)."
        )

    return height, width


def letterbox(
    image: np.ndarray,
    boxes: np.ndarray,
    new_shape: tuple[int, int],
    color: tuple[int, int, int] = (114, 114, 114),
    scale_up: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resize và padding ảnh mà không làm thay đổi tỉ lệ.

    Args:
        image:
            Ảnh BGR có shape [H, W, 3].
        boxes:
            Bounding box normalized xywh, shape [N, 4].
        new_shape:
            Kích thước đầu ra (height, width).

    Returns:
        image:
            Ảnh BGR sau letterbox.
        boxes:
            Bounding box normalized xywh theo kích thước ảnh mới.
    """
    original_height, original_width = image.shape[:2]
    output_height, output_width = new_shape

    ratio = min(
        output_height / original_height,
        output_width / original_width,
    )
    if not scale_up:
        ratio = min(ratio, 1.0)

    resized_width = int(round(original_width * ratio))
    resized_height = int(round(original_height * ratio))

    padding_width = output_width - resized_width
    padding_height = output_height - resized_height

    left = int(round(padding_width / 2 - 0.1))
    right = int(round(padding_width / 2 + 0.1))
    top = int(round(padding_height / 2 - 0.1))
    bottom = int(round(padding_height / 2 + 0.1))

    if (
        resized_width != original_width
        or resized_height != original_height
    ):
        image = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )

    image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color,
    )

    if boxes.shape[0] == 0:
        return image, boxes

    # normalized xywh (ảnh gốc) -> pixel xyxy
    center_x = boxes[:, 0] * original_width
    center_y = boxes[:, 1] * original_height
    box_width = boxes[:, 2] * original_width
    box_height = boxes[:, 3] * original_height

    x1 = (center_x - box_width / 2) * ratio + left
    y1 = (center_y - box_height / 2) * ratio + top
    x2 = (center_x + box_width / 2) * ratio + left
    y2 = (center_y + box_height / 2) * ratio + top

    x1 = np.clip(x1, 0, output_width)
    y1 = np.clip(y1, 0, output_height)
    x2 = np.clip(x2, 0, output_width)
    y2 = np.clip(y2, 0, output_height)

    transformed_boxes = np.stack(
        (
            (x1 + x2) / (2 * output_width),
            (y1 + y2) / (2 * output_height),
            (x2 - x1) / output_width,
            (y2 - y1) / output_height,
        ),
        axis=1,
    ).astype(np.float32, copy=False)

    return image, transformed_boxes


class YOLODetectionDataset(Dataset):
    """
    Dataset cho annotation YOLO:

        <class_id> <center_x> <center_y> <width> <height>

    Bốn giá trị bounding box phải được chuẩn hóa trong khoảng [0, 1].
    """

    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path,
        image_size: int | Sequence[int] = 640,
        num_classes: int | None = None,
        augment: bool = False,
        horizontal_flip_probability: float = 0.5,
        scale_up: bool = True,
        strict_labels: bool = False,
    ) -> None:
        super().__init__()

        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.image_size = _normalize_image_size(image_size)
        self.num_classes = num_classes
        self.augment = augment
        self.horizontal_flip_probability = horizontal_flip_probability
        self.scale_up = scale_up
        self.strict_labels = strict_labels

        if not self.images_dir.is_dir():
            raise FileNotFoundError(
                f"Không tìm thấy thư mục ảnh: {self.images_dir}"
            )

        if not self.labels_dir.is_dir():
            raise FileNotFoundError(
                f"Không tìm thấy thư mục label: {self.labels_dir}"
            )

        if num_classes is not None and num_classes <= 0:
            raise ValueError("num_classes phải lớn hơn 0.")

        if not 0.0 <= horizontal_flip_probability <= 1.0:
            raise ValueError(
                "horizontal_flip_probability phải nằm trong [0, 1]."
            )

        self.image_paths = sorted(
            path
            for path in self.images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not self.image_paths:
            raise RuntimeError(
                f"Không tìm thấy ảnh hợp lệ trong: {self.images_dir}"
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def _label_path(self, image_path: Path) -> Path:
        relative_path = image_path.relative_to(self.images_dir)
        return (self.labels_dir / relative_path).with_suffix(".txt")

    def _load_labels(
        self,
        label_path: Path,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not label_path.exists():
            if self.strict_labels:
                raise FileNotFoundError(
                    f"Không tìm thấy label cho ảnh: {label_path}"
                )
            return (
                np.empty((0,), dtype=np.int64),
                np.empty((0, 4), dtype=np.float32),
            )

        classes: list[int] = []
        boxes: list[list[float]] = []

        with label_path.open("r", encoding="utf-8") as label_file:
            for line_number, line in enumerate(label_file, start=1):
                fields = line.strip().split()
                if not fields:
                    continue

                if len(fields) != 5:
                    raise ValueError(
                        f"{label_path}:{line_number}: "
                        "mỗi dòng phải có đúng 5 giá trị."
                    )

                try:
                    raw_class = float(fields[0])
                    box = [float(value) for value in fields[1:]]
                except ValueError as error:
                    raise ValueError(
                        f"{label_path}:{line_number}: label không hợp lệ."
                    ) from error

                class_id = int(raw_class)
                if raw_class != class_id or class_id < 0:
                    raise ValueError(
                        f"{label_path}:{line_number}: "
                        "class_id phải là số nguyên không âm."
                    )

                if (
                    self.num_classes is not None
                    and class_id >= self.num_classes
                ):
                    raise ValueError(
                        f"{label_path}:{line_number}: class_id={class_id} "
                        f"vượt quá num_classes={self.num_classes}."
                    )

                if not np.isfinite(box).all():
                    raise ValueError(
                        f"{label_path}:{line_number}: bbox chứa NaN hoặc Inf."
                    )

                center_x, center_y, width, height = box
                if (
                    not 0.0 <= center_x <= 1.0
                    or not 0.0 <= center_y <= 1.0
                    or width <= 0.0
                    or height <= 0.0
                    or width > 1.0
                    or height > 1.0
                ):
                    raise ValueError(
                        f"{label_path}:{line_number}: "
                        "bbox normalized xywh không hợp lệ."
                    )

                classes.append(class_id)
                boxes.append(box)

        return (
            np.asarray(classes, dtype=np.int64),
            np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        )

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[index]
        label_path = self._label_path(image_path)

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Không thể đọc ảnh: {image_path}")

        classes, boxes = self._load_labels(label_path)
        image, boxes = letterbox(
            image=image,
            boxes=boxes,
            new_shape=self.image_size,
            scale_up=self.scale_up,
        )

        if (
            self.augment
            and random.random() < self.horizontal_flip_probability
        ):
            image = np.ascontiguousarray(image[:, ::-1])
            if boxes.shape[0] > 0:
                boxes[:, 0] = 1.0 - boxes[:, 0]

        # BGR HWC uint8 -> RGB CHW float32 trong khoảng [0, 1].
        image = np.ascontiguousarray(
            image[:, :, ::-1].transpose(2, 0, 1)
        )
        image_tensor = torch.from_numpy(image).float().div_(255.0)

        return (
            image_tensor,
            torch.from_numpy(classes),
            torch.from_numpy(boxes),
        )

    @staticmethod
    def collate_fn(
        batch: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Gộp sample thành đúng format mà YOLODetectionLoss nhận.

        Returns:
            images:
                FloatTensor [B, 3, H, W].
            targets:
                {
                    "batch_idx": LongTensor [N],
                    "cls": LongTensor [N],
                    "bboxes": FloatTensor [N, 4],
                }
        """
        images, classes_per_image, boxes_per_image = zip(*batch)
        images_tensor = torch.stack(images, dim=0)

        batch_indices = [
            torch.full(
                (classes.shape[0],),
                image_index,
                dtype=torch.long,
            )
            for image_index, classes in enumerate(classes_per_image)
        ]

        targets = {
            "batch_idx": torch.cat(batch_indices, dim=0),
            "cls": torch.cat(classes_per_image, dim=0),
            "bboxes": torch.cat(boxes_per_image, dim=0),
        }

        return images_tensor, targets


def create_dataloader(
    images_dir: str | Path,
    labels_dir: str | Path,
    image_size: int | Sequence[int] = 640,
    batch_size: int = 16,
    num_classes: int | None = None,
    augment: bool = False,
    shuffle: bool | None = None,
    num_workers: int = 4,
    pin_memory: bool | None = None,
    drop_last: bool = False,
) -> DataLoader:
    """Tạo DataLoader sẵn sàng dùng trong vòng lặp huấn luyện."""
    if batch_size <= 0:
        raise ValueError("batch_size phải lớn hơn 0.")

    if num_workers < 0:
        raise ValueError("num_workers không được âm.")

    dataset = YOLODetectionDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        image_size=image_size,
        num_classes=num_classes,
        augment=augment,
    )

    if shuffle is None:
        shuffle = augment

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        collate_fn=dataset.collate_fn,
    )


if __name__ == "__main__":
    data_root = Path(__file__).resolve().parent / "data"
    train_loader = create_dataloader(
        images_dir=data_root / "images" / "train",
        labels_dir=data_root / "labels" / "train",
        image_size=640,
        batch_size=4,
        num_classes=11,
        augment=True,
        num_workers=0,
    )

    images, targets = next(iter(train_loader))
    print("images:", images.shape, images.dtype)
    print("batch_idx:", targets["batch_idx"].shape)
    print("cls:", targets["cls"].shape)
    print("bboxes:", targets["bboxes"].shape)
