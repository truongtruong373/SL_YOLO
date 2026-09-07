from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from dataset import VISDRONE_CLASS_NAMES
from predict_image import (
    DEFAULT_CHECKPOINT,
    DEFAULT_IMAGE,
    DEFAULT_OUTPUT,
    class_color,
    choose_device,
    predict_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test model VisDrone đã train trên một ảnh và hiển thị kết quả."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint best.pt hoặc last.pt.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help="Ảnh cần nhận diện.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Nơi lưu ảnh đã vẽ bounding box.",
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=None,
        help=(
            "Annotation VisDrone của ảnh. Mặc định tự tìm trong thư mục "
            "annotations nằm cạnh thư mục images."
        ),
    )
    parser.add_argument(
        "--ground-truth-output",
        type=Path,
        default=None,
        help="Nơi lưu ảnh ground truth (mặc định cạnh ảnh prediction).",
    )
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:0 hoặc mps.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.image_size <= 0 or args.image_size % 32 != 0:
        raise ValueError(
            "image-size phải lớn hơn 0 và chia hết cho 32."
        )
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("conf phải nằm trong [0, 1].")
    if not 0.0 <= args.iou <= 1.0:
        raise ValueError("iou phải nằm trong [0, 1].")
    if args.max_det <= 0:
        raise ValueError("max-det phải lớn hơn 0.")


def resolve_annotation_path(
    image_path: Path,
    requested_path: Path | None,
) -> Path:
    if requested_path is not None:
        return requested_path.expanduser().resolve()
    return image_path.parent.parent / "annotations" / f"{image_path.stem}.txt"


def draw_ground_truth(
    image_path: Path,
    annotation_path: Path,
    output_path: Path,
) -> int:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Không thể đọc ảnh: {image_path}")
    if not annotation_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy annotation ground truth: {annotation_path}"
        )

    image_height, image_width = image.shape[:2]
    object_count = 0
    with annotation_path.open("r", encoding="utf-8-sig") as label_file:
        for line_number, line in enumerate(label_file, start=1):
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
                left, top, width, height = map(float, fields[:4])
                category_id = int(float(fields[5]))
            except ValueError as error:
                raise ValueError(
                    f"{annotation_path}:{line_number}: annotation không hợp lệ."
                ) from error

            # VisDrone category 0=ignored regions, 11=others.
            if not 1 <= category_id <= len(VISDRONE_CLASS_NAMES):
                continue
            if width <= 0 or height <= 0:
                continue

            class_id = category_id - 1
            x1 = max(0, min(int(round(left)), image_width))
            y1 = max(0, min(int(round(top)), image_height))
            x2 = max(0, min(int(round(left + width)), image_width))
            y2 = max(0, min(int(round(top + height)), image_height))
            if x2 <= x1 or y2 <= y1:
                continue

            color = class_color(class_id)
            label = VISDRONE_CLASS_NAMES[class_id]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            text_y = max(y1, text_height + baseline + 4)
            cv2.rectangle(
                image,
                (x1, text_y - text_height - baseline - 4),
                (x1 + text_width + 4, text_y),
                color,
                -1,
            )
            cv2.putText(
                image,
                label,
                (x1 + 2, text_y - baseline - 2),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA,
            )
            object_count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Không thể lưu ảnh ground truth: {output_path}")
    return object_count


def show_results(prediction_path: Path, ground_truth_path: Path) -> None:
    prediction = cv2.imread(str(prediction_path), cv2.IMREAD_COLOR)
    ground_truth = cv2.imread(str(ground_truth_path), cv2.IMREAD_COLOR)
    if prediction is None:
        raise RuntimeError(f"Không thể đọc ảnh prediction: {prediction_path}")
    if ground_truth is None:
        raise RuntimeError(
            f"Không thể đọc ảnh ground truth: {ground_truth_path}"
        )

    prediction_window = "Prediction"
    ground_truth_window = "Ground truth"
    max_window_width = 620
    prediction_scale = min(
        max_window_width / prediction.shape[1], 1.0
    )
    ground_truth_scale = min(
        max_window_width / ground_truth.shape[1], 1.0
    )
    prediction_size = (
        int(round(prediction.shape[1] * prediction_scale)),
        int(round(prediction.shape[0] * prediction_scale)),
    )
    ground_truth_size = (
        int(round(ground_truth.shape[1] * ground_truth_scale)),
        int(round(ground_truth.shape[0] * ground_truth_scale)),
    )
    try:
        cv2.namedWindow(prediction_window, cv2.WINDOW_NORMAL)
        cv2.namedWindow(ground_truth_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(prediction_window, *prediction_size)
        cv2.resizeWindow(ground_truth_window, *ground_truth_size)
        cv2.moveWindow(prediction_window, 10, 40)
        cv2.moveWindow(
            ground_truth_window, prediction_size[0] + 30, 40
        )
        cv2.imshow(prediction_window, prediction)
        cv2.imshow(ground_truth_window, ground_truth)
        cv2.waitKey(0)
    except cv2.error as error:
        raise RuntimeError(
            "Không thể mở cửa sổ hiển thị. Hãy chạy trong môi trường desktop "
            "có OpenCV GUI (không dùng opencv-python-headless)."
        ) from error
    finally:
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    validate_args(args)

    checkpoint_path = args.checkpoint.expanduser().resolve()
    image_path = args.image.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    annotation_path = resolve_annotation_path(image_path, args.annotation)
    ground_truth_output = (
        args.ground_truth_output.expanduser().resolve()
        if args.ground_truth_output is not None
        else output_path.with_name(
            f"{output_path.stem}_ground_truth{output_path.suffix}"
        )
    )
    device = choose_device(args.device)

    print(f"Device: {device}")
    detections = predict_image(
        checkpoint_path=checkpoint_path,
        image_path=image_path,
        output_path=output_path,
        image_size=args.image_size,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        max_detections=args.max_det,
        device=device,
    )

    print(f"Số detection: {detections.shape[0]}")
    for detection in detections.cpu().tolist():
        x1, y1, x2, y2, confidence, raw_class_id = detection
        class_id = int(raw_class_id)
        class_name = (
            VISDRONE_CLASS_NAMES[class_id]
            if 0 <= class_id < len(VISDRONE_CLASS_NAMES)
            else f"class_{class_id}"
        )
        print(
            f"- {class_name}: conf={confidence:.4f}, "
            f"box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
        )

    ground_truth_count = draw_ground_truth(
        image_path=image_path,
        annotation_path=annotation_path,
        output_path=ground_truth_output,
    )

    print(f"Số ground-truth object: {ground_truth_count}")
    print(f"Đã lưu ảnh prediction: {output_path}")
    print(f"Đã lưu ảnh ground truth: {ground_truth_output}")
    print("Nhấn phím bất kỳ trong một cửa sổ ảnh để đóng cả hai.")
    show_results(output_path, ground_truth_output)


if __name__ == "__main__":
    main()
