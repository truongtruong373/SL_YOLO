from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

if __package__:
    from .model import MyYOLODetectionModel
else:
    from model import MyYOLODetectionModel


TRAINING_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    TRAINING_DIR / "runs" / "ppe_backbone" / "best.pt"
)
DEFAULT_IMAGE = (
    TRAINING_DIR / "data" / "images" / "test" / "image633.jpg"
)
DEFAULT_OUTPUT = (
    TRAINING_DIR / "runs" / "ppe_backbone" / "prediction.jpg"
)

PPE_CLASS_NAMES = (
    "helmet",
    "gloves",
    "vest",
    "boots",
    "goggles",
    "none",
    "Person",
    "no_helmet",
    "no_goggle",
    "no_gloves",
    "no_boots",
)


def choose_device(requested_device: str) -> torch.device:
    if requested_device != "auto":
        device = torch.device(requested_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA không khả dụng.")
        if (
            device.type == "mps"
            and (
                not hasattr(torch.backends, "mps")
                or not torch.backends.mps.is_available()
            )
        ):
            raise RuntimeError("MPS không khả dụng.")
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def letterbox_image(
    image: np.ndarray,
    image_size: int,
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Resize ảnh giữ nguyên tỉ lệ và thêm padding.

    Returns:
        resized_image:
            Ảnh BGR [image_size, image_size, 3].
        ratio:
            Tỉ lệ resize.
        padding:
            Padding thực tế bên trái và bên trên.
    """
    original_height, original_width = image.shape[:2]
    ratio = min(
        image_size / original_height,
        image_size / original_width,
    )

    resized_width = int(round(original_width * ratio))
    resized_height = int(round(original_height * ratio))

    if (
        resized_width != original_width
        or resized_height != original_height
    ):
        image = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )

    padding_width = image_size - resized_width
    padding_height = image_size - resized_height

    left = int(round(padding_width / 2 - 0.1))
    right = int(round(padding_width / 2 + 0.1))
    top = int(round(padding_height / 2 - 0.1))
    bottom = int(round(padding_height / 2 + 0.1))

    image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color,
    )
    return image, ratio, (left, top)


def preprocess_image(
    image_path: Path,
    image_size: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    np.ndarray,
    float,
    tuple[int, int],
]:
    original_image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )
    if original_image is None:
        raise RuntimeError(f"Không thể đọc ảnh: {image_path}")

    processed_image, ratio, padding = letterbox_image(
        original_image,
        image_size=image_size,
    )

    # BGR HWC uint8 -> RGB BCHW float32 [0, 1].
    processed_image = np.ascontiguousarray(
        processed_image[:, :, ::-1].transpose(2, 0, 1)
    )
    image_tensor = (
        torch.from_numpy(processed_image)
        .to(device=device)
        .float()
        .div_(255.0)
        .unsqueeze(0)
    )

    return image_tensor, original_image, ratio, padding


def load_trained_model(
    checkpoint_path: Path,
    device: torch.device,
) -> MyYOLODetectionModel:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise TypeError(
            "best.pt phải là state_dict hoặc checkpoint chứa "
            "'model_state_dict'."
        )

    model = MyYOLODetectionModel(
        nc=len(PPE_CLASS_NAMES)
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def make_anchor_points(
    level_shapes: list[tuple[int, int]],
    strides: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_anchor_points = []
    all_strides = []

    for (height, width), stride in zip(
        level_shapes,
        strides,
        strict=True,
    ):
        y_coordinates = (
            torch.arange(height, device=device, dtype=dtype) + 0.5
        )
        x_coordinates = (
            torch.arange(width, device=device, dtype=dtype) + 0.5
        )
        grid_y, grid_x = torch.meshgrid(
            y_coordinates,
            x_coordinates,
            indexing="ij",
        )

        all_anchor_points.append(
            torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)
        )
        all_strides.append(
            torch.full(
                (height * width, 1),
                float(stride),
                device=device,
                dtype=dtype,
            )
        )

    return (
        torch.cat(all_anchor_points, dim=0),
        torch.cat(all_strides, dim=0),
    )


def decode_predictions(
    raw_outputs: list[tuple[torch.Tensor, torch.Tensor]],
    reg_max: int = 16,
    strides: tuple[int, ...] = (8, 16, 32),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Decode output training model.

    Returns:
        boxes:
            Tensor [B, A, 4] dạng pixel xyxy trên ảnh letterbox.
        class_scores:
            Tensor [B, A, C], đã sigmoid.
    """
    if len(raw_outputs) != len(strides):
        raise ValueError(
            "Số feature level không khớp với số stride."
        )

    batch_size = raw_outputs[0][0].shape[0]
    level_shapes = [
        tuple(box_logits.shape[-2:])
        for box_logits, _ in raw_outputs
    ]

    box_logits = torch.cat(
        [
            level_box.reshape(batch_size, 4 * reg_max, -1)
            for level_box, _ in raw_outputs
        ],
        dim=-1,
    )
    class_logits = torch.cat(
        [
            level_class.reshape(
                batch_size,
                len(PPE_CLASS_NAMES),
                -1,
            )
            for _, level_class in raw_outputs
        ],
        dim=-1,
    )

    distribution = box_logits.reshape(
        batch_size,
        4,
        reg_max,
        -1,
    ).softmax(dim=2)
    projection = torch.arange(
        reg_max,
        device=box_logits.device,
        dtype=box_logits.dtype,
    ).view(1, 1, reg_max, 1)
    distances = (distribution * projection).sum(dim=2)
    distances = distances.permute(0, 2, 1)

    anchor_points, stride_tensor = make_anchor_points(
        level_shapes=level_shapes,
        strides=strides,
        device=box_logits.device,
        dtype=box_logits.dtype,
    )

    top_left = anchor_points - distances[..., :2]
    bottom_right = anchor_points + distances[..., 2:]
    boxes = torch.cat((top_left, bottom_right), dim=-1)
    boxes = boxes * stride_tensor

    class_scores = class_logits.sigmoid().permute(0, 2, 1)
    return boxes, class_scores


def box_iou(
    box: torch.Tensor,
    boxes: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    intersection_top_left = torch.maximum(
        box[:2],
        boxes[:, :2],
    )
    intersection_bottom_right = torch.minimum(
        box[2:],
        boxes[:, 2:],
    )
    intersection_size = (
        intersection_bottom_right - intersection_top_left
    ).clamp(min=0)
    intersection_area = intersection_size.prod(dim=-1)

    box_area = (
        (box[2:] - box[:2]).clamp(min=0).prod()
    )
    boxes_area = (
        (boxes[:, 2:] - boxes[:, :2])
        .clamp(min=0)
        .prod(dim=-1)
    )
    union_area = box_area + boxes_area - intersection_area
    return intersection_area / (union_area + eps)


def non_maximum_suppression(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float,
    max_detections: int,
    max_nms_candidates: int = 3000,
) -> torch.Tensor:
    """Class-aware NMS thuần PyTorch."""
    if boxes.shape[0] == 0:
        return torch.empty(
            (0,),
            dtype=torch.long,
            device=boxes.device,
        )

    sorted_indices = scores.argsort(descending=True)
    sorted_indices = sorted_indices[:max_nms_candidates]

    coordinate_range = (
        boxes.max() - boxes.min()
    ).clamp_min(1.0) + 1.0
    offset_boxes = (
        boxes
        + class_ids[:, None].to(boxes.dtype) * coordinate_range
    )

    kept_indices: list[torch.Tensor] = []
    while sorted_indices.numel() > 0:
        current_index = sorted_indices[0]
        kept_indices.append(current_index)

        if (
            sorted_indices.numel() == 1
            or len(kept_indices) >= max_detections
        ):
            break

        remaining_indices = sorted_indices[1:]
        overlaps = box_iou(
            offset_boxes[current_index],
            offset_boxes[remaining_indices],
        )
        sorted_indices = remaining_indices[
            overlaps <= iou_threshold
        ]

    return torch.stack(kept_indices)


def postprocess(
    boxes: torch.Tensor,
    class_scores: torch.Tensor,
    confidence_threshold: float,
    iou_threshold: float,
    max_detections: int,
) -> torch.Tensor:
    """Trả về detections [N, 6]: x1, y1, x2, y2, conf, cls."""
    boxes = boxes[0]
    class_scores = class_scores[0]

    confidence, class_ids = class_scores.max(dim=-1)
    confidence_mask = confidence >= confidence_threshold

    boxes = boxes[confidence_mask]
    confidence = confidence[confidence_mask]
    class_ids = class_ids[confidence_mask]

    if boxes.shape[0] == 0:
        return torch.empty(
            (0, 6),
            device=class_scores.device,
            dtype=class_scores.dtype,
        )

    kept_indices = non_maximum_suppression(
        boxes=boxes,
        scores=confidence,
        class_ids=class_ids,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
    )

    return torch.cat(
        (
            boxes[kept_indices],
            confidence[kept_indices, None],
            class_ids[kept_indices, None].to(boxes.dtype),
        ),
        dim=-1,
    )


def scale_boxes_to_original(
    detections: torch.Tensor,
    ratio: float,
    padding: tuple[int, int],
    original_shape: tuple[int, int],
) -> torch.Tensor:
    if detections.shape[0] == 0:
        return detections

    left, top = padding
    original_height, original_width = original_shape

    detections[:, [0, 2]] -= left
    detections[:, [1, 3]] -= top
    detections[:, :4] /= ratio

    detections[:, [0, 2]].clamp_(0, original_width)
    detections[:, [1, 3]].clamp_(0, original_height)
    return detections


def class_color(class_id: int) -> tuple[int, int, int]:
    """Màu BGR ổn định cho từng class."""
    return (
        int((37 * class_id + 80) % 255),
        int((17 * class_id + 160) % 255),
        int((29 * class_id + 220) % 255),
    )


def draw_detections(
    image: np.ndarray,
    detections: torch.Tensor,
) -> np.ndarray:
    result = image.copy()

    for detection in detections.cpu().tolist():
        x1, y1, x2, y2, confidence, raw_class_id = detection
        class_id = int(raw_class_id)
        color = class_color(class_id)
        label = (
            f"{PPE_CLASS_NAMES[class_id]} {confidence:.2f}"
            if 0 <= class_id < len(PPE_CLASS_NAMES)
            else f"class_{class_id} {confidence:.2f}"
        )

        point_1 = (int(round(x1)), int(round(y1)))
        point_2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(result, point_1, point_2, color, thickness=2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness,
        )

        text_x = max(point_1[0], 0)
        text_y = max(point_1[1], text_height + baseline + 4)
        cv2.rectangle(
            result,
            (text_x, text_y - text_height - baseline - 4),
            (text_x + text_width + 4, text_y),
            color,
            thickness=-1,
        )
        cv2.putText(
            result,
            label,
            (text_x + 2, text_y - baseline - 2),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            lineType=cv2.LINE_AA,
        )

    return result


def predict_image(
    checkpoint_path: Path,
    image_path: Path,
    output_path: Path,
    image_size: int,
    confidence_threshold: float,
    iou_threshold: float,
    max_detections: int,
    device: torch.device,
) -> torch.Tensor:
    model = load_trained_model(checkpoint_path, device)
    (
        image_tensor,
        original_image,
        ratio,
        padding,
    ) = preprocess_image(
        image_path=image_path,
        image_size=image_size,
        device=device,
    )

    with torch.inference_mode():
        raw_outputs = model(image_tensor)
        boxes, class_scores = decode_predictions(raw_outputs)
        detections = postprocess(
            boxes=boxes,
            class_scores=class_scores,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )
        detections = scale_boxes_to_original(
            detections=detections,
            ratio=ratio,
            padding=padding,
            original_shape=original_image.shape[:2],
        )

    result_image = draw_detections(
        image=original_image,
        detections=detections,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), result_image):
        raise RuntimeError(f"Không thể lưu ảnh: {output_path}")

    return detections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load best.pt và thử nhận diện trên một ảnh PPE."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
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


if __name__ == "__main__":
    arguments = parse_args()
    validate_args(arguments)

    inference_device = choose_device(arguments.device)
    checkpoint_path = arguments.checkpoint.expanduser().resolve()
    image_path = arguments.image.expanduser().resolve()
    output_path = arguments.output.expanduser().resolve()

    print(f"Device: {inference_device}")
    detections = predict_image(
        checkpoint_path=checkpoint_path,
        image_path=image_path,
        output_path=output_path,
        image_size=arguments.image_size,
        confidence_threshold=arguments.conf,
        iou_threshold=arguments.iou,
        max_detections=arguments.max_det,
        device=inference_device,
    )

    print(f"Số detection: {detections.shape[0]}")
    for detection in detections.cpu().tolist():
        x1, y1, x2, y2, confidence, raw_class_id = detection
        class_id = int(raw_class_id)
        class_name = (
            PPE_CLASS_NAMES[class_id]
            if 0 <= class_id < len(PPE_CLASS_NAMES)
            else f"class_{class_id}"
        )
        print(
            f"- {class_name}: conf={confidence:.4f}, "
            f"box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
        )
    print(f"Đã lưu kết quả: {output_path}")
