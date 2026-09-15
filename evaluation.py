from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class ClassMetrics:
    precision: float
    recall: float
    f1: float
    ap50: float
    ap50_95: float
    targets: int
    predictions: int


@dataclass(frozen=True)
class DetectionMetrics:
    precision: float
    recall: float
    f1: float
    map50: float
    map50_95: float
    per_class: dict[str, ClassMetrics]


@dataclass(frozen=True)
class ValidationResult:
    loss: float
    metrics: DetectionMetrics

    @property
    def precision(self) -> float:
        return self.metrics.precision

    @property
    def recall(self) -> float:
        return self.metrics.recall

    @property
    def f1(self) -> float:
        return self.metrics.f1

    @property
    def map50(self) -> float:
        return self.metrics.map50

    @property
    def map50_95(self) -> float:
        return self.metrics.map50_95


def xywh_to_xyxy(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Đổi normalized xywh sang xyxy pixel trên ảnh letterbox."""
    converted = boxes.clone()
    converted[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * width
    converted[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * height
    converted[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * width
    converted[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * height
    return converted


def pairwise_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=2)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp(min=0).prod(dim=1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp(min=0).prod(dim=1)
    return intersection / (area1[:, None] + area2[None, :] - intersection + 1e-7)


def match_predictions(
    detections: torch.Tensor,
    target_boxes: torch.Tensor,
    target_classes: torch.Tensor,
    iou_thresholds: torch.Tensor,
) -> torch.Tensor:
    """Ghép prediction với GT cùng class, mỗi GT chỉ được dùng một lần."""
    correct = torch.zeros(
        (detections.shape[0], iou_thresholds.numel()),
        dtype=torch.bool,
        device=detections.device,
    )
    if detections.shape[0] == 0 or target_boxes.shape[0] == 0:
        return correct

    ious = pairwise_box_iou(detections[:, :4], target_boxes)
    class_matches = detections[:, 5:6].long() == target_classes[None, :].long()
    # postprocess trả prediction theo confidence giảm dần. Ghép theo thứ tự đó
    # để metric tại confidence cố định không bị prediction confidence thấp chiếm GT.
    for threshold_index, threshold in enumerate(iou_thresholds):
        matched_targets: set[int] = set()
        for prediction_index in range(detections.shape[0]):
            candidate_ious = ious[prediction_index].clone()
            candidate_ious[~class_matches[prediction_index]] = -1
            if matched_targets:
                used = torch.tensor(
                    tuple(matched_targets),
                    device=detections.device,
                    dtype=torch.long,
                )
                candidate_ious[used] = -1
            best_iou, target_index = candidate_ious.max(dim=0)
            if best_iou >= threshold:
                correct[prediction_index, threshold_index] = True
                matched_targets.add(int(target_index.item()))
    return correct


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Tính AP bằng nội suy 101 điểm như protocol COCO."""
    sample_points = np.linspace(0.0, 1.0, 101)
    interpolated = np.zeros_like(sample_points)
    for index, recall_level in enumerate(sample_points):
        valid = recall >= recall_level
        if valid.any():
            interpolated[index] = precision[valid].max()
    return float(interpolated.mean())


class DetectionMetricAccumulator:
    def __init__(
        self,
        class_names: tuple[str, ...],
        f1_confidence_threshold: float,
    ) -> None:
        self.class_names = class_names
        self.f1_confidence_threshold = f1_confidence_threshold
        self.iou_thresholds = torch.arange(0.5, 0.96, 0.05)
        self._correct: list[torch.Tensor] = []
        self._scores: list[torch.Tensor] = []
        self._prediction_classes: list[torch.Tensor] = []
        self._target_classes: list[torch.Tensor] = []

    def update(
        self,
        detections: torch.Tensor,
        target_boxes: torch.Tensor,
        target_classes: torch.Tensor,
    ) -> None:
        thresholds = self.iou_thresholds.to(detections.device)
        correct = match_predictions(
            detections=detections,
            target_boxes=target_boxes,
            target_classes=target_classes,
            iou_thresholds=thresholds,
        )
        self._correct.append(correct.cpu())
        self._scores.append(detections[:, 4].detach().float().cpu())
        self._prediction_classes.append(detections[:, 5].detach().long().cpu())
        self._target_classes.append(target_classes.detach().long().cpu())

    def compute(self) -> DetectionMetrics:
        correct = torch.cat(self._correct).numpy()
        scores = torch.cat(self._scores).numpy()
        prediction_classes = torch.cat(self._prediction_classes).numpy()
        target_classes = torch.cat(self._target_classes).numpy()

        order = np.argsort(-scores, kind="stable")
        correct = correct[order]
        scores = scores[order]
        prediction_classes = prediction_classes[order]

        total_tp = total_fp = total_fn = 0
        ap_values: list[np.ndarray] = []
        per_class: dict[str, ClassMetrics] = {}

        for class_id, class_name in enumerate(self.class_names):
            prediction_mask = prediction_classes == class_id
            target_count = int((target_classes == class_id).sum())
            class_correct = correct[prediction_mask]
            class_scores = scores[prediction_mask]

            threshold_mask = class_scores >= self.f1_confidence_threshold
            tp = int(class_correct[threshold_mask, 0].sum())
            prediction_count = int(threshold_mask.sum())
            fp = prediction_count - tp
            fn = target_count - tp
            total_tp += tp
            total_fp += fp
            total_fn += fn

            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-16)

            class_ap = np.zeros(self.iou_thresholds.numel(), dtype=np.float64)
            if target_count > 0 and class_correct.shape[0] > 0:
                true_positives = np.cumsum(class_correct, axis=0)
                false_positives = np.cumsum(~class_correct, axis=0)
                recall_curve = true_positives / target_count
                precision_curve = true_positives / np.maximum(
                    true_positives + false_positives, 1e-16
                )
                for iou_index in range(self.iou_thresholds.numel()):
                    class_ap[iou_index] = compute_ap(
                        recall_curve[:, iou_index],
                        precision_curve[:, iou_index],
                    )
            if target_count > 0:
                ap_values.append(class_ap)

            per_class[class_name] = ClassMetrics(
                precision=precision,
                recall=recall,
                f1=f1,
                ap50=float(class_ap[0]),
                ap50_95=float(class_ap.mean()),
                targets=target_count,
                predictions=prediction_count,
            )

        precision = total_tp / max(total_tp + total_fp, 1)
        recall = total_tp / max(total_tp + total_fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-16)
        mean_ap = np.stack(ap_values).mean(axis=0) if ap_values else np.zeros(10)
        return DetectionMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            map50=float(mean_ap[0]),
            map50_95=float(mean_ap.mean()),
            per_class=per_class,
        )


METRICS_FIELDS = (
    "round",
    "train_loss",
    "val_loss",
    "precision",
    "recall",
    "f1",
    "map50",
    "map50_95",
    "learning_rate",
    "is_best",
)


def initialize_evaluation_files(output_dir: Path) -> None:
    """Khởi tạo file metric cho một lần train mới, tránh lẫn round cũ."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as metrics_file:
        csv.DictWriter(metrics_file, fieldnames=METRICS_FIELDS).writeheader()
    (output_dir / "metrics_per_class.jsonl").write_text(
        "", encoding="utf-8"
    )


def append_evaluation_files(
    output_dir: Path,
    round_index: int,
    train_loss: float,
    result: ValidationResult,
    learning_rate: float,
    is_best: bool,
) -> None:
    """Ghi metric tổng hợp dạng CSV và metric từng class dạng JSONL."""
    csv_path = output_dir / "metrics.csv"
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=METRICS_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "round": round_index,
                "train_loss": train_loss,
                "val_loss": result.loss,
                "precision": result.precision,
                "recall": result.recall,
                "f1": result.f1,
                "map50": result.map50,
                "map50_95": result.map50_95,
                "learning_rate": learning_rate,
                "is_best": is_best,
            }
        )

    per_class_path = output_dir / "metrics_per_class.jsonl"
    payload = {
        "round": round_index,
        "classes": {
            name: asdict(metrics)
            for name, metrics in result.metrics.per_class.items()
        },
    }
    with per_class_path.open("a", encoding="utf-8") as per_class_file:
        per_class_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
