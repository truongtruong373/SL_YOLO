from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Chuyển bounding box:
        [cx, cy, width, height]

    thành:
        [x1, y1, x2, y2]
    """
    center = boxes[..., :2]
    half_size = boxes[..., 2:] / 2

    top_left = center - half_size
    bottom_right = center + half_size

    return torch.cat((top_left, bottom_right), dim=-1)


def dist_to_bbox(
    distances: torch.Tensor,
    anchor_points: torch.Tensor,
) -> torch.Tensor:
    """
    Decode khoảng cách left, top, right, bottom thành xyxy.
    Args:
        distances:
            [B, A, 4]

        anchor_points:
            [A, 2]

    Returns:
        boxes:
            [B, A, 4]
    """
    left_top = distances[..., :2]
    right_bottom = distances[..., 2:]

    top_left = anchor_points - left_top
    bottom_right = anchor_points + right_bottom

    return torch.cat((top_left, bottom_right), dim=-1)


def bbox_to_dist(
    anchor_points: torch.Tensor,
    boxes: torch.Tensor,
    reg_max: int,
) -> torch.Tensor:
    """
    Encode xyxy box thành khoảng cách left, top, right, bottom.
    Args:
        anchor_points:
            [A, 2]

        boxes:
            [B, A, 4]

        reg_max:
            Số bin DFL, ví dụ 16.

    Returns:
        distances:
            [B, A, 4]
    """

    left_top = anchor_points - boxes[..., :2]
    right_bottom = boxes[..., 2:] - anchor_points

    distances = torch.cat((left_top, right_bottom),dim=-1)

    return distances.clamp(min=0, max=reg_max - 1 - 0.01, )


def pairwise_iou(
    gt_boxes: torch.Tensor,
    pred_boxes: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Tính IoU giữa tất cả ground-truth và prediction.

    Args:
        gt_boxes:
            [B, M, 4]

        pred_boxes:
            [B, A, 4]

    Returns:
        iou:
            [B, M, A]
    """

    gt = gt_boxes.unsqueeze(2)
    pred = pred_boxes.unsqueeze(1)

    intersection_top_left = torch.maximum(
        gt[..., :2],
        pred[..., :2],
    )

    intersection_bottom_right = torch.minimum(
        gt[..., 2:],
        pred[..., 2:],
    )

    intersection_size = (
        intersection_bottom_right - intersection_top_left
    ).clamp(min=0)

    intersection_area = intersection_size.prod(dim=-1)

    gt_size = (
        gt_boxes[..., 2:] - gt_boxes[..., :2]
    ).clamp(min=0)

    pred_size = (
        pred_boxes[..., 2:] - pred_boxes[..., :2]
    ).clamp(min=0)

    gt_area = gt_size.prod(dim=-1).unsqueeze(-1)
    pred_area = pred_size.prod(dim=-1).unsqueeze(1)

    union_area = gt_area + pred_area - intersection_area

    return intersection_area / (union_area + eps)


def complete_iou(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """va
    Complete IoU.

    Args:
        pred_boxes:
            [N, 4]

        target_boxes:
            [N, 4]

    Returns:
        ciou:
            [N]
    """

    pred_top_left = pred_boxes[..., :2]
    pred_bottom_right = pred_boxes[..., 2:]

    target_top_left = target_boxes[..., :2]
    target_bottom_right = target_boxes[..., 2:]

    intersection_top_left = torch.maximum(
        pred_top_left,
        target_top_left,
    )

    intersection_bottom_right = torch.minimum(
        pred_bottom_right,
        target_bottom_right,
    )

    intersection_size = (
        intersection_bottom_right - intersection_top_left
    ).clamp(min=0)

    intersection_area = intersection_size.prod(dim=-1)

    pred_size = (
        pred_bottom_right - pred_top_left
    ).clamp(min=eps)

    target_size = (
        target_bottom_right - target_top_left
    ).clamp(min=eps)

    pred_area = pred_size.prod(dim=-1)
    target_area = target_size.prod(dim=-1)

    union_area = pred_area + target_area - intersection_area
    iou = intersection_area / (union_area + eps)

    pred_center = (
        pred_top_left + pred_bottom_right
    ) / 2

    target_center = (
        target_top_left + target_bottom_right
    ) / 2

    center_distance = (
        pred_center - target_center
    ).pow(2).sum(dim=-1)

    enclosing_top_left = torch.minimum(
        pred_top_left,
        target_top_left,
    )

    enclosing_bottom_right = torch.maximum(
        pred_bottom_right,
        target_bottom_right,
    )

    enclosing_diagonal = (
        enclosing_bottom_right - enclosing_top_left
    ).pow(2).sum(dim=-1)

    pred_width = pred_size[..., 0]
    pred_height = pred_size[..., 1]

    target_width = target_size[..., 0]
    target_height = target_size[..., 1]

    aspect_ratio_term = (
        4
        / torch.pi**2
        * (
            torch.atan(target_width / target_height)
            - torch.atan(pred_width / pred_height)
        ).pow(2)
    )

    with torch.no_grad():
        alpha = aspect_ratio_term / (
            1 - iou + aspect_ratio_term + eps
        )

    ciou = (
        iou
        - center_distance / (enclosing_diagonal + eps)
        - alpha * aspect_ratio_term
    )

    return ciou


# =========================================================
# ANCHOR GENERATION
# =========================================================

def make_anchors(
    level_shapes: list[tuple[int, int]],
    strides: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    offset: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Tạo anchor point cho các feature level.

    Returns:
        anchor_points:
            [A, 2], theo feature-grid coordinates.

        stride_tensor:
            [A, 1].
    """

    all_anchor_points = []
    all_strides = []

    for (height, width), stride in zip(
        level_shapes,
        strides,
        strict=True,
    ):
        y_coordinates = torch.arange(
            height,
            device=device,
            dtype=dtype,
        )

        x_coordinates = torch.arange(
            width,
            device=device,
            dtype=dtype,
        )

        grid_y, grid_x = torch.meshgrid(
            y_coordinates,
            x_coordinates,
            indexing="ij",
        )

        anchor_points = torch.stack(
            (
                grid_x + offset,
                grid_y + offset,
            ),
            dim=-1,
        ).reshape(-1, 2)

        stride_values = torch.full(
            (height * width, 1),
            float(stride),
            device=device,
            dtype=dtype,
        )

        all_anchor_points.append(anchor_points)
        all_strides.append(stride_values)

    return (
        torch.cat(all_anchor_points, dim=0),
        torch.cat(all_strides, dim=0),
    )


# =========================================================
# TASK-ALIGNED ASSIGNER
# =========================================================

class TaskAlignedAssigner(nn.Module):
    """
    Gán ground-truth cho anchor theo:

        alignment_metric =
            class_score^alpha * IoU^beta
    """

    def __init__(
        self,
        num_classes: int,
        topk: int = 10,
        alpha: float = 0.5,
        beta: float = 6.0,
        eps: float = 1e-9,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @staticmethod
    def anchors_inside_gt(
        anchor_points: torch.Tensor,
        gt_boxes: torch.Tensor,
        eps: float = 1e-9,
    ) -> torch.Tensor:
        """
        Args:
            anchor_points:
                [A, 2]

            gt_boxes:
                [B, M, 4]

        Returns:
            mask:
                [B, M, A]
        """

        anchors = anchor_points.view(1, 1, -1, 2)

        left_top_distance = (
            anchors - gt_boxes[..., None, :2]
        )

        right_bottom_distance = (
            gt_boxes[..., None, 2:] - anchors
        )

        distances = torch.cat(
            (
                left_top_distance,
                right_bottom_distance,
            ),
            dim=-1,
        )

        return distances.amin(dim=-1) > eps

    def forward(
        self,
        pred_scores: torch.Tensor,
        pred_boxes: torch.Tensor,
        anchor_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_boxes: torch.Tensor,
        valid_gt_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Args:
            pred_scores:
                [B, A, C], đã sigmoid.

            pred_boxes:
                [B, A, 4], pixel xyxy.

            anchor_points:
                [A, 2], pixel coordinates.

            gt_labels:
                [B, M, 1].

            gt_boxes:
                [B, M, 4], pixel xyxy.

            valid_gt_mask:
                [B, M, 1].

        Returns:
            target_labels:
                [B, A]

            target_boxes:
                [B, A, 4]

            target_scores:
                [B, A, C]

            foreground_mask:
                [B, A]
        """

        batch_size, num_anchors, _ = pred_scores.shape
        num_gt = gt_boxes.shape[1]

        if num_gt == 0:
            return (
                torch.zeros(
                    batch_size,
                    num_anchors,
                    device=pred_scores.device,
                    dtype=torch.long,
                ),
                torch.zeros(
                    batch_size,
                    num_anchors,
                    4,
                    device=pred_boxes.device,
                    dtype=pred_boxes.dtype,
                ),
                torch.zeros_like(pred_scores),
                torch.zeros(
                    batch_size,
                    num_anchors,
                    device=pred_scores.device,
                    dtype=torch.bool,
                ),
            )

        overlaps = pairwise_iou(
            gt_boxes,
            pred_boxes,
        ).clamp(min=0)

        gt_class_indices = (
            gt_labels.squeeze(-1)
            .long()
            .clamp(
                min=0,
                max=self.num_classes - 1,
            )
        )

        # pred_scores: [B, A, C] -> [B, C, A]
        scores_by_class = pred_scores.permute(0, 2, 1)

        class_indices = gt_class_indices.unsqueeze(-1).expand(
            -1,
            -1,
            num_anchors,
        )

        gt_class_scores = scores_by_class.gather(
            dim=1,
            index=class_indices,
        )

        alignment_metric = (
            gt_class_scores.pow(self.alpha)
            * overlaps.pow(self.beta)
        )

        inside_mask = self.anchors_inside_gt(
            anchor_points,
            gt_boxes,
        )

        candidate_mask = (
            inside_mask
            & valid_gt_mask.bool()
        )

        candidate_metric = (
            alignment_metric
            * candidate_mask.to(alignment_metric.dtype)
        )

        topk = min(self.topk, num_anchors)

        topk_metrics, topk_indices = torch.topk(
            candidate_metric,
            k=topk,
            dim=-1,
            largest=True,
        )

        valid_topk = (
            topk_metrics > self.eps
        ) & valid_gt_mask.bool()

        topk_mask = torch.zeros_like(
            candidate_metric
        )

        topk_mask.scatter_add_(
            dim=-1,
            index=topk_indices,
            src=valid_topk.to(topk_mask.dtype),
        )

        positive_mask = (
            topk_mask
            * candidate_mask.to(topk_mask.dtype)
        )

        # Một anchor có thể được nhiều GT chọn.
        # Khi đó lấy GT có IoU lớn nhất.
        positive_count = positive_mask.sum(dim=1)

        multiple_gt_mask = positive_count > 1

        if multiple_gt_mask.any():
            best_gt_indices = overlaps.argmax(dim=1)

            best_gt_mask = F.one_hot(
                best_gt_indices,
                num_classes=num_gt,
            ).permute(0, 2, 1)

            positive_mask = torch.where(
                multiple_gt_mask.unsqueeze(1),
                best_gt_mask.to(positive_mask.dtype),
                positive_mask,
            )

        foreground_mask = (
            positive_mask.sum(dim=1) > 0
        )

        target_gt_indices = positive_mask.argmax(dim=1)

        batch_indices = torch.arange(
            batch_size,
            device=pred_scores.device,
        ).unsqueeze(1)

        flat_indices = (
            target_gt_indices
            + batch_indices * num_gt
        )

        flat_gt_labels = gt_class_indices.reshape(-1)
        flat_gt_boxes = gt_boxes.reshape(-1, 4)

        target_labels = flat_gt_labels[
            flat_indices
        ]

        target_boxes = flat_gt_boxes[
            flat_indices
        ]

        target_scores = F.one_hot(
            target_labels,
            num_classes=self.num_classes,
        ).to(pred_scores.dtype)

        target_scores = (
            target_scores
            * foreground_mask.unsqueeze(-1)
        )

        # Chuẩn hóa target score bằng alignment metric và IoU.
        alignment_metric = (
            alignment_metric * positive_mask
        )

        max_alignment = alignment_metric.amax(
            dim=-1,
            keepdim=True,
        )

        max_overlap = (
            overlaps * positive_mask
        ).amax(
            dim=-1,
            keepdim=True,
        )

        normalized_alignment = (
            alignment_metric
            * max_overlap
            / (max_alignment + self.eps)
        ).amax(
            dim=1,
        ).unsqueeze(-1)

        target_scores = (
            target_scores * normalized_alignment
        )

        return (
            target_labels,
            target_boxes,
            target_scores,
            foreground_mask,
        )


# =========================================================
# DISTRIBUTION FOCAL LOSS
# =========================================================

class DistributionFocalLoss(nn.Module):

    def __init__(self, reg_max: int) -> None:
        super().__init__()
        self.reg_max = reg_max

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_distances: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_logits:
                [N, 4, reg_max]

            target_distances:
                [N, 4]

        Returns:
            loss:
                [N]
        """

        targets = target_distances.clamp(min=0,max=self.reg_max - 1 - 0.01,)

        target_left = targets.floor().long()
        target_right = target_left + 1

        weight_left = (target_right.to(targets.dtype) - targets)

        weight_right = 1.0 - weight_left

        flattened_logits = pred_logits.reshape(-1,self.reg_max,)

        left_loss = F.cross_entropy(
            flattened_logits,
            target_left.reshape(-1),
            reduction="none",
        ).reshape(-1, 4)

        right_loss = F.cross_entropy(
            flattened_logits,
            target_right.reshape(-1),
            reduction="none",
        ).reshape(-1, 4)

        loss = (
            left_loss * weight_left
            + right_loss * weight_right
        )

        return loss.mean(dim=-1)


# =========================================================
# YOLO DETECTION LOSS
# =========================================================

@dataclass
class YOLOLossConfig:
    num_classes: int

    reg_max: int = 16
    strides: tuple[int, ...] = (8, 16, 32)

    box_gain: float = 7.5
    cls_gain: float = 0.5
    dfl_gain: float = 1.5

    assigner_topk: int = 10
    assigner_alpha: float = 0.5
    assigner_beta: float = 6.0

    scale_loss_by_batch: bool = True


class YOLODetectionLoss(nn.Module):
    """
    Loss YOLO-style chỉ sử dụng PyTorch.

    Prediction phải có dạng:

        predictions = [
            (box_p3, cls_p3),
            (box_p4, cls_p4),
            (box_p5, cls_p5),
        ]

    Trong đó:

        box_pi: [B, 4 * reg_max, H, W]
        cls_pi: [B, num_classes, H, W]

    Targets:

        targets = {
            "batch_idx": Tensor[N],
            "cls":       Tensor[N] hoặc Tensor[N, 1],
            "bboxes":    Tensor[N, 4],
        }

    bboxes là normalized xywh.
    """

    def __init__(self, config: YOLOLossConfig,) -> None:
        super().__init__()

        if config.num_classes <= 0:
            raise ValueError(
                "num_classes phải lớn hơn 0."
            )

        if config.reg_max <= 1:
            raise ValueError(
                "reg_max phải lớn hơn 1."
            )

        self.config = config
        self.num_classes = config.num_classes
        self.reg_max = config.reg_max

        self.assigner = TaskAlignedAssigner(
            num_classes=config.num_classes,
            topk=config.assigner_topk,
            alpha=config.assigner_alpha,
            beta=config.assigner_beta,
        )

        self.dfl = DistributionFocalLoss(
            reg_max=config.reg_max,
        )

        self.register_buffer(
            "projection",
            torch.arange(
                config.reg_max,
                dtype=torch.float32,
            ),
        )

    def flatten_predictions(self, predictions: list[tuple[torch.Tensor, torch.Tensor]],) -> tuple[torch.Tensor,torch.Tensor,list[tuple[int, int]]]:
        box_predictions = []
        class_predictions = []
        level_shapes = []

        if len(predictions) != len(self.config.strides):
            raise ValueError(
                "Số prediction level phải bằng số stride."
            )

        batch_size = predictions[0][0].shape[0]

        for level_index, (
            box_logits,
            class_logits,
        ) in enumerate(predictions):
            if box_logits.ndim != 4:
                raise ValueError(
                    f"Box level {level_index} "
                    "phải có shape [B, 4R, H, W]."
                )

            if class_logits.ndim != 4:
                raise ValueError(
                    f"Class level {level_index} "
                    "phải có shape [B, C, H, W]."
                )

            if box_logits.shape[0] != batch_size:
                raise ValueError(
                    "Batch size giữa các level không giống nhau."
                )

            if box_logits.shape[-2:] != class_logits.shape[-2:]:
                raise ValueError(
                    "Kích thước box và class map không giống nhau."
                )

            if box_logits.shape[1] != 4 * self.reg_max:
                raise ValueError(
                    f"Box head phải có {4 * self.reg_max} channel."
                )

            if class_logits.shape[1] != self.num_classes:
                raise ValueError(
                    f"Class head phải có {self.num_classes} channel."
                )

            height, width = box_logits.shape[-2:]
            level_shapes.append((height, width))

            box_predictions.append(
                box_logits.reshape(
                    batch_size,
                    4 * self.reg_max,
                    -1,
                )
            )

            class_predictions.append(
                class_logits.reshape(
                    batch_size,
                    self.num_classes,
                    -1,
                )
            )

        pred_dist = torch.cat(box_predictions,dim=-1,).permute(0,2,1,).contiguous()

        pred_scores = torch.cat(class_predictions, dim=-1,).permute(0,2,1,).contiguous()

        return pred_dist, pred_scores, level_shapes

    def decode_boxes(
        self,
        pred_dist: torch.Tensor,
        anchor_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_dist:
                [B, A, 4 * reg_max]

        Returns:
            boxes:
                [B, A, 4], theo feature-grid coordinates.
        """

        batch_size, num_anchors, _ = pred_dist.shape

        distribution = pred_dist.reshape(batch_size,num_anchors,4,self.reg_max)

        probabilities = distribution.softmax(dim=-1)

        projection = self.projection.to(
            device=pred_dist.device,
            dtype=pred_dist.dtype,
        )

        distances = torch.matmul(probabilities,projection,)

        return dist_to_bbox(distances, anchor_points)

    @staticmethod
    def preprocess_targets(
        batch_idx: torch.Tensor,
        classes: torch.Tensor,
        boxes: torch.Tensor,
        batch_size: int,
        image_width: float,
        image_height: float,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Chuyển target phẳng thành padded target.

        Returns:
            gt_labels:
                [B, M, 1]

            gt_boxes:
                [B, M, 4], pixel xyxy

            valid_mask:
                [B, M, 1]
        """

        device = boxes.device
        dtype = boxes.dtype

        batch_idx = batch_idx.reshape(-1).long()
        classes = classes.reshape(-1, 1)
        boxes = boxes.reshape(-1, 4)

        if boxes.shape[0] == 0:
            return (torch.zeros(batch_size,0,1,device=device,dtype=dtype),
                    torch.zeros(batch_size,0,4,device=device,dtype=dtype),
                    torch.zeros(batch_size,0,1,device=device,dtype=torch.bool),
            )

        if batch_idx.min() < 0:
            raise ValueError(
                "batch_idx không được âm."
            )

        if batch_idx.max() >= batch_size:
            raise ValueError(
                "batch_idx vượt quá batch size."
            )

        counts = torch.bincount(batch_idx, minlength=batch_size)

        max_objects = int(counts.max().item())

        gt_labels = torch.zeros(
            batch_size,
            max_objects,
            1,
            device=device,
            dtype=dtype,
        )

        gt_boxes = torch.zeros(
            batch_size,
            max_objects,
            4,
            device=device,
            dtype=dtype,
        )

        valid_mask = torch.zeros(
            batch_size,
            max_objects,
            1,
            device=device,
            dtype=torch.bool,
        )

        scale = torch.tensor(
            [
                image_width,
                image_height,
                image_width,
                image_height,
            ],
            device=device,
            dtype=dtype,
        )

        pixel_boxes = boxes * scale
        pixel_boxes = xywh_to_xyxy(pixel_boxes)

        for image_index in range(batch_size):
            image_mask = batch_idx == image_index
            num_objects = int(image_mask.sum().item())

            if num_objects == 0:
                continue

            gt_labels[image_index,:num_objects,] = classes[image_mask]

            gt_boxes[image_index,:num_objects,] = pixel_boxes[image_mask]

            valid_mask[image_index,:num_objects,] = True

        return gt_labels, gt_boxes, valid_mask

    def forward(
        self,
        predictions: list[
            tuple[torch.Tensor, torch.Tensor]
        ],
        targets: dict[str, torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        pred_dist, pred_scores, level_shapes = (self.flatten_predictions(predictions))

        device = pred_scores.device
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]

        anchor_points, stride_tensor = make_anchors(
            level_shapes=level_shapes,
            strides=self.config.strides,
            device=device,
            dtype=dtype,
        )

        pred_boxes_grid = self.decode_boxes(
            pred_dist,
            anchor_points,
        )

        pred_boxes_pixel = (
            pred_boxes_grid * stride_tensor
        )

        image_height = (
            level_shapes[0][0]
            * self.config.strides[0]
        )

        image_width = (
            level_shapes[0][1]
            * self.config.strides[0]
        )

        gt_labels, gt_boxes, valid_gt_mask = (
            self.preprocess_targets(
                batch_idx=targets["batch_idx"].to(device),
                classes=targets["cls"].to(
                    device=device,
                    dtype=dtype,
                ),
                boxes=targets["bboxes"].to(
                    device=device,
                    dtype=dtype,
                ),
                batch_size=batch_size,
                image_width=float(image_width),
                image_height=float(image_height),
            )
        )

        with torch.no_grad():
            (
                _,
                target_boxes_pixel,
                target_scores,
                foreground_mask,
            ) = self.assigner(
                pred_scores=pred_scores.detach().sigmoid(),
                pred_boxes=pred_boxes_pixel.detach(),
                anchor_points=(
                    anchor_points * stride_tensor
                ),
                gt_labels=gt_labels,
                gt_boxes=gt_boxes,
                valid_gt_mask=valid_gt_mask,
            )

        target_scores_sum = (
            target_scores.sum().clamp_min(1.0)
        )

        # ---------------------------------------------
        # Classification loss
        # ---------------------------------------------
        classification_loss = (
            F.binary_cross_entropy_with_logits(
                pred_scores,
                target_scores,
                reduction="sum",
            )
            / target_scores_sum
        )

        box_loss = pred_dist.sum() * 0.0
        dfl_loss = pred_dist.sum() * 0.0

        # ---------------------------------------------
        # Box loss và DFL
        # ---------------------------------------------
        if foreground_mask.any():
            target_boxes_grid = (
                target_boxes_pixel / stride_tensor
            )

            positive_pred_boxes = pred_boxes_grid[
                foreground_mask
            ]

            positive_target_boxes = target_boxes_grid[
                foreground_mask
            ]

            positive_weights = target_scores[
                foreground_mask
            ].sum(dim=-1)

            ciou = complete_iou(
                positive_pred_boxes,
                positive_target_boxes,
            )

            box_loss = (
                (1.0 - ciou)
                * positive_weights
            ).sum() / target_scores_sum

            target_distances = bbox_to_dist(
                anchor_points=anchor_points,
                boxes=target_boxes_grid,
                reg_max=self.reg_max,
            )

            positive_pred_dist = pred_dist[
                foreground_mask
            ].reshape(
                -1,
                4,
                self.reg_max,
            )

            positive_target_dist = target_distances[
                foreground_mask
            ]

            dfl_values = self.dfl(
                positive_pred_dist,
                positive_target_dist,
            )

            dfl_loss = (
                dfl_values
                * positive_weights
            ).sum() / target_scores_sum

        weighted_box_loss = (
            self.config.box_gain * box_loss
        )

        weighted_cls_loss = (
            self.config.cls_gain
            * classification_loss
        )

        weighted_dfl_loss = (
            self.config.dfl_gain * dfl_loss
        )

        total_loss = (
            weighted_box_loss
            + weighted_cls_loss
            + weighted_dfl_loss
        )

        if self.config.scale_loss_by_batch:
            total_loss = total_loss * batch_size

        loss_items = {
            "box_loss": weighted_box_loss.detach(),
            "cls_loss": weighted_cls_loss.detach(),
            "dfl_loss": weighted_dfl_loss.detach(),
            "total_loss": total_loss.detach(),
            "num_foreground": (
                foreground_mask.sum().detach()
            ),
        }

        return total_loss, loss_items
    
