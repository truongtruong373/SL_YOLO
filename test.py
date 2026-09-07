import torch
from loss import YOLODetectionLoss, YOLOLossConfig

device = "cuda" if torch.cuda.is_available() else "cpu"

loss_config = YOLOLossConfig(
    num_classes=3,
    reg_max=16,
    strides=(8, 16, 32),
    box_gain=7.5,
    cls_gain=0.5,
    dfl_gain=1.5,
    assigner_topk=10,
)

criterion = YOLODetectionLoss(
    loss_config
).to(device)


batch_size = 2
num_classes = 3
reg_max = 16

predictions = [
    (
        torch.randn(
            batch_size,
            4 * reg_max,
            80,
            80,
            device=device,
            requires_grad=True,
        ),
        torch.randn(
            batch_size,
            num_classes,
            80,
            80,
            device=device,
            requires_grad=True,
        ),
    ),
    (
        torch.randn(
            batch_size,
            4 * reg_max,
            40,
            40,
            device=device,
            requires_grad=True,
        ),
        torch.randn(
            batch_size,
            num_classes,
            40,
            40,
            device=device,
            requires_grad=True,
        ),
    ),
    (
        torch.randn(
            batch_size,
            4 * reg_max,
            20,
            20,
            device=device,
            requires_grad=True,
        ),
        torch.randn(
            batch_size,
            num_classes,
            20,
            20,
            device=device,
            requires_grad=True,
        ),
    ),
]

targets = {
    "batch_idx": torch.tensor(
        [0, 0, 1],
        device=device,
    ),
    "cls": torch.tensor(
        [0, 2, 1],
        device=device,
    ),
    "bboxes": torch.tensor(
        [
            [0.50, 0.50, 0.20, 0.30],
            [0.20, 0.25, 0.10, 0.15],
            [0.70, 0.60, 0.25, 0.20],
        ],
        device=device,
    ),
}

total_loss, loss_items = criterion(
    predictions,
    targets,
)
print(total_loss)

print(loss_items)

