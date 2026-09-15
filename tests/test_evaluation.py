from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from evaluation import (
    DetectionMetricAccumulator,
    initialize_evaluation_files,
    match_predictions,
)


class DetectionEvaluationTest(unittest.TestCase):
    def test_perfect_prediction_has_perfect_metrics(self) -> None:
        accumulator = DetectionMetricAccumulator(
            class_names=("object",),
            f1_confidence_threshold=0.25,
        )
        accumulator.update(
            detections=torch.tensor([[0, 0, 10, 10, 0.9, 0.0]]),
            target_boxes=torch.tensor([[0, 0, 10, 10]]),
            target_classes=torch.tensor([0]),
        )

        metrics = accumulator.compute()

        self.assertAlmostEqual(metrics.precision, 1.0)
        self.assertAlmostEqual(metrics.recall, 1.0)
        self.assertAlmostEqual(metrics.f1, 1.0)
        self.assertAlmostEqual(metrics.map50, 1.0)
        self.assertAlmostEqual(metrics.map50_95, 1.0)

    def test_duplicate_prediction_is_false_positive(self) -> None:
        accumulator = DetectionMetricAccumulator(
            class_names=("object",),
            f1_confidence_threshold=0.25,
        )
        accumulator.update(
            detections=torch.tensor(
                [
                    [0, 0, 10, 10, 0.9, 0.0],
                    [0, 0, 10, 10, 0.8, 0.0],
                ]
            ),
            target_boxes=torch.tensor([[0, 0, 10, 10]]),
            target_classes=torch.tensor([0]),
        )

        metrics = accumulator.compute()

        self.assertAlmostEqual(metrics.precision, 0.5)
        self.assertAlmostEqual(metrics.recall, 1.0)
        self.assertAlmostEqual(metrics.f1, 2.0 / 3.0)

    def test_wrong_class_does_not_match(self) -> None:
        detections = torch.tensor([[0, 0, 10, 10, 0.9, 1.0]])
        correct = match_predictions(
            detections=detections,
            target_boxes=torch.tensor([[0, 0, 10, 10]]),
            target_classes=torch.tensor([0]),
            iou_thresholds=torch.tensor([0.5, 0.75]),
        )

        self.assertFalse(correct.any())

    def test_metric_files_are_initialized_for_a_new_run(self) -> None:
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            initialize_evaluation_files(output_dir)

            header = (output_dir / "metrics.csv").read_text(
                encoding="utf-8"
            )
            per_class = (output_dir / "metrics_per_class.jsonl").read_text(
                encoding="utf-8"
            )

        self.assertTrue(header.startswith("round,train_loss,val_loss"))
        self.assertEqual(per_class, "")


if __name__ == "__main__":
    unittest.main()
