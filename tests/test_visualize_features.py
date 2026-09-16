from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn as nn

from visualize_features import (
    capture_activations,
    normalize_map,
    select_informative_channels,
)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


class FeatureVisualizationTest(unittest.TestCase):
    def test_normalize_map_handles_constant_values(self) -> None:
        normalized = normalize_map(np.ones((4, 4), dtype=np.float32))
        np.testing.assert_array_equal(normalized, np.zeros((4, 4)))

    def test_selects_requested_number_of_channels(self) -> None:
        activation = torch.randn(8, 5, 5)
        indices, channels = select_informative_channels(activation, 3)

        self.assertEqual(indices.shape, (3,))
        self.assertEqual(channels.shape, (3, 5, 5))

    def test_capture_activations_uses_layer_indices(self) -> None:
        model = TinyModel().eval()
        activations = capture_activations(
            model=model,
            image_tensor=torch.randn(1, 3, 8, 8),
            layer_indices=(0, 1),
        )

        self.assertEqual(set(activations), {0, 1})
        self.assertEqual(activations[0].shape, (4, 8, 8))
        self.assertEqual(activations[1].shape, (4, 8, 8))

    def test_rejects_duplicate_layers(self) -> None:
        with self.assertRaisesRegex(ValueError, "trùng nhau"):
            capture_activations(
                model=TinyModel(),
                image_tensor=torch.randn(1, 3, 8, 8),
                layer_indices=(0, 0),
            )


if __name__ == "__main__":
    unittest.main()
