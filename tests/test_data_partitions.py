from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path

from scripts.data.create_data_partitions import Sample, partition_iid_samples


def make_sample(index: int, object_count: int, class_id: int) -> Sample:
    class_counts = [0] * 10
    class_counts[class_id] = object_count
    name = f"source_{index:04d}.jpg"
    return Sample(
        source_id=f"source_{index % 3}",
        image_path=Path("images") / name,
        annotation_path=Path("annotations") / f"{Path(name).stem}.txt",
        label_path=Path("labels") / f"{Path(name).stem}.txt",
        object_count=object_count,
        class_counts=tuple(class_counts),
    )


class IidPartitionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.samples = [
            make_sample(index, index % 7, index % 10) for index in range(83)
        ]

    def test_each_sample_is_used_once_and_sizes_are_balanced(self) -> None:
        partitions = partition_iid_samples(self.samples, 8, seed=42)
        assigned_paths = [
            sample.image_path for partition in partitions for sample in partition
        ]

        self.assertCountEqual(
            assigned_paths, [sample.image_path for sample in self.samples]
        )
        self.assertEqual(len(assigned_paths), len(set(assigned_paths)))
        sizes = [len(partition) for partition in partitions]
        self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_every_object_count_stratum_is_balanced(self) -> None:
        partitions = partition_iid_samples(self.samples, 8, seed=42)
        histograms = [
            Counter(sample.object_count for sample in partition)
            for partition in partitions
        ]

        for object_count in {sample.object_count for sample in self.samples}:
            counts = [histogram[object_count] for histogram in histograms]
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_seed_makes_partition_reproducible(self) -> None:
        first = partition_iid_samples(self.samples, 8, seed=123)
        second = partition_iid_samples(self.samples, 8, seed=123)

        self.assertEqual(
            [[sample.image_path for sample in part] for part in first],
            [[sample.image_path for sample in part] for part in second],
        )

    def test_class_objects_are_balanced_when_exact_balance_is_possible(self) -> None:
        samples = [
            make_sample(index, object_count=1, class_id=index % 10)
            for index in range(80)
        ]
        partitions = partition_iid_samples(samples, 8, seed=42)

        for class_id in range(10):
            counts = [
                sum(sample.class_counts[class_id] for sample in partition)
                for partition in partitions
            ]
            self.assertEqual(max(counts), min(counts))

    def test_rejects_more_devices_than_samples(self) -> None:
        with self.assertRaises(ValueError):
            partition_iid_samples(self.samples[:3], 8, seed=42)


if __name__ == "__main__":
    unittest.main()
