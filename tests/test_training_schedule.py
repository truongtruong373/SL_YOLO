from __future__ import annotations

import unittest

from train import should_evaluate_round


class EvaluationScheduleTest(unittest.TestCase):
    def test_evaluates_at_configured_interval(self) -> None:
        evaluated_rounds = [
            round_index
            for round_index in range(1, 13)
            if should_evaluate_round(round_index, total_rounds=12, interval=5)
        ]

        self.assertEqual(evaluated_rounds, [5, 10, 12])

    def test_always_evaluates_final_round(self) -> None:
        self.assertTrue(
            should_evaluate_round(2, total_rounds=2, interval=5)
        )

    def test_interval_one_evaluates_every_round(self) -> None:
        self.assertTrue(
            all(
                should_evaluate_round(index, total_rounds=3, interval=1)
                for index in range(1, 4)
            )
        )


if __name__ == "__main__":
    unittest.main()
