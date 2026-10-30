import unittest
import numpy as np
import pandas as pd

from flowpulse.data import normalize_from_train, make_valid_windows
from flowpulse.federated import fedavg, run_federated_training
from flowpulse.model import init_weights, parameter_bytes
from flowpulse.simulation import run_simulation


class CoreTests(unittest.TestCase):
    def test_training_only_normalization(self):
        series = pd.Series([1.0, 2.0, 3.0, 1000.0])
        scaled, mean, std = normalize_from_train(series, 3)
        self.assertAlmostEqual(mean, 2.0)
        self.assertAlmostEqual(std, 1.0)
        self.assertAlmostEqual(scaled.iloc[3], 998.0)

    def test_missing_inputs_and_targets_excluded(self):
        values = np.array([1., 2., 3., np.nan, 5., 6., 7., 8.])
        x, y = make_valid_windows(
            values, np.isfinite(values), window_size=2
        )
        np.testing.assert_allclose(x, [[1., 2.], [5., 6.], [6., 7.]])
        np.testing.assert_allclose(y, [3., 7., 8.])

    def test_sample_weighted_fedavg(self):
        models = [
            {"w": np.array([1., 3.]), "b": np.array(2.)},
            {"w": np.array([5., 7.]), "b": np.array(6.)},
        ]
        averaged = fedavg(models, [1, 3])
        np.testing.assert_allclose(averaged["w"], [4., 6.])
        np.testing.assert_allclose(averaged["b"], 5.)

    def test_model_payload_size(self):
        self.assertEqual(parameter_bytes(init_weights(24, 42)), 200)

    def test_selection_and_round_accounting(self):
        rng = np.random.default_rng(42)
        clients = [
            (rng.normal(size=(8, 24)), rng.normal(size=8))
            for _ in range(4)
        ]
        for strategy in ("full", "random", "flowpulse"):
            for budget in (1, 2, 4):
                with self.subTest(strategy=strategy, budget=budget):
                    initial = init_weights(24, 42)
                    original = {k: v.copy() for k, v in initial.items()}
                    params, history = run_federated_training(
                        clients, initial, rounds=4, local_epochs=1,
                        strategy=strategy, budget=budget, seed=42
                    )
                    count = 4 if strategy == "full" else budget
                    expected_bytes = 200 * 2 * count
                    self.assertEqual(len(history), 4)
                    for index, row in enumerate(history, start=1):
                        ids = [int(i) for i in row["selected_clients"].split(",")]
                        self.assertEqual(row["round"], index)
                        self.assertEqual(len(ids), count)
                        self.assertEqual(len(set(ids)), count)
                        self.assertTrue(all(1 <= i <= 4 for i in ids))
                        self.assertEqual(row["num_selected"], count)
                        self.assertEqual(row["communication_bytes"], expected_bytes)
                        self.assertEqual(
                            row["cumulative_bytes"], expected_bytes * index
                        )
                        self.assertTrue(np.isfinite(row["train_mse"]))
                    for key in params:
                        self.assertTrue(np.isfinite(params[key]).all())
                        np.testing.assert_array_equal(initial[key], original[key])

    def test_simulation_rejects_invalid_inputs_before_loading_data(self):
        invalid_cases = [
            {"rounds": 0}, {"rounds": 31}, {"rounds": True},
            {"local_epochs": 0}, {"local_epochs": 6},
            {"budget": 0}, {"budget": 5},
            {"seed": -1}, {"seed": 2**32},
            {"partition_mode": "invalid"}, {"strategy": "centralized"},
        ]
        for options in invalid_cases:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    run_simulation(
                        data_path="__unused_missing_dataset__.csv.gz",
                        **options
                    )


if __name__ == "__main__":
    unittest.main()
