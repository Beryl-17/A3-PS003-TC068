"""In-memory simulation for the interactive dashboard.

Does not write experiment outputs or modify benchmark results.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from .data import prepare_data
from .federated import run_federated_training
from .model import init_weights, parameter_bytes, predict


def run_simulation(
    data_path: str | Path,
    partition_mode="non_iid",
    strategy="flowpulse",
    rounds=12,
    local_epochs=3,
    budget=2,
    seed=42,
):
    """Run a four-client, 24-hour-window simulation at learning rate 0.01."""
    for name, value, low, high in [
        ("rounds", rounds, 1, 30),
        ("local_epochs", local_epochs, 1, 5),
        ("budget", budget, 1, 4),
        ("seed", seed, 0, 2**32 - 1),
    ]:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer.")
        if not low <= value <= high:
            raise ValueError(f"{name} must be between {low} and {high}.")

    if partition_mode not in ("iid", "non_iid"):
        raise ValueError("Invalid partition mode.")
    if strategy not in ("full", "random", "flowpulse"):
        raise ValueError("Invalid selection strategy.")

    data = prepare_data(
        data_path=data_path,
        num_clients=4,
        partition_mode=partition_mode,
        seed=seed,
        window_size=24,
    )
    initial = init_weights(24, seed)
    params, history = run_federated_training(
        clients=data["clients"],
        initial_params=initial,
        rounds=rounds,
        local_epochs=local_epochs,
        learning_rate=0.01,
        strategy=strategy,
        budget=budget,
        seed=seed,
    )

    actual = data["y_test"] * data["std"] + data["mean"]
    forecast = predict(params, data["x_test"]) * data["std"] + data["mean"]

    if not np.isfinite(actual).all() or not np.isfinite(forecast).all():
        raise ValueError("Simulation produced non-finite values.")

    history_df = pd.DataFrame(history)
    total_bytes = int(history_df["communication_bytes"].sum())
    full_bytes = parameter_bytes(initial) * 2 * 4 * rounds

    return {
        "config": {
            "partition_mode": partition_mode,
            "strategy": strategy,
            "rounds": rounds,
            "local_epochs": local_epochs,
            "budget": budget,
            "seed": seed,
            "clients": 4,
            "window_size": 24,
            "learning_rate": 0.01,
            "preprocessing": "hourly_v1",
        },
        "mae": float(np.mean(np.abs(actual - forecast))),
        "rmse": float(np.sqrt(np.mean((actual - forecast) ** 2))),
        "communication_bytes": total_bytes,
        "full_participation_bytes": full_bytes,
        "payload_saved_percent": 100.0 * (1.0 - total_bytes / full_bytes),
        "history": history_df,
        "predictions": pd.DataFrame({
            "actual": actual,
            "predicted": forecast,
        }),
        "train_windows": sum(len(y) for _, y in data["clients"]),
        "test_windows": len(actual),
    }
