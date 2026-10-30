"""Run centralized and federated traffic-forecasting experiments."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import prepare_data
from .federated import run_federated_training
from .model import init_weights, parameter_bytes, predict, train_local


def regression_metrics(y_true_scaled, y_pred_scaled, mean, std):
    """Compute MAE and RMSE after converting predictions back to vehicle counts."""
    y_true = np.asarray(y_true_scaled).reshape(-1) * std + mean
    y_pred = np.asarray(y_pred_scaled).reshape(-1) * std + mean
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"mae": mae, "rmse": rmse}


def train_centralized(data, rounds, local_epochs, learning_rate, seed):
    """Train one model on all training windows as the centralized baseline."""
    x_train = np.vstack([client_x for client_x, _ in data["clients"]])
    y_train = np.concatenate([client_y for _, client_y in data["clients"]])

    params = init_weights(data["window_size"], seed)
    history = []

    for round_idx in range(rounds):
        params = train_local(
            params, x_train, y_train,
            epochs=local_epochs, learning_rate=learning_rate
        )
        history.append({
            "round": round_idx + 1,
            "train_mse": float(np.mean((predict(params, x_train) - y_train) ** 2)),
            "communication_bytes": 0,
            "cumulative_bytes": 0,
        })

    return params, history


def evaluate(params, data):
    """Evaluate a model on the chronological test set using real units."""
    y_pred = predict(params, data["x_test"])
    return regression_metrics(
        data["y_test"], y_pred, data["mean"], data["std"]
    )


def save_run_outputs(tag, history, metrics, tables_dir, plots_dir):
    """Save one experiment's round history, final metrics, and convergence plot."""
    history_df = pd.DataFrame(history)
    history_df["experiment"] = tag
    history_df.to_csv(tables_dir / f"rounds_{tag}.csv", index=False)

    with open(tables_dir / f"metrics_{tag}.json", "w") as handle:
        json.dump({"experiment": tag, **metrics}, handle, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history_df["round"], history_df["train_mse"], marker="o")
    axes[0].set_title(f"{tag}: training MSE")
    axes[0].set_xlabel("Communication round")
    axes[0].set_ylabel("Training MSE")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        history_df["round"], history_df["cumulative_bytes"], marker="o", color="tab:red"
    )
    axes[1].set_title(f"{tag}: cumulative communication")
    axes[1].set_xlabel("Communication round")
    axes[1].set_ylabel("Bytes")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plots_dir / f"convergence_{tag}.png", dpi=160)
    plt.close(fig)

    return history_df


def main():
    parser = argparse.ArgumentParser(description="FlowPulse-FL experiment runner")
    parser.add_argument("--data", default="data/raw/traffic_volume.csv.gz")
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--modes", nargs="+", default=["iid", "non_iid"],
        choices=["iid", "non_iid"]
    )
    parser.add_argument(
        "--strategies", nargs="+",
        default=["full", "random", "flowpulse"],
        choices=["full", "random", "flowpulse"]
    )
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument(
        "--output-dir", default="outputs",
        help="Directory for this invocation's tables and plots."
    )
    args = parser.parse_args()

    raw_path = Path(args.data)
    if not raw_path.exists():
        raise FileNotFoundError(f"Dataset not found: {raw_path.resolve()}")

    output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_history = []
    summary_rows = []

    for mode in args.modes:
        data = prepare_data(
            data_path=raw_path,
            num_clients=args.clients,
            partition_mode=mode,
            seed=args.seed,
            window_size=args.window_size,
        )

        centralized_params, centralized_history = train_centralized(
            data, args.rounds, args.local_epochs, args.learning_rate, args.seed
        )
        centralized_metrics = evaluate(centralized_params, data)
        centralized_metrics["communication_bytes"] = 0
        centralized_metrics["rounds"] = args.rounds
        centralized_metrics["partition_mode"] = mode
        centralized_metrics["selection_strategy"] = "centralized"
        centralized_metrics["seed"] = args.seed

        tag = f"centralized_{mode}_seed{args.seed}"
        all_history.append(
            save_run_outputs(tag, centralized_history, centralized_metrics,
                             tables_dir, plots_dir)
        )
        summary_rows.append(centralized_metrics)

        initial_params = init_weights(data["window_size"], args.seed)

        for strategy in args.strategies:
            final_params, history = run_federated_training(
                clients=data["clients"],
                initial_params=initial_params,
                rounds=args.rounds,
                local_epochs=args.local_epochs,
                learning_rate=args.learning_rate,
                strategy=strategy,
                budget=args.budget,
                seed=args.seed,
            )

            metrics = evaluate(final_params, data)
            metrics["communication_bytes"] = int(history[-1]["cumulative_bytes"])
            metrics["rounds"] = args.rounds
            metrics["partition_mode"] = mode
            metrics["selection_strategy"] = strategy
            metrics["seed"] = args.seed

            tag = f"fed_{mode}_{strategy}_seed{args.seed}"
            all_history.append(
                save_run_outputs(tag, history, metrics, tables_dir, plots_dir)
            )
            summary_rows.append(metrics)

            print(
                f"{tag}: MAE={metrics['mae']:.2f}, "
                f"RMSE={metrics['rmse']:.2f}, "
                f"bytes={metrics['communication_bytes']}"
            )

    summary = pd.DataFrame(summary_rows)
    column_order = [
        "partition_mode", "selection_strategy", "mae", "rmse",
        "communication_bytes", "rounds", "seed",
    ]
    summary = summary[[col for col in column_order if col in summary.columns]]
    summary.to_csv(tables_dir / "experiment_summary.csv", index=False)

    pd.concat(all_history, ignore_index=True).to_csv(
        tables_dir / "all_round_history.csv", index=False
    )

    print("\nAll experiments completed.")
    print(f"Summary: {tables_dir / 'experiment_summary.csv'}")
    print(f"Round history: {tables_dir / 'all_round_history.csv'}")
    print(f"Plots: {plots_dir}")


if __name__ == "__main__":
    main()
