"""Validate and aggregate the three corrected-hourly experiment runs."""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "hourly_v1"
SEEDS = [42, 7, 2026]
MODES = ["iid", "non_iid"]
STRATEGIES = ["centralized", "full", "random", "flowpulse"]
KEYS = ["partition_mode", "selection_strategy"]
METRICS = ["mae", "rmse", "communication_bytes"]

frames = []
histories = []

for seed in SEEDS:
    tables = BASE / f"seed{seed}" / "tables"
    frame = pd.read_csv(tables / "experiment_summary.csv")

    expected = {(mode, strategy) for mode in MODES for strategy in STRATEGIES}
    actual = set(zip(frame["partition_mode"], frame["selection_strategy"]))

    if len(frame) != 8 or actual != expected:
        raise ValueError(f"Seed {seed}: missing or duplicate experiment rows.")
    if not frame["seed"].eq(seed).all():
        raise ValueError(f"Seed {seed}: incorrect seed labels.")
    if not frame["rounds"].eq(12).all():
        raise ValueError(f"Seed {seed}: expected 12 rounds.")
    if not np.isfinite(frame[METRICS].to_numpy(dtype=float)).all():
        raise ValueError(f"Seed {seed}: non-finite metrics.")
    if (frame[METRICS] < 0).any().any():
        raise ValueError(f"Seed {seed}: negative metrics.")

    expected_bytes = frame["selection_strategy"].map({
        "centralized": 0,
        "full": 19200,
        "random": 9600,
        "flowpulse": 9600,
    })
    if not frame["communication_bytes"].eq(expected_bytes).all():
        raise ValueError(f"Seed {seed}: unexpected communication totals.")

    frames.append(frame)

    history = pd.read_csv(tables / "all_round_history.csv")
    if len(history) != 96 or history["experiment"].nunique() != 8:
        raise ValueError(f"Seed {seed}: unexpected history size.")
    for tag, group in history.groupby("experiment"):
        if sorted(group["round"].tolist()) != list(range(1, 13)):
            raise ValueError(f"Seed {seed}: invalid rounds for {tag}.")
    history["seed"] = seed
    histories.append(history)

combined = pd.concat(frames, ignore_index=True)
grouped = combined.groupby(KEYS, sort=True)

summary = grouped[METRICS].agg(["mean", "std"])
summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
summary = summary.reset_index()
summary["runs"] = grouped.size().to_numpy()

if not summary["runs"].eq(3).all():
    raise ValueError("Every comparison must contain exactly three runs.")

destination = BASE / "tables"
destination.mkdir(parents=True, exist_ok=True)

combined.to_csv(destination / "experiment_summary.csv", index=False)
summary.to_csv(destination / "multiseed_summary.csv", index=False)
pd.concat(histories, ignore_index=True).to_csv(
    destination / "all_round_history.csv", index=False
)

print(summary.round(3).to_string(index=False))
print("\nValidation passed: 24 experiments, 3 seeds, 12 rounds each.")
print("Standard deviations use sample std (ddof=1).")
print(f"Combined tables saved to: {destination}")
