from pathlib import Path
import numpy as np
import pandas as pd


def load_traffic_series(data_path: str | Path) -> pd.Series:
    """Load a deduplicated, regular hourly traffic-volume series."""
    df = pd.read_csv(data_path, compression="gzip")
    df["date_time"] = pd.to_datetime(df["date_time"])

    series = (
        df.groupby("date_time")["traffic_volume"]
        .first()
        .sort_index()
        .asfreq("h")
        .astype(float)
    )
    return series


def chronological_split(series: pd.Series, train_ratio: float = 0.8):
    """Split the regular hourly timeline by target time."""
    split_at = int(len(series) * train_ratio)
    train = series.iloc[:split_at].copy()
    test = series.iloc[split_at:].copy()
    return train, test


def make_windows(values: np.ndarray, window_size: int = 24):
    """Use the previous window_size observations to predict the next one."""
    x, y = [], []
    for end in range(window_size, len(values)):
        x.append(values[end - window_size:end])
        y.append(values[end])
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def make_valid_windows(
    values: np.ndarray,
    observed_mask: np.ndarray,
    window_size: int = 24,
):
    """Create only complete windows: 24 observed inputs plus an observed target."""
    x, y = [], []
    for end in range(window_size, len(values)):
        if observed_mask[end - window_size:end + 1].all():
            x.append(values[end - window_size:end])
            y.append(values[end])
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def normalize_from_train(series: pd.Series, split_at: int):
    """Scale using observed training-period values only."""
    train_values = series.iloc[:split_at].dropna()

    mean = float(train_values.mean())
    std = float(train_values.std())
    std = std if std > 0 else 1.0

    scaled = (series - mean) / std
    return scaled, mean, std


def make_client_partitions(x, y, num_clients: int, mode: str, seed: int):
    """
    IID: shuffle windows, then distribute equal shares.
    non_iid: contiguous time blocks.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(x))

    if mode == "iid":
        rng.shuffle(indices)
    elif mode != "non_iid":
        raise ValueError("mode must be 'iid' or 'non_iid'")

    chunks = np.array_split(indices, num_clients)
    return [(x[idx], y[idx]) for idx in chunks]


def prepare_data(
    data_path: str | Path,
    num_clients: int = 4,
    partition_mode: str = "non_iid",
    seed: int = 42,
    window_size: int = 24,
):
    """Prepare valid hourly windows for federated traffic forecasting."""
    series = load_traffic_series(data_path)
    split_at = int(len(series) * 0.8)

    scaled_series, mean, std = normalize_from_train(series, split_at)
    observed_mask = series.notna().to_numpy()

    x_all, y_all = make_windows(
        scaled_series.to_numpy(), window_size=window_size
    )

    target_indices = np.arange(window_size, len(series))
    # Require every input hour AND the target hour to be observed.
    valid_targets = np.array(
        [
            observed_mask[end - window_size:end + 1].all()
            for end in target_indices
        ],
        dtype=bool,
    )

    train_target_mask = (target_indices < split_at) & valid_targets
    test_target_mask = (target_indices >= split_at) & valid_targets

    x_train = x_all[train_target_mask]
    y_train = y_all[train_target_mask]
    x_test = x_all[test_target_mask]
    y_test = y_all[test_target_mask]

    clients = make_client_partitions(
        x_train, y_train, num_clients, partition_mode, seed
    )

    return {
        "clients": clients,
        "x_test": x_test,
        "y_test": y_test,
        "mean": mean,
        "std": std,
        "window_size": window_size,
        "train_points": len(x_train),
        "test_points": len(x_test),
    }
