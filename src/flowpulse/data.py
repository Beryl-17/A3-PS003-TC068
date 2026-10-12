from pathlib import Path
import numpy as np
import pandas as pd


def load_traffic_series(data_path: str | Path) -> pd.Series:
    """Load, sort, and return the public traffic-volume time series."""
    df = pd.read_csv(data_path, compression="gzip")
    df["date_time"] = pd.to_datetime(df["date_time"])
    df = df.sort_values("date_time").dropna(subset=["traffic_volume"])
    series = df.set_index("date_time")["traffic_volume"].astype(float)
    return series


def chronological_split(series: pd.Series, train_ratio: float = 0.8):
    """Split by time: earlier samples train, later samples test."""
    split_at = int(len(series) * train_ratio)
    return series.iloc[:split_at].copy(), series.iloc[split_at:].copy()


def make_windows(values: np.ndarray, window_size: int = 24):
    """Use the previous window_size hours to predict the next hour."""
    x, y = [], []
    for end in range(window_size, len(values)):
        x.append(values[end - window_size:end])
        y.append(values[end])
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def normalize_from_train(train: pd.Series, test: pd.Series):
    """Scale using only training statistics to prevent test-data leakage."""
    mean = float(train.mean())
    std = float(train.std())
    std = std if std > 0 else 1.0
    return (train - mean) / std, (test - mean) / std, mean, std


def make_client_partitions(x, y, num_clients: int, mode: str, seed: int):
    """
    IID: shuffle windows, then distribute equal shares.
    non_iid: contiguous time blocks; each client sees a different traffic regime.
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    indices = np.arange(n)

    if mode == "iid":
        rng.shuffle(indices)
    elif mode != "non_iid":
        raise ValueError("mode must be 'iid' or 'non_iid'")

    chunks = np.array_split(indices, num_clients)
    return [(x[idx], y[idx]) for idx in chunks]


def prepare_data(data_path: str | Path, num_clients: int = 4,
                 partition_mode: str = "non_iid", seed: int = 42,
                 window_size: int = 24):
    """End-to-end preparation for the federated traffic forecasting simulator."""
    series = load_traffic_series(data_path)
    train, test = chronological_split(series)
    train_scaled, test_scaled, mean, std = normalize_from_train(train, test)

    x_train, y_train = make_windows(train_scaled.to_numpy(), window_size)
    x_test, y_test = make_windows(
        np.concatenate([train_scaled.to_numpy()[-window_size:], test_scaled.to_numpy()]),
        window_size,
    )

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
        "train_points": len(train),
        "test_points": len(test),
    }
