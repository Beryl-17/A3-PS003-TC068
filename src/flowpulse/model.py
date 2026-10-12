import numpy as np


def init_weights(window_size: int, seed: int):
    """Initialize a tiny linear autoregressive forecaster."""
    rng = np.random.default_rng(seed)
    return {
        "w": rng.normal(0, 0.05, size=window_size),
        "b": np.array(0.0),
    }


def predict(params, x):
    return x @ params["w"] + params["b"]


def mse_loss(params, x, y):
    error = predict(params, x) - y
    return float(np.mean(error ** 2))


def train_local(params, x, y, epochs: int = 3, learning_rate: float = 0.01):
    """Full-batch gradient descent performed independently by one client."""
    updated = {"w": params["w"].copy(), "b": params["b"].copy()}
    n = max(len(x), 1)

    for _ in range(epochs):
        error = predict(updated, x) - y
        grad_w = (2.0 / n) * (x.T @ error)
        grad_b = np.array(2.0 * np.mean(error))
        updated["w"] -= learning_rate * grad_w
        updated["b"] -= learning_rate * grad_b

    return updated


def parameter_bytes(params):
    """Bytes sent for one model upload or one global-model download."""
    return int(params["w"].nbytes + params["b"].nbytes)
