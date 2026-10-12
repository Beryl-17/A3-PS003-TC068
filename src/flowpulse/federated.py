import numpy as np

from .model import mse_loss, parameter_bytes, train_local


def fedavg(local_models, sample_counts):
    """Weighted FedAvg aggregation of client parameters."""
    total = float(sum(sample_counts))
    averaged = {}
    for key in local_models[0]:
        averaged[key] = sum(
            model[key] * (count / total)
            for model, count in zip(local_models, sample_counts)
        )
    return averaged


def traffic_change_score(client_y):
    """Higher score means the client's recent target values vary more."""
    if len(client_y) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(client_y))))


def choose_clients(clients, strategy, round_idx, budget, rng, last_selected,
                   fairness_timeout=3):
    """
    full: every client participates.
    random: random limited selection.
    flowpulse: change-aware selection with simulated readiness and fairness timeout.
    """
    n_clients = len(clients)
    all_ids = list(range(n_clients))

    if strategy == "full":
        selected = all_ids
    elif strategy == "random":

        selected = rng.choice(all_ids, size=min(budget, n_clients), replace=False).tolist()
    elif strategy == "flowpulse":
        forced = [
            client_id for client_id in all_ids
            if round_idx - last_selected[client_id] >= fairness_timeout
        ]
        remaining = [client_id for client_id in all_ids if client_id not in forced]
        readiness = rng.uniform(0.6, 1.0, size=n_clients)
        ranked = sorted(
            remaining,
            key=lambda client_id: (
                0.75 * traffic_change_score(clients[client_id][1])
                + 0.25 * readiness[client_id]
            ),
            reverse=True,
        )
        selected = (forced + ranked)[:min(budget, n_clients)]
    else:
        raise ValueError("strategy must be 'full', 'random', or 'flowpulse'")

    for client_id in selected:
        last_selected[client_id] = round_idx
    return selected


def run_federated_training(clients, initial_params, rounds=12, local_epochs=3,
                           learning_rate=0.01, strategy="full", budget=2,
                           seed=42):
    """Train clients locally, aggregate using FedAvg, and log communication/loss."""
    rng = np.random.default_rng(seed)
    global_params = {key: value.copy() for key, value in initial_params.items()}
    history = []
    last_selected = {client_id: -3 for client_id in range(len(clients))}
    model_size = parameter_bytes(global_params)

    for round_idx in range(rounds):
        selected = choose_clients(
            clients, strategy, round_idx, budget, rng, last_selected
        )
        local_models, sample_counts, local_losses = [], [], []

        for client_id in selected:
            x_client, y_client = clients[client_id]
            trained = train_local(
                global_params, x_client, y_client,
                epochs=local_epochs, learning_rate=learning_rate
            )
            local_models.append(trained)
            sample_counts.append(len(x_client))
            local_losses.append(mse_loss(trained, x_client, y_client))

        global_params = fedavg(local_models, sample_counts)
        round_bytes = model_size * 2 * len(selected)
        history.append(
            {
                "round": round_idx + 1,
                "selected_clients": ",".join(str(i + 1) for i in selected),
                "num_selected": len(selected),
                "train_mse": float(np.mean(local_losses)),
                "communication_bytes": int(round_bytes),
                "cumulative_bytes": int(
                    sum(row["communication_bytes"] for row in history) + round_bytes
                ),
            }
        )

    return global_params, history
