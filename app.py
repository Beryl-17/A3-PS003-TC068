import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="FlowPulse-FL",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .main { background: #0e1117; }
      div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #16213a, #101828);
        border: 1px solid #24344d;
        border-radius: 14px;
        padding: 16px 18px;
      }
      div[data-testid="stMetricValue"] {
        color: #4fd1c5;
      }
      div[data-testid="stForm"] {
        background: rgba(22, 33, 58, 0.35);
        border: 1px solid #24344d;
        border-radius: 16px;
        padding: 20px;
      }

      h1, h2, h3 { color: #e8f1ff; }
      .flowpulse-card {
        background: linear-gradient(145deg, #16213a, #101828);
        border: 1px solid #24344d;
        border-radius: 16px;
        padding: 18px 20px;
        color: #dbe7ff;
      }
      .flowpulse-metric {
        font-size: 2rem; font-weight: 700; color: #4fd1c5;
      }
      .flowpulse-label {
        font-size: 0.85rem; color: #9fb3c8; text-transform: uppercase;
        letter-spacing: 0.08em;
      }
      div[data-testid="stDataFrame"] {
        border: 1px solid #24344d; border-radius: 12px; overflow: hidden;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

project_root = Path(__file__).resolve().parent
results_root = project_root / "outputs" / "hourly_v1"
tables = results_root / "tables"
plots = results_root
summary_path = tables / "multiseed_summary.csv"

with st.sidebar:
    st.markdown("## 🚦 FlowPulse-FL")
    st.markdown(
        "Federated traffic-flow forecasting across simulated data partitions. "
        "Clients train locally; the server aggregates only model updates with FedAvg."
    )
    st.divider()
    st.markdown("**Fixed benchmark setup**")
    st.markdown(
        "- 4 simulated clients (partitions of one dataset)\n"
        "- 12 communication rounds\n"
        "- 3 local epochs\n"
        "- Selection budget: 2 clients\n"
        "- Seeds: 42, 7, 2026"
    )
    st.caption(
        "These settings describe the saved three-seed benchmark. "
        "The Run Simulation tab lets you configure a separate in-memory run."
    )
    st.divider()
    st.markdown("**Outputs**")
    st.markdown("- MAE and RMSE\n- Communication bytes\n- Convergence plots")

st.markdown(
    """
    <div class="flowpulse-card">
      <h1 style="margin:0;">FlowPulse-FL</h1>
      <p style="margin:6px 0 0; color:#9fb3c8;">
        Communication-aware federated learning for urban traffic forecasting
        <br><span style="font-size:0.85rem;color:#4fd1c5;">Team Elevate Tech &nbsp; / &nbsp; AIORI-3 Level 2 &nbsp; / &nbsp; Historical-data simulator</span>
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

if not summary_path.exists():
    st.error("Generate corrected results first: `outputs/hourly_v1/tables/multiseed_summary.csv`.")
    st.stop()

df = pd.read_csv(summary_path)
full = df[df["selection_strategy"] == "full"]
selected = df[df["selection_strategy"].isin(["random", "flowpulse"])]


def render_fixed_benchmark_cards():
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div class="flowpulse-card">
              <div class="flowpulse-label">Benchmark clients</div>
              <div class="flowpulse-metric">4</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="flowpulse-card">
              <div class="flowpulse-label">Full-participation bytes</div>
              <div class="flowpulse-metric">19.2 KB</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="flowpulse-card">
              <div class="flowpulse-label">Selected-client bytes</div>
              <div class="flowpulse-metric">9.6 KB</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
            <div class="flowpulse-card">
              <div class="flowpulse-label">Communication saved</div>
              <div class="flowpulse-metric">50%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "Fixed three-seed benchmark values: 12 rounds, 3 local epochs, "
        "budget 2, and seeds 42, 7, and 2026. They are not affected by "
        "settings in the Run Simulation tab."
    )


def render_simulation_panel():
    import sys

    source_path = str(project_root / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)

    from flowpulse.simulation import run_simulation

    st.subheader("Run a traffic-forecasting simulation")
    st.caption(
        "Historical data; four simulated clients; linear autoregressive model. "
        "Each run trains a fresh model in memory and leaves benchmark files unchanged."
    )

    with st.form("simulation_controls"):
        left, middle, right = st.columns(3)
        with left:
            mode = st.selectbox("Data partition", ["non_iid", "iid"])
            strategy = st.selectbox(
                "Client selection", ["flowpulse", "random", "full"]
            )
        with middle:
            rounds = st.slider("Training rounds", 1, 30, 12)
            epochs = st.slider("Local epochs per round", 1, 5, 3)
        with right:
            budget = st.slider("Client budget", 1, 4, 2)
            seed = st.selectbox("Random seed", [42, 7, 2026])

        st.caption(
            "Fixed: 24-hour input window, learning rate 0.01, corrected hourly "
            "preprocessing. Full participation always uses all four clients, "
            "regardless of budget."
        )
        submitted = st.form_submit_button("Run simulation", type="primary")

    if submitted:
        try:
            with st.spinner("Training and evaluating the model..."):
                result = run_simulation(
                    data_path=project_root / "data/raw/traffic_volume.csv.gz",
                    partition_mode=mode,
                    strategy=strategy,
                    rounds=int(rounds),
                    local_epochs=int(epochs),
                    budget=int(budget),
                    seed=int(seed),
                )
            st.session_state["flowpulse_simulation"] = result
        except Exception as exc:
            st.error(f"Simulation failed: {exc}")

    result = st.session_state.get("flowpulse_simulation")
    if result is None:
        st.info("Choose settings and press Run simulation.")
        return

    config = result["config"]
    st.caption(
        f"Last successful run: {config['partition_mode']} | "
        f"{config['strategy']} | {config['rounds']} rounds | "
        f"{config['local_epochs']} local epochs | "
        f"budget {config['budget']} | seed {config['seed']}. "
        "Displayed results belong to this run, not unsubmitted control changes."
    )

    a, b, c, d = st.columns(4)
    a.metric("Test MAE", f"{result['mae']:.2f}")
    b.metric("Test RMSE", f"{result['rmse']:.2f}")
    c.metric("Model-payload bytes", f"{result['communication_bytes']:,}")
    d.metric("Payload saved vs full", f"{result['payload_saved_percent']:.1f}%")

    st.caption(
        f"Training windows: {result['train_windows']:,}; "
        f"test windows: {result['test_windows']:,}. "
        "Errors are in original traffic-volume units. Communication counts "
        "model uploads/downloads only, not metadata or network overhead."
    )

    st.markdown("#### Actual versus predicted traffic")
    count = st.slider("Number of test targets to display", 24, 240, 96)
    predictions = result["predictions"].head(count).copy()
    predictions.index.name = "Valid test-target index"
    st.line_chart(predictions, color=["#4fd1c5", "#f6ad55"])
    st.caption(
        "First valid test targets, in chronological order. The horizontal "
        "axis is a target index, not uninterrupted hourly time: gaps may occur."
    )

    st.markdown("#### Client participation and communication")
    st.dataframe(result["history"], hide_index=True, use_container_width=True)

    export_prefix = (
        f"flowpulse_{config['partition_mode']}_{config['strategy']}"
        f"_seed{config['seed']}_rounds{config['rounds']}"
        f"_epochs{config['local_epochs']}_budget{config['budget']}"
    )
    run_summary = {
        "config": dict(config),
        "mae": result["mae"],
        "rmse": result["rmse"],
        "communication_bytes": result["communication_bytes"],
        "full_participation_bytes": result["full_participation_bytes"],
        "payload_saved_percent": result["payload_saved_percent"],
        "train_windows": result["train_windows"],
        "test_windows": result["test_windows"],
        "notes": {
            "scope": "Historical-data simulator with four simulated clients.",
            "error_units": "Original traffic-volume units.",
            "communication": (
                "Model uploads/downloads only; excludes selection metadata, "
                "readiness/score exchange, and network overhead."
            ),
            "budget": (
                "Budget records the configured value. Full participation "
                "always selects all four clients regardless of this value."
            ),
        },
    }

    st.download_button(
        "Download this run's test predictions",
        data=result["predictions"].to_csv(index=False).encode("utf-8"),
        file_name=f"{export_prefix}_predictions.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download run history",
        data=result["history"].to_csv(index=False).encode("utf-8"),
        file_name=f"{export_prefix}_history.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download run summary",
        data=json.dumps(run_summary, indent=2, allow_nan=False).encode("utf-8"),
        file_name=f"{export_prefix}_summary.json",
        mime="application/json",
    )

    with st.expander("How FlowPulse client selection works"):
        st.caption(
            "FlowPulse currently combines a static target-difference score, "
            "simulated readiness, and timeout-based selection priority. "
            "The score is not a live or recent-traffic signal; in IID partitions, "
            "target order is shuffled. Timeout priority is budget-limited, "
            "not a guaranteed participation deadline."
        )


def render_benchmark_results():
    results_tab1, results_tab2, results_tab3 = st.tabs(
        ["📊 Forecast performance", "📡 Communication", "📈 Convergence"]
    )

    with results_tab1:
        st.markdown("### Three-seed forecasting performance")
        with st.expander("Preprocessing and evaluation scope"):
            st.info(
                "Corrected hourly preprocessing (hourly_v1): duplicate timestamps "
                "collapsed; missing hours left unfilled; only complete 24-hour input "
                "windows with an observed next-hour target retained. "
                "18,893 training windows and 9,978 test windows."
            )
            st.caption(
                "Historical-data simulator: not a live traffic feed or deployment "
                "at four real intersections."
            )
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(
            "Mean and sample standard deviation across seeds 42, 7, and 2026. "
            "MAE and RMSE use original traffic-volume units. Original raw-row "
            "results use a different split and are not directly comparable."
        )

    with results_tab2:
        st.markdown("### Average model-payload communication cost")
        with st.expander("What communication cost includes"):
            st.caption(
                "Simulated model uploads and downloads only. Excludes selection "
                "metadata, readiness/score exchange, and network overhead. "
                "Centralized zero means no simulated FL exchange, not zero "
                "data-transfer cost."
            )
        chart = df.pivot(
            index="partition_mode",
            columns="selection_strategy",
            values="communication_bytes_mean",
        )
        fig, ax = plt.subplots(figsize=(10, 4))
        chart[["full", "random", "flowpulse"]].plot.bar(
            ax=ax,
            stacked=False,
            color=["#63b3ed", "#ed64a6", "#f6ad55"],
            rot=0,
        )
        ax.set_xlabel("Training-data partition")
        ax.set_ylabel("Model-payload bytes per run")
        ax.set_title("Alternative strategies — not additive costs")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(title="Selection strategy")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "Centralized baseline omitted from this chart: its recorded zero "
            "represents no simulated FL model exchange."
        )
        st.caption(
            "Full participation communicates with all 4 clients. Random and FlowPulse "
            "select 2 clients per round, reducing communication by 50%."
        )

    with results_tab3:
        st.markdown("### Training convergence")
        with st.expander("What the convergence metric means"):
            st.caption(
                "Federated training MSE: mean selected-client post-local-training "
                "loss, not global-model loss on a fixed dataset. Centralized loss "
                "uses all training windows. These are not equivalent evaluation measures."
            )
        plot_files = sorted(plots.glob("seed*/plots/convergence_*.png"))
        if plot_files:
            choice = st.selectbox(
                "Select experiment",
                plot_files,
                format_func=lambda p: p.stem.replace("convergence_", "").replace("_", " "),
            )
            st.image(str(choice), use_container_width=True)
        else:
            st.warning("No convergence plots found.")


overview_tab, simulation_tab, benchmark_tab = st.tabs(
    ["🏠 Overview", "🧪 Run Simulation", "📚 Benchmark Results"]
)

with overview_tab:
    st.markdown("### Forecast traffic. Explore the communication trade-off.")
    st.caption(
        "A historical-data experiment workspace for comparing next-hour "
        "traffic-volume forecasting under different client-selection strategies."
    )

    st.markdown("#### From local observations to a shared forecast")
    st.markdown(
        """
        <div style="display:flex;flex-wrap:wrap;gap:12px;margin:12px 0 20px;">
          <div class="flowpulse-card" style="flex:1;min-width:180px;">
            <div class="flowpulse-label">01 / Simulated edge</div>
            <h3 style="margin:10px 0;">4 local clients</h3>
            <p style="margin:0;color:#b8c7dc;">
              Historical windows partitioned into IID or chronological blocks.
            </p>
          </div>
          <div class="flowpulse-card" style="flex:1;min-width:180px;">
            <div class="flowpulse-label">02 / Selection</div>
            <h3 style="margin:10px 0;">Choose participants</h3>
            <p style="margin:0;color:#b8c7dc;">
              Full participation, random selection, or FlowPulse selection.
            </p>
          </div>
          <div class="flowpulse-card" style="flex:1;min-width:180px;">
            <div class="flowpulse-label">03 / Aggregation</div>
            <h3 style="margin:10px 0;">FedAvg server</h3>
            <p style="margin:0;color:#b8c7dc;">
              Combine local model updates using sample-count weighting.
            </p>
          </div>
          <div class="flowpulse-card" style="flex:1;min-width:180px;">
            <div class="flowpulse-label">04 / Evaluation</div>
            <h3 style="margin:10px 0;">Next-hour forecast</h3>
            <p style="margin:0;color:#b8c7dc;">
              Compare held-out forecast errors and model-payload bytes.
            </p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Fixed benchmark / communication at a glance")
    render_fixed_benchmark_cards()
    st.caption(
        "Both random and FlowPulse selection achieve this 50% payload reduction "
        "by selecting 2 of 4 clients per round. It is not a unique FlowPulse gain. "
        "KB uses decimal units: 1 KB = 1,000 bytes."
    )

    st.write("")
    forecast_col, select_col, compare_col = st.columns(3)
    with forecast_col:
        st.markdown("#### Forecast")
        st.write(
            "Use 24 observed hourly traffic-volume values to predict the next "
            "hour with a linear autoregressive model."
        )
    with select_col:
        st.markdown("#### Select")
        st.write(
            "Configure participation budget, training rounds, local epochs, "
            "partition mode, and seed in Run Simulation."
        )
    with compare_col:
        st.markdown("#### Compare")
        st.write(
            "Inspect actual versus predicted values, client participation, "
            "payload costs, and saved three-seed benchmark results."
        )

    st.info(
        "Start in Run Simulation to train a fresh model. Open Benchmark Results "
        "for the saved experiments, including the centralized reference."
    )

    with st.expander("Scope, interpretation, and limitations"):
        st.write(
            "This is a historical-data simulator, not a live traffic feed or "
            "physical edge deployment. Four clients are partitions of one dataset. "
            "The simulated server aggregates model updates; the application "
            "itself runs locally in one process."
        )
        st.write(
            "Payload accounting includes model uploads and downloads, not "
            "selection metadata, readiness/score exchange, or network overhead. "
            "FlowPulse is not universally the most accurate strategy; observed "
            "differences are small relative to across-seed variability."
        )
        st.write(
            "FlowPulse uses a static target-difference score and simulated "
            "readiness. The score is not a recent-traffic signal, and timeout "
            "priority does not guarantee a participation deadline."
        )

with simulation_tab:
    render_simulation_panel()

with benchmark_tab:
    render_benchmark_results()

st.divider()
with st.expander("Reproduce the fixed benchmark — seed 42"):
    st.code(
        "PYTHONPATH=src python -m flowpulse.run_experiments --seed 42 "
        "--rounds 12 --local-epochs 3 --clients 4 --budget 2 "
        "--modes iid non_iid --strategies full random flowpulse --output-dir outputs/hourly_v1/seed42",
        language="bash",
    )
