from pathlib import Path
import pandas as pd
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

tables = Path("outputs/tables")
plots = Path("outputs/plots")
summary_path = tables / "multiseed_summary.csv"

with st.sidebar:
    st.markdown("## 🚦 FlowPulse-FL")
    st.markdown(
        "Federated traffic-flow forecasting across simulated intersections. "
        "Clients train locally; the server aggregates only model updates with FedAvg."
    )
    st.divider()
    st.markdown("**Experiment setup**")
    st.markdown("- 4 simulated intersections\n- 12 communication rounds\n- 3 local epochs\n- Selection budget: 2 clients\n- Seeds: 42, 7, 2026")
    st.divider()
    st.markdown("**Outputs**")
    st.markdown("- MAE and RMSE\n- Communication bytes\n- Convergence plots")

st.markdown(
    """
    <div class="flowpulse-card">
      <h1 style="margin:0;">FlowPulse-FL</h1>
      <p style="margin:6px 0 0; color:#9fb3c8;">
        Communication-aware federated learning for urban traffic forecasting
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

if not summary_path.exists():
    st.error("Run the experiments first to generate `outputs/tables/multiseed_summary.csv`.")
    st.stop()

df = pd.read_csv(summary_path)
full = df[df["selection_strategy"] == "full"]
selected = df[df["selection_strategy"].isin(["random", "flowpulse"])]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
        <div class="flowpulse-card">
          <div class="flowpulse-label">Clients</div>
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

st.write("")

tab1, tab2, tab3 = st.tabs(["📊 Results", "📡 Communication", "📈 Convergence"])

with tab1:
    st.markdown("### Three-seed forecasting performance")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption("Mean and standard deviation across seeds 42, 7, and 2026.")

with tab2:
    st.markdown("### Average communication cost")
    chart = df.pivot(
        index="partition_mode",
        columns="selection_strategy",
        values="communication_bytes_mean",
    )
    st.bar_chart(chart, color=["#4fd1c5", "#f6ad55", "#63b3ed", "#ed64a6"])
    st.caption(
        "Full participation communicates with all 4 clients. Random and FlowPulse "
        "select 2 clients per round, reducing communication by 50%."
    )

with tab3:
    st.markdown("### Training convergence")
    plot_files = sorted(plots.glob("convergence_*.png"))
    if plot_files:
        choice = st.selectbox(
            "Select experiment",
            plot_files,
            format_func=lambda p: p.stem.replace("convergence_", "").replace("_", " "),
        )
        st.image(str(choice), use_container_width=True)
    else:
        st.warning("No convergence plots found.")

st.divider()
st.markdown("### Reproduce the experiment")
st.code(
    "PYTHONPATH=src python -m flowpulse.run_experiments --seed 42 "
    "--rounds 12 --local-epochs 3 --clients 4 --budget 2 "
    "--modes iid non_iid --strategies full random flowpulse",
    language="bash",
)
