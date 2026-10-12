from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="FlowPulse-FL", layout="wide")
st.title("FlowPulse-FL")
st.subheader("Federated Traffic-Flow Forecasting Simulator")

st.markdown(
    "FlowPulse-FL treats traffic intersections as federated clients. "
    "Clients train locally and share only model updates; the server combines "
    "updates using FedAvg."
)

tables = Path("outputs/tables")
plots = Path("outputs/plots")
summary_path = tables / "multiseed_summary.csv"

st.header("Three-seed results")
if summary_path.exists():
    st.dataframe(pd.read_csv(summary_path), use_container_width=True)
else:
    st.warning("Run the experiments to create multiseed_summary.csv.")

st.header("Communication cost")
if summary_path.exists():
    df = pd.read_csv(summary_path)
    chart = df.pivot(
        index="partition_mode",
        columns="selection_strategy",
        values="communication_bytes_mean",
    )
    st.bar_chart(chart)

st.header("Convergence plot")
plot_files = sorted(plots.glob("convergence_*.png"))
if plot_files:
    choice = st.selectbox("Experiment", plot_files)
    st.image(str(choice), use_container_width=True)
else:
    st.warning("No plots found.")

st.header("Reproduce")
st.code(
    "PYTHONPATH=src python -m flowpulse.run_experiments --seed 42 "
    "--rounds 12 --local-epochs 3 --clients 4 --budget 2 "
    "--modes iid non_iid --strategies full random flowpulse"
)
