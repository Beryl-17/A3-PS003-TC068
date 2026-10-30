# FlowPulse-FL

AIORI-3 Hackathon · Level 2  
Team Elevate Tech · Team code: A3-PS003-TC068

FlowPulse-FL is an interactive historical traffic-volume forecasting simulator for exploring the trade-off between forecasting accuracy and federated-learning model communication.

It uses the public UCI Metro Interstate Traffic Volume dataset, a compact linear autoregressive forecaster, and four simulated clients partitioned from one dataset. A Streamlit dashboard supports configurable simulations and inspection of saved three-seed benchmarks.

This is not a live traffic service, a physical edge deployment, or a production smart-city system.

## Demo and project materials

- Public live demo: [FlowPulse-FL](https://a3-ps003-tc068.onrender.com) — deployed and publicly tested.
- Repository: https://github.com/aiori-hackathon/A3-PS003-TC068
- Team introduction video: https://youtu.be/IC9YH1ylTws
- Updated presentation: `A3-PS003-TC068-FlowPulse-FL-Presentation.pdf`
  — final Level-2 accuracy review completed against saved three-seed benchmark artifacts.

## What the prototype compares

- Centralized training as a reference.
- Federated training with standard sample-count-weighted FedAvg.
- IID and non-IID training-data partitions.
- Full participation, Random selection, and FlowPulse selection.
- Held-out MAE and RMSE in original traffic-volume units.
- Simulated model-upload/download payload.
- Per-round participation and training-loss histories.

Centralized training appears in saved benchmarks; it is not a selectable
strategy in the interactive simulation.

## Dataset and preprocessing

Dataset: UCI Metro Interstate Traffic Volume  
Source: https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume  
Local file: `data/raw/traffic_volume.csv.gz`

The current preprocessing pipeline is identified as `hourly_v1`:

1. Group duplicate timestamps and retain the first traffic-volume value.
2. Sort timestamps and regularize the timeline to hourly frequency.
3. Leave missing hours unfilled.
4. Split at 80% of the regular hourly timeline.
5. Fit normalization mean and standard deviation using observed
   training-period values only.
6. Retain windows only when all 24 hourly inputs and the next-hour target
   are observed.
7. Assign each window to training or testing according to its target's
   position relative to the split.

A test window may use earlier training-period observations as input
context. Test-period values are not used to fit normalization statistics.

The current dataset produces:

- 18,893 training windows.
- 9,978 test windows.

Four clients are simulated partitions of the training windows, not four
independently measured intersections:

- IID: shuffle training-window indices and divide them among clients.
- Non-IID: divide chronologically ordered training windows into
  contiguous blocks.

## Forecasting model and aggregation

The model predicts the next hour's traffic volume from the preceding
24 hourly values:

    prediction = input_window @ weights + bias

It has 24 weights and one bias. Local training uses full-batch gradient
descent. The fixed benchmark uses learning rate 0.01.

FedAvg combines selected clients' locally trained parameters in proportion
to their training sample counts. FlowPulse changes client selection;
it does not introduce a special aggregation weighting rule.

## Client-selection strategies

### Full participation

Select all four clients every round. The configured client budget does
not restrict Full participation.

### Random selection

Select clients randomly within the configured per-round budget.

### FlowPulse selection

FlowPulse combines:

- A static client target-difference score.
- Simulated readiness sampled uniformly from 0.6 to 1.0 each round.
- Timeout-based selection priority.

The static score is:

    mean(abs(diff(client_targets)))

It is calculated over the client's full target array, not a recent or
live traffic window. In IID partitions, target order is shuffled, so this
score must not be interpreted as chronological traffic variation.

Clients for which `round_idx - last_selected >= 3` receive timeout
priority. Remaining candidates are ranked using:

    0.75 * static_score + 0.25 * readiness

The selection is then limited by the client budget. Timeout priority
does not guarantee selection within a fixed number of rounds, and
fairness guarantees have not been established.

## Verified three-seed benchmark

Configuration:

- Seeds: 42, 7, 2026.
- Four simulated clients.
- 12 training rounds.
- Three local epochs per round.
- Learning rate: 0.01.
- Input window: 24 hours.
- Budget: two clients for Random and FlowPulse.
- Full participation: four clients every round.

The table reports mean ± sample standard deviation across three runs
per group, using `ddof=1`. Values are rounded to three decimal places.

| Partition | Strategy | MAE, mean ± SD | RMSE, mean ± SD | Model-payload bytes per run |
|---|---|---:|---:|---:|
| IID | Centralized | 569.392 ± 13.197 | 807.224 ± 16.416 | 0 |
| IID | Full | 569.405 ± 13.191 | 807.235 ± 16.415 | 19,200 |
| IID | Random | 569.311 ± 12.886 | 807.121 ± 16.952 | 9,600 |
| IID | FlowPulse | 569.152 ± 13.765 | 807.040 ± 16.441 | 9,600 |
| Non-IID | Centralized | 569.392 ± 13.197 | 807.224 ± 16.416 | 0 |
| Non-IID | Full | 569.430 ± 13.192 | 807.252 ± 16.407 | 19,200 |
| Non-IID | Random | 568.858 ± 14.127 | 807.188 ± 16.658 | 9,600 |
| Non-IID | FlowPulse | 569.107 ± 13.730 | 806.955 ± 16.819 | 9,600 |

These are 24 experiment records: three seeds × two partition modes ×
four strategies. Centralized training uses all training windows in each
mode; its IID and non-IID entries are references, not different
centralized client-selection methods.

### Interpretation

Both Random and FlowPulse reduce model-payload communication by 50%
relative to Full participation by selecting two of four clients per
round. This reduction is not unique to FlowPulse.

Forecasting differences are small relative to the across-seed
variability. These results do not establish statistically significant
superiority or universal accuracy gains for FlowPulse.

The table uses corrected hourly preprocessing. Older artifacts under
`outputs/tables/` and `outputs/plots/`, if present, are not the current
benchmark and should not be compared directly with `hourly_v1`.

## Communication accounting

The current NumPy model occupies 200 bytes per upload or download.

Per-round model payload is calculated as:

    2 × model_size_bytes × number_of_selected_clients

For 12 rounds:

- Full: `2 × 200 × 4 × 12 = 19,200 bytes`.
- Budget-two Random/FlowPulse:
  `2 × 200 × 2 × 12 = 9,600 bytes`.

This counts model uploads and downloads only. It excludes:

- Selection metadata.
- Readiness and score exchange.
- Protocol headers and network overhead.
- Dataset distribution and hosting traffic.

Centralized zero means no simulated federated model exchange, not zero
data-transfer cost.

## Dashboard

Start the dashboard from the repository root:

```bash
python -m streamlit run app.py
```

The dashboard contains:

- Overview: project workflow, scope, and fixed benchmark communication.
- Run Simulation: configurable partition mode, selection strategy,
  rounds, local epochs, client budget, and seed.
- Benchmark Results: saved forecast metrics, communication comparisons,
  and convergence plots.

Each simulation trains a fresh model in memory without rewriting
saved benchmark files.

Displayed results identify the last successful run. Changing controls
without submitting does not change those results.

Downloads for the displayed run include:

- Test predictions CSV: `actual` and `predicted`.
- Round-history CSV: participation, loss, and model-payload accounting.
- Run-summary JSON: configuration, metrics, window counts, and caveats.

Export filenames include partition mode, strategy, seed, rounds,
local epochs, and configured budget. Full participation still uses all
four clients regardless of that budget.

The prediction chart uses valid test-target index, not an uninterrupted
hourly timeline; excluded missing-data windows may create time gaps.

The dashboard requires the saved benchmark summary at:

    outputs/hourly_v1/tables/multiseed_summary.csv

## Setup

Run from the repository root.

First-time setup only — skip this command if `.venv` already exists:

```bash
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Dependencies are listed in `requirements.txt`:
NumPy, pandas, Matplotlib, and Streamlit.

## Reproduce the current benchmarks

The following commands execute training and write or replace artifacts
in the specified output directories. They are not read-only checks.

### One seed

```bash
PYTHONPATH=src python -m flowpulse.run_experiments \
  --seed 42 \
  --rounds 12 \
  --local-epochs 3 \
  --learning-rate 0.01 \
  --window-size 24 \
  --clients 4 \
  --budget 2 \
  --modes iid non_iid \
  --strategies full random flowpulse \
  --output-dir outputs/hourly_v1/seed42
```

The runner also generates the centralized reference for each mode.

### All three seeds and aggregation

```bash
for seed in 42 7 2026; do
  PYTHONPATH=src python -m flowpulse.run_experiments \
    --seed "$seed" \
    --rounds 12 \
    --local-epochs 3 \
    --learning-rate 0.01 \
    --window-size 24 \
    --clients 4 \
    --budget 2 \
    --modes iid non_iid \
    --strategies full random flowpulse \
    --output-dir "outputs/hourly_v1/seed${seed}"
done

python scripts/aggregate_hourly_results.py
```

The aggregation script is specific to the fixed benchmark: seeds
42, 7, and 2026; 12 rounds; and expected model-payload totals of
19,200 bytes for Full and 9,600 bytes for Random/FlowPulse. It is not
a general-purpose aggregator for arbitrary simulation settings.

The script writes the combined tables. It has no command-line argument
parser: passing `--help` also executes aggregation and must not be used
as a read-only inspection command.

## Current artifact locations

Per-seed tables:

```text
outputs/hourly_v1/seed{seed}/tables/experiment_summary.csv
outputs/hourly_v1/seed{seed}/tables/all_round_history.csv
```

Per-seed plots:

```text
outputs/hourly_v1/seed{seed}/plots/convergence_*.png
```

Combined tables:

```text
outputs/hourly_v1/tables/experiment_summary.csv
outputs/hourly_v1/tables/all_round_history.csv
outputs/hourly_v1/tables/multiseed_summary.csv
```

## Tests and validation

Run the existing unittest suite:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The six existing tests passed and cover:

- Missing inputs and targets excluded from windows.
- Training-only normalization.
- Model payload size.
- Sample-weighted FedAvg.
- Selection and round accounting.
- Simulation input rejection before data loading.

An independent saved-artifact reconciliation also passed, checking
experiment coverage, history rounds, selected-client counts, payload
totals, combined/source-table consistency, and recalculated multi-seed
means and sample standard deviations.

Local dashboard smoke checks confirmed:

- Full participation selects four clients across all 12 rounds.
- Prediction CSV opens with the expected columns and numeric visible rows.
- Saved forecast, communication, and convergence views render.
- New history and summary exports open with the expected content.

These checks do not establish correctness for every application behavior
or replace testing of the eventual public deployment.

## Project structure

```text
app.py                              Streamlit dashboard
requirements.txt                    Python dependencies
data/raw/traffic_volume.csv.gz      Local public dataset
src/flowpulse/
  __init__.py
  data.py                           Hourly preprocessing and client partitions
  model.py                          Linear model and local training
  federated.py                      Selection, FedAvg, and training loop
  simulation.py                     In-memory dashboard simulation
  run_experiments.py                Benchmark runner and artifacts
scripts/
  aggregate_hourly_results.py        Combined benchmark summaries
tests/
  test_core.py                      Unit tests
outputs/hourly_v1/                   Current saved benchmark artifacts
presentation_assets/                Existing supporting images
A3-PS003-TC068-FlowPulse-FL-Presentation.pdf
```

## Limitations

- Historical observations, not live traffic.
- One dataset partitioned into four simulated clients.
- Local single-process simulation, not distributed physical devices.
- Linear forecasting model; no LSTM or other deep-learning model.
- Static selection score and simulated readiness.
- No guaranteed fairness or participation deadline.
- No measured end-to-end network overhead.
- No differential privacy, secure aggregation, or formal privacy
  guarantee is claimed.
- Three seeds provide a limited descriptive comparison, not evidence
  of statistically significant superiority.

Federated `train_mse` is the mean selected-client post-local-training
loss. It is not the global model's loss on a fixed evaluation dataset.
Centralized training loss uses all training windows. These quantities
are not equivalent and should not alone support claims of faster global
convergence.

## Status

- [x] Public dataset integrated
- [x] Corrected hourly preprocessing and chronological split
- [x] Training-only normalization
- [x] IID and non-IID simulated client partitions
- [x] Compact linear forecasting model
- [x] Sample-weighted FedAvg
- [x] Full, Random, and FlowPulse client selection
- [x] MAE, RMSE, model-payload accounting, and training histories
- [x] Multi-seed statistical summary
- [x] Three-seed summary and artifact reconciliation
- [x] Local Streamlit dashboard and smoke checks
- [x] Prediction, history, and run-summary downloads
- [x] Existing team introduction video
- [x] Final Level-2 presentation review
- [x] Public demo deployment and public-link testing
- [x] Final repository and submission review
