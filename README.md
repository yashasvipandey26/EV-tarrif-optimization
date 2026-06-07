# EV Dynamic Tariff Optimization

Agentic AI-based dynamic tariff optimization pipeline for EV charging networks, using ACN session logs and the UrbanEV Shenzhen district dataset.

The project compares a dynamic pricing strategy against a fixed baseline tariff of `₹15/kWh`, then generates model outputs, monitoring metrics, and a PPT-ready presentation deck.

## Project Structure

```text
.
├── ev_tariff_optimization_pipeline.py
├── create_ev_tariff_ppt_deck.py
├── acndata_sessions.json.xlsx - Sheet1.csv
├── UrbanEV_ SZ_districts/
├── urbanEV_SZ_districts -> UrbanEV_ SZ_districts
├── model_evaluation_summary.csv
├── station_utilization_indices.csv
├── dynamic_tariff_table.csv
├── monitoring_learning_simulation.csv
├── engineered_station_hourly_features.csv
├── EV_Dynamic_Tariff_Optimization_Deck.pptx
├── EV_Dynamic_Tariff_Optimization_Deck_Outline.md
└── ppt_assets/
```

`urbanEV_SZ_districts` is a symlink that lets the code use clean prompt-specified paths while preserving the original folder name in the workspace.

## Environment

Use the existing virtual environment in the project root:

```bash
source venv/bin/activate
```

Required packages used by the pipeline and deck generator include:

```bash
pip install pandas numpy scikit-learn lightgbm openpyxl python-pptx matplotlib pillow
```

## Run the Pipeline

```bash
./venv/bin/python ev_tariff_optimization_pipeline.py
```

The pipeline performs:

- Data ingestion from ACN and UrbanEV files
- 5-minute to hourly UrbanEV aggregation
- ACN session aggregation into behavioral day/hour profiles
- Feature engineering for utilization, revenue, occupancy density, queue proxy, spatial clusters, and neighborhood context
- Robust temporal and spatial-neighbor imputation
- Demand prediction using LightGBM with sklearn fallback
- Dynamic tariff generation with bounded surge and discount logic
- Monitoring-agent simulation using a transparent elasticity assumption
- CSV export of all key artifacts

## Generated Outputs

| File | Purpose |
|---|---|
| `model_evaluation_summary.csv` | Final prediction, pricing, and system quality metrics |
| `station_utilization_indices.csv` | Grid-level utilization, queue proxy, throughput, and cluster summaries |
| `dynamic_tariff_table.csv` | Station-hour dynamic tariff recommendations |
| `monitoring_learning_simulation.csv` | Feedback-loop simulation results on the test horizon |
| `engineered_station_hourly_features.csv` | Compact station-hour feature table for EDA and audit |

## Generate the Presentation Deck

```bash
./venv/bin/python create_ev_tariff_ppt_deck.py
```

This creates:

- `EV_Dynamic_Tariff_Optimization_Deck.pptx`
- `EV_Dynamic_Tariff_Optimization_Deck_Outline.md`
- Supporting charts in `ppt_assets/`

The deck includes:

- Cover page
- Executive summary
- Six main slides covering data, EDA, modeling, pricing, monitoring, and implications
- Appendix slides for robustness checks, assumptions, limitations, and output artifacts

## Current Benchmark Snapshot

From the latest run:

- Demand model `R2 Score`: `0.892962`
- Demand model `MAE`: `115.703181 kWh`
- Simulated revenue gain over `₹15/kWh` baseline: `5.267314%`
- Simulated off-peak uplift: `18.926529%`
- Simulated waiting-time proxy reduction: `38.609613%`
- Pricing efficiency score: `₹14.804836/kWh`

## Assumptions and Limitations

- Dynamic pricing outcomes are simulation-based and should not be presented as causal claims.
- The elasticity curve is an explicit modeling assumption, not observed customer response.
- Queue length and waiting time are proxies derived from utilization and occupancy, not directly observed queues.
- ACN and UrbanEV represent different geographies and infrastructure contexts; ACN is used as a supplemental behavioral signal.
- Spatial-neighbor imputation assumes adjacent grids are sufficiently similar for missing interval recovery.
- A phased pilot or randomized rollout is recommended before production pricing policy adoption.

## Recommended Next Steps

- Add rolling-origin backtests against a naive seasonal baseline.
- Run low, medium, and high elasticity sensitivity scenarios.
- Add fairness audits by district, station class, and time of day.
- Validate tariff recommendations through a controlled pilot before operational deployment.
