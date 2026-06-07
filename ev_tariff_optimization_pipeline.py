"""
Agentic AI-Based Dynamic Tariff Optimization for EV charging networks.

This script builds a schema-aware, end-to-end pipeline over the ACN session logs
and UrbanEV Shenzhen district panel. It creates local compatibility aliases when
needed, then ingests the exact paths requested in the project brief.
"""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from lightgbm import LGBMRegressor
except Exception:  # LightGBM is preferred, sklearn fallback keeps the pipeline runnable.
    LGBMRegressor = None

SEED = 42
BASELINE_TARIFF_RUPEE_PER_KWH = 15.0
SURGE_UTILIZATION_THRESHOLD = 0.80
DISCOUNT_UTILIZATION_THRESHOLD = 0.30
MIN_TARIFF_MULTIPLIER = 0.85
MAX_TARIFF_MULTIPLIER = 1.45
PRICE_ELASTICITY = -0.28

random.seed(SEED)
np.random.seed(SEED)


def rmse(actual: pd.Series, predicted: pd.Series) -> float:
    return float(np.sqrt(mean_squared_error(actual, predicted)))


@dataclass(frozen=True)
class PipelineOutputs:
    panel: pd.DataFrame
    forecasts: pd.DataFrame
    tariffs: pd.DataFrame
    simulation: pd.DataFrame
    benchmarks: pd.DataFrame
    station_indices: pd.DataFrame


def ensure_input_aliases() -> None:
    """Make prompt-specified paths available without mutating source data.

    The workspace currently stores UrbanEV under ``UrbanEV_ SZ_districts`` and
    ACN as an Excel workbook. The modeling code below still uses the exact
    ``pd.read_csv`` paths from the brief; this bridge only creates derived local
    aliases when those prompt paths are absent.
    """
    requested_urban = Path("urbanEV_SZ_districts")
    actual_urban = Path("UrbanEV_ SZ_districts")
    if not requested_urban.exists() and actual_urban.exists():
        try:
            requested_urban.symlink_to(actual_urban, target_is_directory=True)
        except OSError:
            shutil.copytree(actual_urban, requested_urban)

    requested_acn = Path("acndata_sessions.json.xlsx - Sheet1.csv")
    actual_acn = Path("acndata_sessions.json.xlsx")
    if not requested_acn.exists() and actual_acn.exists():
        pd.read_excel(actual_acn, sheet_name="Sheet1").to_csv(requested_acn, index=False)


def read_inputs() -> Dict[str, pd.DataFrame]:
    ensure_input_aliases()

    acn = pd.read_csv('acndata_sessions.json.xlsx - Sheet1.csv')
    duration = pd.read_csv('urbanEV_SZ_districts/duration.csv')
    information = pd.read_csv('urbanEV_SZ_districts/information.csv')
    occupancy = pd.read_csv('urbanEV_SZ_districts/occupancy.csv')
    volume = pd.read_csv('urbanEV_SZ_districts/volume.csv')
    price = pd.read_csv('urbanEV_SZ_districts/price.csv')
    stations = pd.read_csv('urbanEV_SZ_districts/stations.csv')
    adj = pd.read_csv('urbanEV_SZ_districts/adj.csv')
    distance = pd.read_csv('urbanEV_SZ_districts/distance.csv')
    time_index = pd.read_csv('urbanEV_SZ_districts/time.csv')

    return {
        "acn": acn,
        "duration": duration,
        "information": information,
        "occupancy": occupancy,
        "volume": volume,
        "price": price,
        "stations": stations,
        "adj": adj,
        "distance": distance,
        "time": time_index,
    }


def attach_datetime(panel: pd.DataFrame, time_index: pd.DataFrame) -> pd.DataFrame:
    indexed_time = time_index.copy()
    indexed_time["datetime"] = pd.to_datetime(indexed_time[["year", "month", "day", "hour", "minute", "second"]])
    timestamp_to_datetime = pd.Series(indexed_time["datetime"].values, index=np.arange(1, len(indexed_time) + 1))
    output = panel.copy()
    output["datetime"] = output["timestamp"].map(timestamp_to_datetime)
    return output


def wide_to_hourly_long(raw: pd.DataFrame, time_index: pd.DataFrame, value_name: str, agg: str) -> pd.DataFrame:
    dated = attach_datetime(raw, time_index)
    station_cols = [col for col in dated.columns if col not in {"timestamp", "datetime"}]
    long = dated.melt(id_vars=["timestamp", "datetime"], value_vars=station_cols, var_name="grid", value_name=value_name)
    long["grid"] = long["grid"].astype(int)
    long["hour_start"] = long["datetime"].dt.floor("h")
    grouped = long.groupby(["hour_start", "grid"], observed=True)[value_name]
    hourly = getattr(grouped, agg)().reset_index()
    return hourly


def build_spatial_matrices(adj: pd.DataFrame, distance: pd.DataFrame, grids: Iterable[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    grid_list = [str(grid) for grid in grids]
    adj_index_col = "node_id" if "node_id" in adj.columns else adj.columns[0]
    adj_matrix = adj.rename(columns={adj_index_col: "grid"}).copy()
    adj_matrix["grid"] = adj_matrix["grid"].astype(int).astype(str)
    adj_matrix = adj_matrix.set_index("grid").reindex(index=grid_list, columns=grid_list).fillna(0).astype(float)

    dist_index_col = "node_id" if "node_id" in distance.columns else distance.columns[0]
    dist_matrix = distance.rename(columns={dist_index_col: "grid"}).copy()
    if "grid" not in dist_matrix.columns:
        dist_matrix.insert(0, "grid", grid_list[: len(dist_matrix)])
    dist_matrix["grid"] = pd.Series(grid_list[: len(dist_matrix)], index=dist_matrix.index)
    dist_matrix = dist_matrix.set_index("grid").reindex(index=grid_list, columns=grid_list).fillna(0).astype(float)
    return adj_matrix, dist_matrix


def add_station_clusters(info: pd.DataFrame, n_clusters: int = 8) -> pd.DataFrame:
    enriched = info.copy()
    coords = enriched[["lon", "la"]].to_numpy(dtype=float)
    n_clusters = min(n_clusters, len(enriched))
    labels = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10).fit_predict(coords)
    enriched["spatial_cluster"] = labels
    return enriched


def enrich_grids_with_station_metadata(info: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    enriched = info.copy()
    station_points = stations[["latitude", "longitude", "fast", "slow", "count"]].dropna().copy()
    if station_points.empty:
        enriched["nearby_station_count"] = 0
        enriched["nearby_station_fast_count"] = 0
        enriched["nearby_station_slow_count"] = 0
        enriched["nearest_station_distance_km"] = 0.0
        return enriched

    grid_lat = np.deg2rad(enriched["la"].to_numpy(dtype=float))
    grid_lon = np.deg2rad(enriched["lon"].to_numpy(dtype=float))
    station_lat = np.deg2rad(station_points["latitude"].to_numpy(dtype=float))[:, None]
    station_lon = np.deg2rad(station_points["longitude"].to_numpy(dtype=float))[:, None]
    dlat = station_lat - grid_lat
    dlon = station_lon - grid_lon
    haversine = np.sin(dlat / 2) ** 2 + np.cos(station_lat) * np.cos(grid_lat) * np.sin(dlon / 2) ** 2
    distances_km = 6371.0 * 2 * np.arcsin(np.sqrt(haversine))
    nearest_grid_positions = distances_km.argmin(axis=1)
    nearest_distances = distances_km.min(axis=1)

    assignments = pd.DataFrame(
        {
            "grid": enriched.iloc[nearest_grid_positions]["grid"].to_numpy(),
            "station_fast": station_points["fast"].to_numpy(),
            "station_slow": station_points["slow"].to_numpy(),
            "station_count": station_points["count"].to_numpy(),
            "nearest_station_distance_km": nearest_distances,
        }
    )
    grid_station_features = (
        assignments.groupby("grid", observed=True)
        .agg(
            nearby_station_count=("station_count", "sum"),
            nearby_station_fast_count=("station_fast", "sum"),
            nearby_station_slow_count=("station_slow", "sum"),
            nearest_station_distance_km=("nearest_station_distance_km", "min"),
        )
        .reset_index()
    )
    enriched = enriched.merge(grid_station_features, on="grid", how="left")
    fill_values = {
        "nearby_station_count": 0,
        "nearby_station_fast_count": 0,
        "nearby_station_slow_count": 0,
        "nearest_station_distance_km": float(np.nanmedian(nearest_distances)),
    }
    enriched = enriched.fillna(fill_values)
    return enriched


def spatial_temporal_impute(panel: pd.DataFrame, numeric_cols: List[str], adj_matrix: pd.DataFrame) -> pd.DataFrame:
    output = panel.sort_values(["grid", "hour_start"]).copy()
    output[numeric_cols] = output.groupby("grid", observed=True)[numeric_cols].ffill()
    output[numeric_cols] = output.groupby("grid", observed=True)[numeric_cols].bfill()

    missing_mask = output[numeric_cols].isna().any(axis=1)
    if missing_mask.any():
        # Assumption: if an interval is absent after temporal forward/back-fill,
        # nearby connected grids with valid same-hour readings are the best proxy.
        for row_index in output.index[missing_mask]:
            grid = str(int(output.at[row_index, "grid"]))
            hour = output.at[row_index, "hour_start"]
            if grid not in adj_matrix.index:
                continue
            neighbors = adj_matrix.columns[adj_matrix.loc[grid].to_numpy() > 0].astype(int).tolist()
            if not neighbors:
                continue
            neighbor_rows = output[(output["hour_start"] == hour) & (output["grid"].isin(neighbors))]
            for col in numeric_cols:
                if pd.isna(output.at[row_index, col]) and not neighbor_rows[col].dropna().empty:
                    output.at[row_index, col] = float(neighbor_rows[col].mean())

    output[numeric_cols] = output[numeric_cols].fillna(output[numeric_cols].median(numeric_only=True))
    return output


def aggregate_acn_hourly(acn: pd.DataFrame) -> pd.DataFrame:
    acn_clean = acn.copy()
    acn_clean["connectionTime"] = pd.to_datetime(acn_clean["connectionTime"], utc=True, errors="coerce")
    acn_clean["disconnectTime"] = pd.to_datetime(acn_clean["disconnectTime"], utc=True, errors="coerce")
    acn_clean["doneChargingTime"] = pd.to_datetime(acn_clean["doneChargingTime"], utc=True, errors="coerce")
    acn_clean["kWhDelivered"] = pd.to_numeric(acn_clean["kWhDelivered"], errors="coerce").fillna(0)
    acn_clean = acn_clean.dropna(subset=["connectionTime", "disconnectTime"])
    acn_clean["connection_duration_hours"] = (
        (acn_clean["disconnectTime"] - acn_clean["connectionTime"]).dt.total_seconds() / 3600
    ).clip(lower=0, upper=48)
    acn_clean["hour_start"] = acn_clean["connectionTime"].dt.floor("h").dt.tz_localize(None)
    hourly = (
        acn_clean.groupby("hour_start", observed=True)
        .agg(
            acn_sessions=("sessionID", "count"),
            acn_energy_kwh=("kWhDelivered", "sum"),
            acn_duration_hours=("connection_duration_hours", "sum"),
            acn_unique_stations=("stationID", "nunique"),
        )
        .reset_index()
    )
    hourly["hour"] = hourly["hour_start"].dt.hour
    hourly["dayofweek"] = hourly["hour_start"].dt.dayofweek
    profile = hourly.groupby(["dayofweek", "hour"], observed=True).mean(numeric_only=True).reset_index()
    profile = profile.rename(columns={col: f"profile_{col}" for col in profile.columns if col not in {"dayofweek", "hour"}})
    return profile


def build_feature_panel(inputs: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    info = add_station_clusters(enrich_grids_with_station_metadata(inputs["information"], inputs["stations"]))
    grids = info["grid"].astype(int).tolist()
    adj_matrix, dist_matrix = build_spatial_matrices(inputs["adj"], inputs["distance"], grids)

    volume_h = wide_to_hourly_long(inputs["volume"], inputs["time"], "volume_kwh", "sum")
    duration_h = wide_to_hourly_long(inputs["duration"], inputs["time"], "charging_duration_hours", "sum")
    occupancy_h = wide_to_hourly_long(inputs["occupancy"], inputs["time"], "busy_piles", "mean")
    price_h = wide_to_hourly_long(inputs["price"], inputs["time"], "historical_price_multiplier", "mean")

    panel = volume_h.merge(duration_h, on=["hour_start", "grid"], how="outer")
    panel = panel.merge(occupancy_h, on=["hour_start", "grid"], how="outer")
    panel = panel.merge(price_h, on=["hour_start", "grid"], how="outer")
    panel = panel.merge(info, on="grid", how="left")
    panel = spatial_temporal_impute(
        panel,
        ["volume_kwh", "charging_duration_hours", "busy_piles", "historical_price_multiplier"],
        adj_matrix,
    )

    panel["count"] = panel["count"].replace(0, np.nan).fillna(panel["count"].median())
    panel["fast_share"] = panel["fast_count"] / panel["count"].clip(lower=1)
    panel["slow_share"] = panel["slow_count"] / panel["count"].clip(lower=1)
    panel["utilization_rate"] = (panel["charging_duration_hours"] / panel["count"].clip(lower=1)).clip(0, 1.5)
    panel["occupancy_density"] = (panel["busy_piles"] / panel["count"].clip(lower=1)).clip(lower=0)
    panel["queue_length_proxy"] = (panel["busy_piles"] - panel["count"]).clip(lower=0) + (
        panel["occupancy_density"] - 0.85
    ).clip(lower=0) * panel["count"] * 0.25
    panel["baseline_revenue_rupee"] = panel["volume_kwh"] * BASELINE_TARIFF_RUPEE_PER_KWH
    panel["historical_revenue_rupee"] = (
        panel["volume_kwh"] * BASELINE_TARIFF_RUPEE_PER_KWH * panel["historical_price_multiplier"]
    )
    panel["hour"] = panel["hour_start"].dt.hour
    panel["dayofweek"] = panel["hour_start"].dt.dayofweek
    panel["is_weekend"] = panel["dayofweek"].isin([5, 6]).astype(int)
    panel["is_peak_hour"] = panel["hour"].isin([8, 9, 10, 17, 18, 19, 20, 21]).astype(int)
    panel["sin_hour"] = np.sin(2 * np.pi * panel["hour"] / 24)
    panel["cos_hour"] = np.cos(2 * np.pi * panel["hour"] / 24)

    for metric in ["volume_kwh", "utilization_rate", "occupancy_density", "queue_length_proxy"]:
        wide = panel.pivot(index="hour_start", columns="grid", values=metric).reindex(columns=grids)
        neighbor_sum = wide.dot(adj_matrix.to_numpy())
        neighbor_count = adj_matrix.sum(axis=0).replace(0, np.nan).to_numpy()
        neighbor_mean = pd.DataFrame(
            neighbor_sum.to_numpy() / neighbor_count,
            index=wide.index,
            columns=[f"neighbor_{metric}_{grid}" for grid in grids],
        )
        long_neighbor = neighbor_mean.reset_index().melt("hour_start", var_name="neighbor_metric", value_name=f"neighbor_{metric}")
        long_neighbor["grid"] = long_neighbor["neighbor_metric"].str.rsplit("_", n=1).str[-1].astype(int)
        panel = panel.merge(long_neighbor[["hour_start", "grid", f"neighbor_{metric}"]], on=["hour_start", "grid"], how="left")
        panel[f"neighbor_{metric}"] = panel[f"neighbor_{metric}"].fillna(panel[metric])

    acn_profile = aggregate_acn_hourly(inputs["acn"])
    panel = panel.merge(acn_profile, on=["dayofweek", "hour"], how="left")
    acn_cols = [col for col in panel.columns if col.startswith("profile_acn_")]
    panel[acn_cols] = panel[acn_cols].fillna(panel[acn_cols].median(numeric_only=True))

    panel = panel.sort_values(["grid", "hour_start"]).reset_index(drop=True)
    panel["target_volume_kwh"] = panel.groupby("grid", observed=True)["volume_kwh"].shift(-1)
    panel["target_utilization_rate"] = panel.groupby("grid", observed=True)["utilization_rate"].shift(-1)
    panel = panel.dropna(subset=["target_volume_kwh", "target_utilization_rate"]).reset_index(drop=True)

    station_indices = (
        panel.groupby("grid", observed=True)
        .agg(
            avg_utilization_rate=("utilization_rate", "mean"),
            p95_utilization_rate=("utilization_rate", lambda x: np.percentile(x, 95)),
            avg_occupancy_density=("occupancy_density", "mean"),
            avg_queue_length_proxy=("queue_length_proxy", "mean"),
            total_volume_kwh=("volume_kwh", "sum"),
            baseline_revenue_rupee=("baseline_revenue_rupee", "sum"),
            spatial_cluster=("spatial_cluster", "first"),
            charger_count=("count", "first"),
            nearby_station_count=("nearby_station_count", "first"),
        )
        .reset_index()
    )
    return panel, station_indices, dist_matrix


class DemandPredictionAgent:
    def __init__(self, random_state: int = SEED):
        self.random_state = random_state
        self.volume_model = self._make_model("volume")
        self.utilization_model = self._make_model("utilization")
        self.feature_columns: List[str] = []
        self.metrics: Dict[str, float] = {}

    def _make_model(self, name: str):
        if LGBMRegressor is not None:
            return LGBMRegressor(
                objective="regression",
                n_estimators=260,
                learning_rate=0.055,
                num_leaves=63,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
            )
        return HistGradientBoostingRegressor(max_iter=220, learning_rate=0.06, random_state=self.random_state)

    def fit(self, panel: pd.DataFrame) -> pd.DataFrame:
        self.feature_columns = [
            "volume_kwh",
            "charging_duration_hours",
            "busy_piles",
            "historical_price_multiplier",
            "count",
            "fast_count",
            "slow_count",
            "nearby_station_count",
            "nearby_station_fast_count",
            "nearby_station_slow_count",
            "nearest_station_distance_km",
            "area",
            "lon",
            "la",
            "CBD",
            "dynamic_pricing",
            "spatial_cluster",
            "fast_share",
            "slow_share",
            "utilization_rate",
            "occupancy_density",
            "queue_length_proxy",
            "hour",
            "dayofweek",
            "is_weekend",
            "is_peak_hour",
            "sin_hour",
            "cos_hour",
            "neighbor_volume_kwh",
            "neighbor_utilization_rate",
            "neighbor_occupancy_density",
            "neighbor_queue_length_proxy",
            "profile_acn_sessions",
            "profile_acn_energy_kwh",
            "profile_acn_duration_hours",
            "profile_acn_unique_stations",
        ]
        train_idx, test_idx = train_test_split(np.arange(len(panel)), test_size=0.2, random_state=self.random_state, shuffle=False)
        train, test = panel.iloc[train_idx], panel.iloc[test_idx]
        x_train = train[self.feature_columns]
        x_test = test[self.feature_columns]
        self.volume_model.fit(x_train, train["target_volume_kwh"])
        self.utilization_model.fit(x_train, train["target_utilization_rate"])

        predictions = test[["hour_start", "grid", "target_volume_kwh", "target_utilization_rate"]].copy()
        predictions["predicted_volume_kwh"] = np.clip(self.volume_model.predict(x_test), 0, None)
        predictions["predicted_utilization_rate"] = np.clip(self.utilization_model.predict(x_test), 0, 1.5)
        predictions["congestion_probability"] = 1 / (
            1 + np.exp(-12 * (predictions["predicted_utilization_rate"] - SURGE_UTILIZATION_THRESHOLD))
        )
        self.metrics = {
            "RMSE": rmse(predictions["target_volume_kwh"], predictions["predicted_volume_kwh"]),
            "MAE": float(mean_absolute_error(predictions["target_volume_kwh"], predictions["predicted_volume_kwh"])),
            "R2 Score": float(r2_score(predictions["target_volume_kwh"], predictions["predicted_volume_kwh"])),
            "Utilization RMSE": rmse(
                predictions["target_utilization_rate"], predictions["predicted_utilization_rate"]
            ),
        }
        return predictions

    def predict(self, panel: pd.DataFrame) -> pd.DataFrame:
        output = panel[["hour_start", "grid", "target_volume_kwh", "target_utilization_rate"]].copy()
        output["predicted_volume_kwh"] = np.clip(self.volume_model.predict(panel[self.feature_columns]), 0, None)
        output["predicted_utilization_rate"] = np.clip(self.utilization_model.predict(panel[self.feature_columns]), 0, 1.5)
        output["congestion_probability"] = 1 / (
            1 + np.exp(-12 * (output["predicted_utilization_rate"] - SURGE_UTILIZATION_THRESHOLD))
        )
        return output


class TariffPricingAgent:
    def __init__(self, baseline_tariff: float = BASELINE_TARIFF_RUPEE_PER_KWH):
        self.baseline_tariff = baseline_tariff

    def price(self, forecasts: pd.DataFrame) -> pd.DataFrame:
        tariffs = forecasts.copy()
        util = tariffs["predicted_utilization_rate"]
        congestion = tariffs["congestion_probability"]
        surge_component = np.where(
            util > SURGE_UTILIZATION_THRESHOLD,
            1 + 0.75 * (util - SURGE_UTILIZATION_THRESHOLD) + 0.28 * congestion,
            1.0,
        )
        discount_component = np.where(
            util < DISCOUNT_UTILIZATION_THRESHOLD,
            1 - 0.12 * (DISCOUNT_UTILIZATION_THRESHOLD - util) / DISCOUNT_UTILIZATION_THRESHOLD,
            1.0,
        )
        multiplier = np.clip(surge_component * discount_component, MIN_TARIFF_MULTIPLIER, MAX_TARIFF_MULTIPLIER)
        tariffs["tariff_multiplier"] = multiplier
        tariffs["dynamic_tariff_rupee_per_kwh"] = self.baseline_tariff * multiplier
        tariffs["pricing_signal"] = np.select(
            [util > SURGE_UTILIZATION_THRESHOLD, util < DISCOUNT_UTILIZATION_THRESHOLD],
            ["SURGE", "DISCOUNT"],
            default="BASELINE",
        )
        tariffs["expected_dynamic_revenue_rupee"] = tariffs["predicted_volume_kwh"] * tariffs[
            "dynamic_tariff_rupee_per_kwh"
        ]
        tariffs["expected_flat_revenue_rupee"] = tariffs["predicted_volume_kwh"] * self.baseline_tariff
        return tariffs


class MonitoringLearningAgent:
    def __init__(self, baseline_tariff: float = BASELINE_TARIFF_RUPEE_PER_KWH, elasticity: float = PRICE_ELASTICITY):
        self.baseline_tariff = baseline_tariff
        self.elasticity = elasticity

    def simulate(self, priced: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
        simulated = priced.copy()
        relative_price_change = (simulated["dynamic_tariff_rupee_per_kwh"] - self.baseline_tariff) / self.baseline_tariff
        demand_factor = np.clip(1 + self.elasticity * relative_price_change, 0.72, 1.22)
        off_peak_discount_boost = np.where(simulated["pricing_signal"].eq("DISCOUNT"), 0.18, 0.0)
        simulated["simulated_volume_kwh"] = simulated["predicted_volume_kwh"] * (demand_factor + off_peak_discount_boost)
        simulated["simulated_utilization_rate"] = np.clip(
            simulated["predicted_utilization_rate"] * (0.72 + 0.28 * (demand_factor + off_peak_discount_boost)), 0, 1.5
        )
        simulated["flat_revenue_rupee"] = simulated["predicted_volume_kwh"] * self.baseline_tariff
        simulated["dynamic_revenue_rupee"] = simulated["simulated_volume_kwh"] * simulated[
            "dynamic_tariff_rupee_per_kwh"
        ]
        simulated["baseline_waiting_proxy"] = np.maximum(0, simulated["predicted_utilization_rate"] - 0.85) * 60
        simulated["dynamic_waiting_proxy"] = np.maximum(0, simulated["simulated_utilization_rate"] - 0.85) * 60

        peak = simulated["predicted_utilization_rate"] >= SURGE_UTILIZATION_THRESHOLD
        off_peak = simulated["predicted_utilization_rate"] <= DISCOUNT_UTILIZATION_THRESHOLD
        revenue_gain_pct = 100 * (
            simulated["dynamic_revenue_rupee"].sum() / simulated["flat_revenue_rupee"].sum() - 1
        )
        stabilization_pct = 100 * (
            simulated.loc[peak, "predicted_utilization_rate"].mean()
            - simulated.loc[peak, "simulated_utilization_rate"].mean()
        ) / max(simulated.loc[peak, "predicted_utilization_rate"].mean(), 1e-9)
        off_peak_uplift_pct = 100 * (
            simulated.loc[off_peak, "simulated_volume_kwh"].sum()
            / max(simulated.loc[off_peak, "predicted_volume_kwh"].sum(), 1e-9)
            - 1
        )
        waiting_reduction_pct = 100 * (
            simulated["baseline_waiting_proxy"].mean() - simulated["dynamic_waiting_proxy"].mean()
        ) / max(simulated["baseline_waiting_proxy"].mean(), 1e-9)
        pricing_efficiency = simulated["dynamic_revenue_rupee"].sum() / max(simulated["simulated_volume_kwh"].sum(), 1e-9)
        metrics = {
            "Revenue Gain %": float(revenue_gain_pct),
            "Net Utilization Stabilization %": float(stabilization_pct if np.isfinite(stabilization_pct) else 0),
            "Off-Peak Uplift %": float(off_peak_uplift_pct if np.isfinite(off_peak_uplift_pct) else 0),
            "Average Waiting Time Reduction %": float(waiting_reduction_pct if np.isfinite(waiting_reduction_pct) else 0),
            "Pricing Efficiency Score": float(pricing_efficiency),
        }
        return simulated, metrics


def evaluate(agent_metrics: Dict[str, float], operational_metrics: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for metric, value in agent_metrics.items():
        rows.append({"metric_group": "Prediction Metrics", "metric": metric, "value": value})
    for metric, value in operational_metrics.items():
        group = "Pricing Optimization Metrics"
        if metric in {"Average Waiting Time Reduction %", "Pricing Efficiency Score"}:
            group = "System Quality Metrics"
        rows.append({"metric_group": group, "metric": metric, "value": value})
    return pd.DataFrame(rows)


def run_pipeline() -> PipelineOutputs:
    inputs = read_inputs()
    panel, station_indices, _ = build_feature_panel(inputs)
    demand_agent = DemandPredictionAgent(random_state=SEED)
    test_forecasts = demand_agent.fit(panel)
    all_forecasts = demand_agent.predict(panel)
    pricing_agent = TariffPricingAgent()
    tariffs = pricing_agent.price(all_forecasts)
    test_tariffs = pricing_agent.price(test_forecasts)
    monitoring_agent = MonitoringLearningAgent()
    simulation, operational_metrics = monitoring_agent.simulate(test_tariffs)
    benchmarks = evaluate(demand_agent.metrics, operational_metrics)
    return PipelineOutputs(panel, all_forecasts, tariffs, simulation, benchmarks, station_indices)


def save_outputs(outputs: PipelineOutputs) -> None:
    outputs.benchmarks.to_csv("model_evaluation_summary.csv", index=False)
    outputs.station_indices.to_csv("station_utilization_indices.csv", index=False)
    outputs.tariffs.to_csv("dynamic_tariff_table.csv", index=False)
    outputs.simulation.to_csv("monitoring_learning_simulation.csv", index=False)
    compact_features = outputs.panel[
        [
            "hour_start",
            "grid",
            "volume_kwh",
            "utilization_rate",
            "occupancy_density",
            "queue_length_proxy",
            "baseline_revenue_rupee",
            "historical_revenue_rupee",
            "spatial_cluster",
        ]
    ]
    compact_features.to_csv("engineered_station_hourly_features.csv", index=False)


if __name__ == "__main__":
    outputs = run_pipeline()
    save_outputs(outputs)
    print("\nFinal performance benchmarks")
    print(outputs.benchmarks.to_string(index=False))
    print("\nSaved CSV outputs")
    for output_file in [
        "model_evaluation_summary.csv",
        "station_utilization_indices.csv",
        "dynamic_tariff_table.csv",
        "monitoring_learning_simulation.csv",
        "engineered_station_hourly_features.csv",
    ]:
        print(f"- {output_file}")
