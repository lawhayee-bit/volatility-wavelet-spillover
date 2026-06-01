from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .utils import ensure_directory, utc_datestamp


STOOQ_DAILY_URL = "https://stooq.com/q/d/l/?s={symbol}&i=d"
FRED_GRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def _fetch_csv_text(url: str, timeout: int = 60) -> str:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def _snapshot_path(root: Path, source: str, series_name: str, extension: str = "csv") -> Path:
    snapshot_dir = ensure_directory(root / source)
    return snapshot_dir / f"{series_name}_{utc_datestamp()}.{extension}"


def download_stooq_snapshot(symbol: str, raw_root: str | Path) -> Path:
    raw_root = Path(raw_root)
    series_name = symbol.replace("^", "").replace(".", "_").lower()
    output_path = _snapshot_path(raw_root, "stooq", series_name)
    csv_text = _fetch_csv_text(STOOQ_DAILY_URL.format(symbol=symbol))
    output_path.write_text(csv_text, encoding="utf-8")
    return output_path


def download_fred_snapshot(series_id: str, raw_root: str | Path) -> Path:
    raw_root = Path(raw_root)
    output_path = _snapshot_path(raw_root, "fred", series_id.lower())
    csv_text = _fetch_csv_text(FRED_GRAPH_URL.format(series_id=series_id))
    output_path.write_text(csv_text, encoding="utf-8")
    return output_path


def load_stooq_csv(path: str | Path, index_id: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [col.strip().lower() for col in df.columns]
    df = df.rename(columns={"volume": "index_volume"})
    df["date"] = pd.to_datetime(df["date"])
    numeric_cols = ["open", "high", "low", "close", "index_volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["index_id"] = index_id
    return df.sort_values("date").reset_index(drop=True)


def load_fred_csv(path: str | Path, series_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, na_values=["."])
    df.columns = ["date", series_name]
    df["date"] = pd.to_datetime(df["date"])
    df[series_name] = pd.to_numeric(df[series_name], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def download_all_sources(config: dict[str, Any], raw_root: str | Path = "data/raw") -> dict[str, Any]:
    raw_root = Path(raw_root)
    outputs: dict[str, Any] = {"indices": {}, "fred": {}}

    for item in config["data"]["indices"]:
        index_id = item["id"]
        outputs["indices"][index_id] = {
            "price": download_stooq_snapshot(item["stooq_symbol"], raw_root),
            "etf": download_stooq_snapshot(item["etf_symbol"], raw_root),
            "vix": download_fred_snapshot(item["fred_vix_series"], raw_root),
        }

    for field_name, series_id in config["data"]["fred_series"].items():
        outputs["fred"][field_name] = download_fred_snapshot(series_id, raw_root)

    return outputs


def build_raw_panel(config: dict[str, Any], raw_root: str | Path = "data/raw") -> pd.DataFrame:
    raw_root = Path(raw_root)
    start_date = pd.Timestamp(config["data"]["start_date"])
    end_date = pd.Timestamp(config["data"]["end_date"])

    shared_frames: list[pd.DataFrame] = []
    for field_name, series_id in config["data"]["fred_series"].items():
        latest_path = sorted((raw_root / "fred").glob(f"{series_id.lower()}_*.csv"))[-1]
        shared_frames.append(load_fred_csv(latest_path, field_name))

    shared = shared_frames[0]
    for frame in shared_frames[1:]:
        shared = shared.merge(frame, on="date", how="outer")
    shared = shared.sort_values("date").ffill()

    panels: list[pd.DataFrame] = []
    for item in config["data"]["indices"]:
        index_id = item["id"]
        stooq_name = item["stooq_symbol"].replace("^", "").replace(".", "_").lower()
        etf_name = item["etf_symbol"].replace("^", "").replace(".", "_").lower()
        fred_name = item["fred_vix_series"].lower()

        price_path = sorted((raw_root / "stooq").glob(f"{stooq_name}_*.csv"))[-1]
        etf_path = sorted((raw_root / "stooq").glob(f"{etf_name}_*.csv"))[-1]
        vix_path = sorted((raw_root / "fred").glob(f"{fred_name}_*.csv"))[-1]

        price = load_stooq_csv(price_path, index_id=index_id)
        etf = load_stooq_csv(etf_path, index_id=index_id)[["date", "close", "index_volume"]]
        etf = etf.rename(columns={"close": "etf_close_proxy", "index_volume": "etf_volume_proxy"})
        vix = load_fred_csv(vix_path, "vix_like")

        merged = price.merge(etf, on="date", how="left")
        merged = merged.merge(vix, on="date", how="left")
        merged = merged.merge(shared, on="date", how="left")
        merged = merged.sort_values("date").ffill()
        merged["index_label"] = item["label"]

        mask = (merged["date"] >= start_date) & (merged["date"] <= end_date)
        merged = merged.loc[mask].copy()
        panels.append(merged)

    panel = pd.concat(panels, ignore_index=True)
    panel = panel.sort_values(["index_id", "date"]).reset_index(drop=True)
    return panel


def save_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    ensure_directory(path.parent)
    if path.suffix.endswith("gz") or path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)
    return path


def read_dataframe(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])
