"""Оценка замороженных национальных прогнозов на 09.2026, когда СберИндекс опубликует месячный файл."""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.forecasting import load_panel


def main():
    frozen = ROOT / "prospective/national_2026-09_frozen.csv"
    if hashlib.sha256(frozen.read_bytes()).hexdigest() != (ROOT / "prospective/national_2026-09_frozen.sha256").read_text().split()[0]:
        sys.exit("Хэш не совпал: файл менялся после заморозки")
    cfg = json.loads((ROOT / "configs/national_forecast.json").read_text())
    national = load_panel(ROOT / cfg["input"])   # положите обновлённый consumer-spending*.csv.zip и укажите его в config
    target = pd.Timestamp("2026-09-01")
    if target not in national.index:
        sys.exit("В месячном файле ещё нет 2026-09: обновите consumer-spending*.csv.zip и configs/consumer_sources.json")
    df = pd.read_csv(frozen)
    df["y_true"] = df.series_id.map(national.loc[target])
    cols = ["nowcast_weekly_calibrated", "nowcast_weekly_uncalibrated", "h1_ensemble_N01"]
    res = pd.DataFrame({c: [(df.y_true - df[c]).abs().mean()] for c in cols})
    res.to_csv(ROOT / "prospective/score_national_2026-09.csv", index=False)
    print(res.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
