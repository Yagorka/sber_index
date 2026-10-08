"""Замораживает национальные прогнозы на сентябрь 2026 г. до выхода фактов.

Содержит: (а) прогноз h=1 из N01 (DevWeightedEnsemble, origin 08.2026), (б) недельный nowcast
по неделям сентября, вышедшим к дате заморозки (по 27.09.2026), в двух вариантах. Файл хэшируется;
после выхода месячного файла СберИндекса выполните scripts/score_prospective_national.py.
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.nowcast_weekly import load_weekly, month_estimates, BIAS_MONTHS, MIN_BIAS_MONTHS
from src.forecasting import load_panel


def main():
    cfg = json.loads((ROOT / "configs/national_forecast.json").read_text())
    national = load_panel(ROOT / cfg["input"])
    weekly = load_weekly()
    target = pd.Timestamp("2026-09-01")
    run = ROOT / json.loads((ROOT / "artifacts/forecast_runs/latest.json").read_text())["directory"]
    fut = pd.read_csv(run / "future_forecasts.csv")
    rows = []
    for series in national.columns:
        yoy_actual = 100 * (national[series] / national[series].shift(12) - 1)
        full_est, _ = month_estimates(weekly, series, target, None)
        if pd.isna(full_est):
            continue
        prev = []
        for back in range(1, BIAS_MONTHS + 1):
            m = target - pd.DateOffset(months=back)
            f, _ = month_estimates(weekly, series, m, None)
            if pd.notna(f) and pd.notna(yoy_actual.get(m)):
                prev.append(yoy_actual[m] - f)
        bias = sum(prev) / len(prev) if len(prev) >= MIN_BIAS_MONTHS else 0.
        base = national[series][target - pd.DateOffset(years=1)]
        rows.append({"series_id": series, "target_month": target.date(),
                     "nowcast_weekly_calibrated": base * (1 + (full_est + bias) / 100),
                     "nowcast_weekly_uncalibrated": base * (1 + full_est / 100),
                     "weeks_used": int(((weekly.series == series) & (weekly.period.dt.to_period("M") == target.to_period("M"))).sum())})
    df = pd.DataFrame(rows)
    h1 = fut[(fut.horizon_months == 1)] if "horizon_months" in fut else fut
    cols = [c for c in h1.columns]
    name_col = "series_id" if "series_id" in cols else cols[0]
    val_col = "y_pred" if "y_pred" in cols else cols[-1]
    ens = h1[h1.model == "DevWeightedEnsemble"] if "model" in cols else h1
    df = df.merge(ens[[name_col, val_col]].rename(columns={name_col: "series_id", val_col: "h1_ensemble_N01"}), on="series_id", how="left")
    out = ROOT / "prospective"
    out.mkdir(exist_ok=True)
    path = out / "national_2026-09_frozen.csv"
    df.to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / "national_2026-09_frozen.sha256").write_text(f"{digest}  national_2026-09_frozen.csv\n")
    (out / "national_2026-09_frozen.meta.json").write_text(json.dumps({
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "last_weekly_period": str(weekly.period.max().date()),
        "last_monthly_period": str(national.index.max().date()), "national_run": run.name, "sha256": digest}, indent=1))
    print(df.round(1).to_string(index=False)); print(digest[:16])


if __name__ == "__main__":
    main()
