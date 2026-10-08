"""Недельный nowcast месячных расходов России.

Недельные ряды роста г/г идут до 27.09.2026, месячные — до 08.2026. Для целевого месяца M
по уже вышедшим неделям строим оценку г/г месяца и переводим её в уровень:
    уровень_M = уровень_{M-12} * (1 + yoy_hat/100).
Доступность недели консервативна: период + 7 дней (неясно, обозначает ли дата начало или
конец недели). Параметры не подбирались по тесту; поправка bias считается только по месяцам,
завершённым раньше целевого. Сравнение — на тех же 12 тестовых месяцах (09.2025–08.2026),
на которых оценивался прогноз N01. Это nowcast с частичной информацией месяца, а не h=1.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.consumer import read_export
from src.forecasting import load_panel

WEEKLY_MAP = {"Все категории": "Всего", "Продовольственные товары": "Продовольственные товары",
              "Непродовольственные товары": "Непродовольственные товары",
              "Общественное питание": "Общественное питание", "Услуги": "Услуги"}
LAG_DAYS = 7
BIAS_MONTHS = 6
MIN_BIAS_MONTHS = 3


def load_weekly():
    cfg = json.loads((ROOT / "configs/consumer_sources.json").read_text())
    raw, _ = read_export(ROOT / cfg["datasets"]["weekly_growth"]["preferred_file"])
    raw["category"] = raw.category.str.strip()
    raw = raw[raw.category.isin(WEEKLY_MAP)]
    raw["series"] = raw.category.map(WEEKLY_MAP)
    raw["period"] = pd.to_datetime(raw.period)
    raw["available_at"] = raw.period + pd.Timedelta(days=LAG_DAYS)
    return raw[["period", "available_at", "series", "value"]].sort_values(["series", "period"])


def month_estimates(weekly, series, month, k_weeks):
    w = weekly[(weekly.series == series) & (weekly.period.dt.to_period("M") == month.to_period("M"))]
    w = w.sort_values("period")
    if k_weeks is not None:
        w = w.head(k_weeks)
    if w.empty:
        return np.nan, pd.NaT
    return float(w.value.mean()), w.available_at.max()


def nowcast(national, weekly, targets, k_weeks, calibrate):
    rows = []
    for series in national.columns:
        yoy_actual = 100 * (national[series] / national[series].shift(12) - 1)
        for month in targets:
            est, issued = month_estimates(weekly, series, month, k_weeks)
            if not np.isfinite(est):
                continue
            bias = 0.
            if calibrate:
                prev = []
                for back in range(1, BIAS_MONTHS + 1):
                    m = month - pd.DateOffset(months=back)
                    full, _ = month_estimates(weekly, series, m, None)
                    if np.isfinite(full) and np.isfinite(yoy_actual.get(m, np.nan)):
                        prev.append(yoy_actual[m] - full)
                bias = float(np.mean(prev)) if len(prev) >= MIN_BIAS_MONTHS else 0.
            yoy_hat = est + bias
            level = national[series][month - pd.DateOffset(years=1)] * (1 + yoy_hat / 100)
            rows.append({"series_id": series, "target_date": month, "k_weeks": k_weeks or "all",
                         "calibrated": calibrate, "issued_not_before": issued, "yoy_hat": yoy_hat,
                         "y_pred": level, "y_true": national[series][month]})
    return pd.DataFrame(rows)


def main():
    cfg = json.loads((ROOT / "configs/national_forecast.json").read_text())
    national = load_panel(ROOT / cfg["input"])
    weekly = load_weekly()
    run = ROOT / json.loads((ROOT / "artifacts/forecast_runs/latest.json").read_text())["directory"]
    ref = pd.read_csv(run / "predictions.csv", parse_dates=["origin", "target_date"])
    test_targets = sorted(ref[(ref.stage == "holdout")].target_date.unique())
    test_targets = [pd.Timestamp(t) for t in test_targets]
    # Те же месяцы для всех: от них зависит MAE.
    all_targets = [m for m in national.index if m >= pd.Timestamp("2024-01-01")]
    frames = []
    for k in (1, 2, 3, None):
        for calibrate in (False, True):
            frames.append(nowcast(national, weekly, all_targets, k, calibrate))
    est = pd.concat(frames, ignore_index=True)
    est["stage"] = np.where(est.target_date.isin(test_targets), "holdout", "development")
    est["k_weeks"] = est.k_weeks.astype(str)
    # референсы h=1 на тех же месяцах
    h1 = ref[(ref.horizon_months == 1) & (ref.stage == "holdout")]
    refs = h1[h1.model.isin(["DevWeightedEnsemble", "SeasonalGrowth", "LastValue"])]
    keep = ["Всего", "Продовольственные товары", "Непродовольственные товары", "Общественное питание", "Услуги"]
    rows = []
    for (k, cal), part in est[est.stage == "holdout"].groupby(["k_weeks", "calibrated"]):
        rows.append({"method": f"Nowcast(weeks={k},calibrated={cal})", "n": len(part),
                     "mae": float((part.y_true - part.y_pred).abs().mean())})
    for model, part in refs.groupby("model"):
        part = part[part.series_id.isin(keep)]
        rows.append({"method": f"h=1 {model}", "n": len(part), "mae": float((part.y_true - part.y_pred).abs().mean())})
    table = pd.DataFrame(rows)
    out = ROOT / "artifacts/nowcast"
    out.mkdir(parents=True, exist_ok=True)
    est.to_csv(out / "nowcast_predictions.csv", index=False)
    table.to_csv(out / "nowcast_vs_h1_holdout.csv", index=False)
    print(table.to_string(index=False))
    print("последняя неделя:", weekly.period.max().date(), "последний месяц:", national.index.max().date())


if __name__ == "__main__":
    main()
