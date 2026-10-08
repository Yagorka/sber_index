"""Прогноз 2025 г. по МО замораживается; сверка подразумеваемого роста с национальными фактами.

Фактов по МО за 2025 г. в проекте нет. Но национальные месячные расходы за 2025 г. известны
(модель их не использовала: контекст до 12.2024). Поэтому проверяем согласованность:
средний по МО г/г рост прогноза (нижняя граница честности — выборка МО не равна стране) против
национального г/г роста тех же месяцев. Это санити-проверка, а не оценка точности по МО.
Файл прогноза хэшируется; scripts/score_prospective.py посчитает MAE, когда появятся факты МО.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.mun_data import NationalPriors, load_municipal
from src.frozen_artifacts import preserve_or_create

PAIR = {"Все категории": "Всего", "Продовольствие": "Продовольственные товары",
        "Общественное питание": "Общественное питание"}


def main():
    run = ROOT / json.loads((ROOT / "artifacts/municipal_runs/latest.json").read_text())["directory"]
    f = pd.read_csv(run / "forecast_2025.csv", parse_dates=["target_month"])
    panel, _ = load_municipal(ROOT)
    nat = NationalPriors.from_file(ROOT / json.loads((ROOT / "configs/national_forecast.json").read_text())["input"])
    val24 = pd.DataFrame(panel.values[:, 12:], columns=pd.date_range("2024-01-01", periods=12, freq="MS"))
    val24["territory_id"], val24["category"] = panel.meta.territory_id.to_numpy(), panel.meta.category.to_numpy()
    rows = []
    for cat, ncol in PAIR.items():
        base = val24[val24.category == cat].set_index("territory_id").drop(columns="category")
        for model in ["y_pred_ensemble", "y_pred_prophet", "y_pred_structural"]:
            part = f[f.category == cat].pivot(index="territory_id", columns="target_month", values=model)
            for m in range(12):
                month25 = pd.Timestamp(2025, m + 1, 1)
                month24 = pd.Timestamp(2024, m + 1, 1)
                yoy_mo = 100 * (part[month25] / base[month24] - 1)
                nat_yoy = 100 * (nat.panel[ncol][month25] / nat.panel[ncol][month24] - 1)
                rows.append({"category": cat, "model": model.replace("y_pred_", ""), "month": month25,
                             "mean_mo_yoy_pct": float(yoy_mo.mean()), "median_mo_yoy_pct": float(yoy_mo.median()),
                             "national_yoy_pct": float(nat_yoy)})
    df = pd.DataFrame(rows)
    df["abs_gap_pp"] = (df.mean_mo_yoy_pct - df.national_yoy_pct).abs()
    out = ROOT / "artifacts/check_2025"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "yoy_vs_national.csv", index=False)
    summary = df.groupby(["category", "model"]).abs_gap_pp.mean().unstack()
    summary.to_csv(out / "summary_gap_pp.csv")
    print(summary.round(2).to_string())
    # заморозка
    frozen = ROOT / "prospective"
    frozen.mkdir(exist_ok=True)
    target = frozen / "forecast_2025_frozen.csv"
    keep = ["territory_id", "category", "target_month", "horizon_months", "y_pred_ensemble", "y_pred_prophet", "lo90", "hi90"]
    created,different,digest = preserve_or_create(target,f[keep].to_csv(index=False),run / "forecast_2025_candidate.csv")
    if not created:
        print("Исходный frozen-файл сохранён; новый вариант отличается:",different)
        if different:
            print("Новый вариант:",run / "forecast_2025_candidate.csv")
        return
    (frozen / "forecast_2025_frozen.sha256").write_text(f"{digest}  forecast_2025_frozen.csv\n")
    (frozen / "README.md").write_text(
        "# Замороженный прогноз 2025\n\nФайл `forecast_2025_frozen.csv` построен по данным до 2024-12 (origin = 2024-12), "
        f"run `{run.name}`, SHA256 `{digest}`.\n\nКогда СберИндекс опубликует муниципальные расходы за 2025 г., "
        "положите файл в `data/inputs/municipal_consumption_2025.parquet` и запустите `python scripts/score_prospective.py` — "
        "скрипт посчитает MAE по горизонтам и сравнение с Prophet из того же запуска.\n"
        "Для независимой временной метки добавьте файл в git-коммит/релиз до публикации новых данных.\n")
    print("заморожено", target, digest[:16])


if __name__ == "__main__":
    main()
