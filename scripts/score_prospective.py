"""Оценка замороженного прогноза 2025 г. по фактам, когда они появятся (честный out-of-time тест).

Ожидается файл data/inputs/municipal_consumption_2025.parquet с теми же колонками, что consumption.parquet
(date 'YYYY-MM', territory_id, category, value). Целевые месяцы и горизонты берутся из прогноза
(origin = 2024-12, horizon = номер месяца 2025). Проверяется хэш замороженного файла.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    frozen = ROOT / "prospective/forecast_2025_frozen.csv"
    expected = (ROOT / "prospective/forecast_2025_frozen.sha256").read_text().split()[0]
    if hashlib.sha256(frozen.read_bytes()).hexdigest() != expected:
        sys.exit("Хэш замороженного прогноза не совпал: файл менялся")
    facts_path = ROOT / "data/inputs/municipal_consumption_2025.parquet"
    if not facts_path.exists():
        sys.exit("Фактов за 2025 г. по МО ещё нет: положите data/inputs/municipal_consumption_2025.parquet")
    facts = pd.read_parquet(facts_path)
    facts["target_month"] = pd.to_datetime(facts.date + "-01")
    full = pd.read_csv(frozen, parse_dates=["target_month"])
    f = full.merge(facts[["territory_id", "category", "target_month", "value"]], on=["territory_id", "category", "target_month"])
    rows = []
    for h in sorted(f.horizon_months.unique()):
        part = f[f.horizon_months == h]
        rows.append({"h": int(h), "n": len(part),
                     "mae_ensemble": float((part.value - part.y_pred_ensemble).abs().mean()),
                     "mae_prophet": float((part.value - part.y_pred_prophet).abs().mean()),
                     "coverage_90": float(((part.value >= part.lo90) & (part.value <= part.hi90)).mean())})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "prospective/score_2025.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
