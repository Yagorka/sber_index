"""Абляция «новостных» признаков (решения ЦБ и реестр региональных событий) + плацебо.

Все варианты используют одну быструю конфигурацию HGB и одну сетку, чтобы разница
относилась только к признакам. Плацебо: реестр переносится на случайные МО и месяцы
(5 повторов), ставка — со случайным сдвигом назад на 3..12 мес. (остаётся прошлым).
Если реальный вариант не лучше распределения плацебо, пользы новостей нет.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.mun_data import NationalPriors, load_municipal
from src.mun_eval import make_grid, score_model
from src.mun_models import (Context, apply_drift, fit_hgb, predict_drift, training_set)
from src.mun_news import NewsFeatures

FIRST, LAST = 5, 23
QUICK = {"max_iter": 80, "learning_rate": 0.1, "max_depth": 4, "min_samples_leaf": 200, "l2_regularization": 1.0}


def run_variant(ctx, wp, seed, shrink, base3):
    drift = np.full((24, 13, ctx.panel.n_series), np.nan, dtype=np.float32)
    for origin in range(FIRST, LAST + 1):
        train = training_set(ctx, origin, wp, max_rows=150000, seed=seed)
        model = fit_hgb(train, QUICK, seed) if train is not None and len(train[1]) >= 3000 else None
        drift[origin] = predict_drift(ctx, origin, model)
    return apply_drift(base3, drift, shrink)


def main():
    started = time.time()
    latest = json.loads((ROOT / "artifacts/municipal_runs/latest.json").read_text())
    frozen = json.loads((ROOT / latest["directory"] / "frozen_selection.json").read_text())
    wp, shrink = frozen["structural"]["weight_power"], frozen["structural"]["shrink"]
    cfg = json.loads((ROOT / "configs/municipal.json").read_text())
    nat = NationalPriors.from_file(ROOT / json.loads((ROOT / "configs/national_forecast.json").read_text())["input"])
    panel, _ = load_municipal(ROOT)
    grid = make_grid()
    rate = pd.read_csv(ROOT / "data/external/cbr_key_rate_daily.csv", parse_dates=["date"])
    registry = pd.read_csv(ROOT / "data/external/event_registry.csv", parse_dates=["event_date"])
    base3 = np.load(ROOT / "artifacts/municipal_runs/_cache/NatPath_K3.npy")
    rows = []

    def evaluate(label, extra_provider, kind):
        ctx = Context(panel, nat)
        ctx.extra = extra_provider
        pred = run_variant(ctx, wp, cfg["seed"], shrink, base3)
        for stage in ["validation", "test"]:
            m = score_model(panel, pred, grid, stage, label)
            for r in m.itertuples():
                rows.append({"variant": label, "kind": kind, "stage": stage, "h": r.h, "mae": r.mae})
        print(f"  {label} {time.time() - started:.0f}s", flush=True)

    class Sub:   # обёртка: только выбранные колонки признаков
        def __init__(self, nf, cols):
            self.nf, self.cols = nf, cols
        def __call__(self, a):
            return self.nf(a)[:, self.cols]

    nf = NewsFeatures(panel, rate, registry)
    evaluate("base_no_news", None, "base")
    evaluate("+key_rate", Sub(nf, [0, 1, 2, 3]), "real")
    evaluate("+regional_events", Sub(nf, [4, 5]), "real")
    evaluate("+key_rate+events", Sub(nf, list(range(6))), "real")
    for seed in range(5):
        rng = np.random.default_rng(100 + seed)
        lag = int(rng.integers(3, 13))
        shifted = rate.copy()
        shifted["date"] = shifted.date + pd.DateOffset(months=lag)
        nfp = NewsFeatures(panel, shifted, registry, placebo_seed=100 + seed)
        evaluate(f"placebo_rate+events_{seed}", Sub(nfp, list(range(6))), "placebo")
        evaluate(f"placebo_events_{seed}", Sub(nfp, [4, 5]), "placebo_events")
    df = pd.DataFrame(rows)
    out = ROOT / "artifacts/news_ablation"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "ablation_runs.csv", index=False)
    piv = df.pivot_table(index=["kind", "variant"], columns=["stage", "h"], values="mae")
    piv.to_csv(out / "ablation_pivot.csv")
    base = df[df.variant == "base_no_news"].set_index(["stage", "h"]).mae
    df["gain_vs_base_pct"] = 100 * (df.set_index(["stage", "h"]).index.map(base) - df.mae) / df.set_index(["stage", "h"]).index.map(base)
    summ = df.groupby(["kind", "stage", "h"]).gain_vs_base_pct.agg(["mean", "min", "max"]).reset_index()
    summ.to_csv(out / "ablation_summary.csv", index=False)
    print(summ.round(2).to_string())


if __name__ == "__main__":
    main()
