"""Prophet как baseline на муниципальной панели: по каждому ряду и origin, только прошлое.

Настройки заданы заранее и не подбирались: месячные даты, годовая сезонность 'auto'
(включается Prophet'ом только при истории свыше двух лет, т.е. здесь фактически выключена),
остальное по умолчанию. Это воспроизводимый baseline «из коробки»; Prophet организатора
иначе может быть настроен, настройки неизвестны.
"""

import json
import logging
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.mun_data import load_municipal

MIN_ORIGIN = 5
HORIZONS = 12


def fit_chunk(args):
    from prophet import Prophet
    logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
    logging.getLogger("prophet").setLevel(logging.CRITICAL)
    import cmdstanpy
    cmdstanpy.utils.get_logger().setLevel(logging.CRITICAL)
    series_ids, origin, values, start = args
    ds = pd.date_range(start, periods=origin + 1, freq="MS")
    future = pd.DataFrame({"ds": pd.date_range(ds[-1] + pd.DateOffset(months=1), periods=HORIZONS, freq="MS")})
    rows = []
    for sid, y in zip(series_ids, values):
        model = Prophet(weekly_seasonality=False, daily_seasonality=False)
        try:
            model.fit(pd.DataFrame({"ds": ds, "y": y[:origin + 1]}))
            yhat = model.predict(future).yhat.to_numpy()
            status = "ok"
        except Exception:  # fallback фиксируется в статусе, не скрывается
            yhat = np.full(HORIZONS, y[origin])
            status = "fallback_last_value"
        for h, v in enumerate(yhat, 1):
            rows.append((origin, sid, h, max(0., float(v)), status))
    return rows


def main():
    only = int(sys.argv[sys.argv.index("--only") + 1]) if "--only" in sys.argv else None
    panel, _ = load_municipal(ROOT)
    out_dir = ROOT / "artifacts/municipal_runs/_prophet"
    out_dir.mkdir(parents=True, exist_ok=True)
    start = panel.months[0]
    chunk = 200
    jobs = []
    origins = [only] if only is not None else range(MIN_ORIGIN, panel.n_months)
    for origin in origins:
        for lo in range(0, panel.n_series, chunk):
            jobs.append((np.arange(lo, min(lo + chunk, panel.n_series)), origin,
                         panel.values[lo:lo + chunk], start))
    t = time.time()
    rows = []
    with Pool(10) as pool:
        for k, r in enumerate(pool.imap_unordered(fit_chunk, jobs)):
            rows += r
            if k % 50 == 0:
                print(f"{k}/{len(jobs)} {time.time() - t:.0f}s", flush=True)
    df = pd.DataFrame(rows, columns=["origin_idx", "series_idx", "h", "y_pred", "status"])
    target = out_dir / "prophet_default.parquet"
    if only is not None and target.exists():
        df = pd.concat([pd.read_parquet(target), df], ignore_index=True).drop_duplicates(["origin_idx", "series_idx", "h"], keep="last")
    df.to_parquet(target)
    (out_dir / "prophet_meta.json").write_text(json.dumps({
        "config": "Prophet(weekly_seasonality=False, daily_seasonality=False), остальное по умолчанию",
        "fallbacks": int((df.status != "ok").sum()), "rows": len(df),
        "elapsed_seconds": time.time() - t}, ensure_ascii=False, indent=1))
    print("done", time.time() - t, df.status.value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    main()
