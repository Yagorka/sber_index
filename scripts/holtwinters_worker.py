"""Обучить настоящую statsmodels Holt–Winters в отдельном Python-окружении."""

import argparse
import json
import pickle
from pathlib import Path
import warnings

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    jobs = json.loads(Path(args.input).read_text())
    results = []
    for i, job in enumerate(jobs):
        y = np.asarray(job["values"], dtype=float)
        scale = y[-12:].mean()
        fallback = False
        error = None
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            try:
                fitted = ExponentialSmoothing(y / scale, trend="add", seasonal_periods=12,
                                              initialization_method="estimated", **job["params"]).fit(
                                                  optimized=True, use_brute=False)
                forecast = np.asarray(fitted.forecast(12)) * scale
                if not np.isfinite(forecast).all():
                    raise ValueError("Нечисловой прогноз")
                if job.get("save_path"):
                    with Path(job["save_path"]).open("wb") as file:
                        pickle.dump({"fitted": fitted, "scale": scale}, file)
            except Exception as exc:
                fallback = True
                error = f"{type(exc).__name__}: {exc}"
                forecast = y[-12:]
        results.append({"origin_index": job["origin_index"], "series_id": job["series_id"],
                        "variant": job["variant"], "forecast": np.maximum(forecast, 0).tolist(),
                        "fallback": fallback, "error": error, "warnings": [str(w.message) for w in recorded]})
        if (i + 1) % 100 == 0:
            print(f"Holt–Winters: {i + 1}/{len(jobs)}", flush=True)
    Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
