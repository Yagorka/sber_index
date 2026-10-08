"""Фундаментальные модели на муниципальной панели (zero-shot, без дообучения).

Чекпойнты и ревизии фиксируются в meta.json. Для каждого origin 5..23 и каждого ряда
контекстом служит только history[:origin+1]. Выход: (24, 13, S) медианные прогнозы,
плюс квантили 0.1/0.9 для интервалов.

  python scripts/mun_foundation.py --model chronos2|chronos2_nat|bolt|timesfm [--limit N]

Оговорка о возможном пересечении: обучающие корпуса этих чекпойнтов могут содержать
данные 2023–2024 гг. Проверить это невозможно, поэтому результат — zero-shot оценка
современной модели, а не исторический deployment.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.mun_data import NationalPriors, load_municipal
from src.mun_models import Context

CHECKPOINTS = {"chronos2": "amazon/chronos-2", "chronos2_nat": "amazon/chronos-2",
               "bolt": "amazon/chronos-bolt-small", "timesfm": "google/timesfm-2.5-200m-pytorch"}
MIN_ORIGIN, LAST_ORIGIN = 5, 23
BATCH = {"chronos2": 512, "chronos2_nat": 256, "bolt": 1024, "timesfm": 512}


def revision(repo):
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(repo).sha
    except Exception as exc:  # без сети ревизию узнать нельзя — фиксируем это
        return f"unknown ({exc.__class__.__name__})"


def build(model_name):
    if model_name.startswith("chronos2"):
        from chronos import Chronos2Pipeline
        return Chronos2Pipeline.from_pretrained(CHECKPOINTS[model_name], device_map="cpu")
    if model_name == "bolt":
        from chronos import BaseChronosPipeline
        return BaseChronosPipeline.from_pretrained(CHECKPOINTS[model_name], device_map="cpu",
                                                  torch_dtype=torch.float32)
    import timesfm
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(CHECKPOINTS[model_name])
    m.compile(timesfm.ForecastConfig(max_context=32, max_horizon=12, normalize_inputs=True,
                                     use_continuous_quantile_head=False, force_flip_invariance=True,
                                     infer_is_positive=True, fix_quantile_crossing=True))
    return m


def forecast_batch(model_name, model, panel, ctx, origin, rows):
    contexts = [panel.values[i, :origin + 1] for i in rows]
    if model_name == "chronos2":
        out = model.predict([torch.tensor(c, dtype=torch.float32) for c in contexts], prediction_length=12)
        q = np.stack([o[0].numpy() for o in out])               # (n, n_quantiles, 12)
        levels = list(model.quantiles)
        return q[:, levels.index(0.5)], q[:, levels.index(0.1)], q[:, levels.index(0.9)]
    if model_name == "chronos2_nat":
        # национальный путь (по данным <= origin) как known-future ковариата
        nat_hist = np.array([ctx.nat.log[ctx.nat_idx[t]] for t in range(origin + 1)])    # (o+1, 5)
        cur = ctx.nat.log[ctx.nat_idx[origin]]
        future = np.stack([ctx.nat.path(panel.months[origin], h) + cur for h in range(1, 13)])
        inputs = []
        for i in rows:
            col = ctx.pick[i]
            inputs.append({"target": torch.tensor(panel.values[i, :origin + 1], dtype=torch.float32),
                           "past_covariates": {"nat": torch.tensor(nat_hist[:, col], dtype=torch.float32)},
                           "future_covariates": {"nat": torch.tensor(future[:, col], dtype=torch.float32)}})
        out = model.predict(inputs, prediction_length=12)
        q = np.stack([o[0].numpy() for o in out])
        levels = list(model.quantiles)
        return q[:, levels.index(0.5)], q[:, levels.index(0.1)], q[:, levels.index(0.9)]
    if model_name == "bolt":
        q, mean = model.predict_quantiles([torch.tensor(c, dtype=torch.float32) for c in contexts],
                                          prediction_length=12, quantile_levels=[0.1, 0.5, 0.9])
        q = q.numpy()
        return q[:, :, 1], q[:, :, 0], q[:, :, 2]
    point, quant = model.forecast(horizon=12, inputs=[c.astype(np.float32) for c in contexts])
    return np.asarray(quant[:, :, 5]), np.asarray(quant[:, :, 1]), np.asarray(quant[:, :, 9])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(CHECKPOINTS))
    ap.add_argument("--limit", type=int, default=0, help="только первые N рядов (проверка)")
    ap.add_argument("--sample-mo", type=int, default=0,
                    help="случайные N МО (все 6 категорий), seed 42; для медленных моделей на CPU")
    ap.add_argument("--threads", type=int, default=10)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    cfg = json.loads((ROOT / "configs/national_forecast.json").read_text())
    nat = NationalPriors.from_file(ROOT / cfg["input"])
    panel, _ = load_municipal(ROOT)
    ctx = Context(panel, nat)
    n = args.limit or panel.n_series
    sample_rows = None
    if args.sample_mo:
        terr = np.sort(panel.meta.territory_id.unique())
        chosen = np.random.default_rng(42).choice(terr, args.sample_mo, replace=False)
        sample_rows = np.where(panel.meta.territory_id.isin(chosen).to_numpy())[0]
    out_dir = ROOT / "artifacts/municipal_runs/_foundation"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build(args.model)
    shape = (24, 13, panel.n_series)
    med, lo, hi = (np.full(shape, np.nan, dtype=np.float32) for _ in range(3))
    started = time.time()
    batch = BATCH[args.model]
    for origin in range(MIN_ORIGIN, LAST_ORIGIN + 1):
        all_rows = sample_rows if sample_rows is not None else np.arange(n)
        for start in range(0, len(all_rows), batch):
            rows = [int(r) for r in all_rows[start:start + batch]]
            m, l, h = forecast_batch(args.model, model, panel, ctx, origin, rows)
            med[origin, 1:, rows] = np.maximum(m, 0)
            lo[origin, 1:, rows] = np.maximum(l, 0)
            hi[origin, 1:, rows] = np.maximum(h, 0)
        print(f"{args.model} origin {origin} {time.time() - started:.0f}s", flush=True)
    suffix = f"_limit{n}" if args.limit else (f"_sample{args.sample_mo}" if args.sample_mo else "")
    np.save(out_dir / f"{args.model}{suffix}_median.npy", med)
    np.save(out_dir / f"{args.model}{suffix}_q10.npy", lo)
    np.save(out_dir / f"{args.model}{suffix}_q90.npy", hi)
    (out_dir / f"{args.model}{suffix}_meta.json").write_text(json.dumps({
        "model": args.model, "checkpoint": CHECKPOINTS[args.model],
        "revision": revision(CHECKPOINTS[args.model]), "zero_shot": True, "context": "history[:origin+1], raw rubles",
        "covariates": "national mapped SeasonalGrowth path (known at origin)" if args.model == "chronos2_nat" else None,
        "elapsed_seconds": time.time() - started, "series": int(len(sample_rows)) if sample_rows is not None else n,
        "sample_mo": args.sample_mo or None,
        "torch": torch.__version__}, ensure_ascii=False, indent=1))
    print("done", time.time() - started)


if __name__ == "__main__":
    main()
