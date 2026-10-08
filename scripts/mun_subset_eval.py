"""Сравнение TimesFM (считался только на случайных 200 МО = 1200 рядов) с остальными на той же подвыборке."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.mun_data import load_municipal
from src.mun_eval import HORIZONS, make_grid, pair_errors


def main():
    run = ROOT / json.loads((ROOT / "artifacts/municipal_runs/latest.json").read_text())["directory"]
    panel, _ = load_municipal(ROOT)
    grid = make_grid()
    path = ROOT / "artifacts/municipal_runs/_foundation/timesfm_sample200_median.npy"
    if not path.exists():
        sys.exit("TimesFM ещё не посчитан")
    tfm = np.load(path)
    rows = np.where(np.isfinite(tfm[:, 1:, :]).any(axis=(0, 1)))[0]
    models = {"TimesFM": tfm}
    for name in ["Prophet", "Ensemble", "NatPath_K3", "SeasonalNaive_NatGrowth", "Chronos2", "Chronos2_NatCov", "LastValue"]:
        p = run / f"pred_{name}.npy"
        if p.exists():
            models[name] = np.load(p)
    out = []
    for stage in ["validation", "test"]:
        for name, arr in models.items():
            g, y, p, _ = pair_errors(panel, arr, grid, stage)
            for h in HORIZONS:
                m = (g.h == h).to_numpy()
                err = np.abs(y[m][:, rows] - p[m][:, rows])
                out.append({"stage": stage, "model": name, "h": h, "mae": float(np.nanmean(err)),
                            "n_series": len(rows), "coverage": float(np.isfinite(p[m][:, rows]).mean())})
    df = pd.DataFrame(out)
    df.to_csv(run / "subset200_timesfm_comparison.csv", index=False)
    pivot = df[df.stage == "test"].pivot(index="model", columns="h", values="mae")
    pivot["mean_h"] = pivot.mean(axis=1)
    print(pivot.sort_values("mean_h").round(1).to_string())


if __name__ == "__main__":
    main()
