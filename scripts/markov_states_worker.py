"""Два состояния годового роста: fit на calibration, затем только filtering."""

import argparse
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("panel"); parser.add_argument("output"); parser.add_argument("--cut", type=int, default=62)
    args = parser.parse_args()
    panel = pd.read_csv(args.panel, index_col="period", parse_dates=True)
    records, models = [], []
    for series in panel.columns:
        growth = np.log(panel[series] / panel[series].shift(12)).dropna() * 100
        train = growth[growth.index <= panel.index[args.cut]]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                model = MarkovRegression(train, k_regimes=2, trend="c", switching_variance=True)
                fitted = model.fit(disp=False, maxiter=300, em_iter=10)
                high = int(fitted.params["const[1]"] > fitted.params["const[0]"])
                filtered = MarkovRegression(growth, k_regimes=2, trend="c", switching_variance=True).filter(fitted.params)
                probability = filtered.filtered_marginal_probabilities[high]
                # Prefix check: будущие наблюдения не должны менять уже filtered posterior.
                prefix = growth.iloc[:-6]
                prefix_p = MarkovRegression(prefix, k_regimes=2, trend="c", switching_variance=True).filter(fitted.params).filtered_marginal_probabilities[high]
                if not np.allclose(probability.loc[prefix.index], prefix_p, atol=1e-10):
                    raise ValueError("Filtering зависит от будущего")
                stable = None; pending = None; consecutive = 0
                for date, value in probability.items():
                    if date <= panel.index[args.cut]: continue
                    state = "faster_growth" if value >= .8 else "slower_growth" if value <= .2 else "uncertain"
                    if state == pending and state != "uncertain": consecutive += 1
                    else: pending, consecutive = state, 1
                    transition = False
                    if state != "uncertain" and consecutive >= 2:
                        transition = stable is not None and stable != state
                        stable = state
                    records.append({"period": date, "series_id": series, "probability_faster_growth": value,
                                    "state": state, "confirmed_state": stable, "transition_confirmed": transition,
                                    "log_yoy_growth": growth.loc[date], "parameters_known_through": panel.index[args.cut]})
                models.append({"series_id": series, "params": fitted.params.to_dict(), "high_state": high,
                               "converged": bool(fitted.mle_retvals.get("converged", False)),
                               "prefix_filtering_check": True, "warnings": [str(w.message) for w in caught]})
            except Exception as exc:
                models.append({"series_id": series, "error": f"{type(exc).__name__}: {exc}"})
    out = Path(args.output)
    pd.DataFrame(records).to_csv(out / "markov_states.csv", index=False)
    (out / "markov_parameters.json").write_text(json.dumps(models, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__": main()
