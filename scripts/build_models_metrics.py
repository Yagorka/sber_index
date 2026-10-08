"""Собирает docs/MODELS_AND_METRICS_RU.md: что обучено и какие метрики получены (из файлов результатов)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
H = [1, 3, 6, 12]


def md(df, fmt=".1f"):
    return df.to_markdown(index=False, floatfmt=fmt, missingval="—")


def rd(p):
    p = ROOT / p
    return pd.read_csv(p) if p.exists() else None


def main():
    run = ROOT / json.loads((ROOT / "artifacts/municipal_runs/latest.json").read_text())["directory"]
    man = json.loads((run / "manifest.json").read_text())
    fro = json.loads((run / "frozen_selection.json").read_text())
    cfg = json.loads((ROOT / "configs/municipal.json").read_text())
    pm = json.loads((ROOT / "artifacts/municipal_runs/_prophet/prophet_meta.json").read_text())
    fm = {}
    for key in ["bolt", "chronos2", "chronos2_nat", "timesfm_sample200"]:
        p = ROOT / f"artifacts/municipal_runs/_foundation/{key}_meta.json"
        if p.exists():
            fm[key] = json.loads(p.read_text())
    out = []
    out.append(f"""# Что обучено и какие метрики получены

Документ собран автоматически (`scripts/build_models_metrics.py`) из файлов результатов. Муниципальный запуск: `{run.name}`; хэш данных `{man['data_sha256'][:16]}…`, конфигурации `{man['config_sha256'][:16]}…`. Единицы муниципальных метрик — рубли на жителя в месяц; национальных — млрд руб. Протокол: validation = цели янв.–июнь 2024, test = июль–декабрь 2024 (допущение, правила организатора неизвестны).

## 1. Муниципальный блок: что обучено

| Модель | Тип | Что обучалось / считалось | Параметры | Где |
|---|---|---|---|---|
| LastValue, SeasonalNaive, SeasonalNaive_NatGrowth, NatPath_K1, NatPath_K3, LocalBase_K3 | правила, без обучения | прогноз по формуле для каждого origin 5…23 | K — число месяцев в оценке уровня | `src/mun_models.py` |
| Prophet | статистическая | {pm['rows'] // 12:,} отдельных подгонок (12 096 рядов × 19 origin), fallback: {pm['fallbacks']} | месячные даты, остальное по умолчанию; годовая сезонность `auto` | `scripts/mun_prophet.py`, `_prophet/` |
| StructHGB | градиентный бустинг | остаток относительно NatPath_K3; по одной модели на origin 6…23 для каждого из 3 вариантов веса уровня (0 / 0.5 / 1) | `HistGradientBoostingRegressor(loss=absolute_error, max_iter={cfg['structural']['hgb_params']['max_iter']}, learning_rate={cfg['structural']['hgb_params']['learning_rate']}, max_depth={cfg['structural']['hgb_params']['max_depth']}, min_samples_leaf={cfg['structural']['hgb_params']['min_samples_leaf']}, l2={cfg['structural']['hgb_params']['l2_regularization']})`, до {cfg['structural']['max_train_rows']:,} строк, 27 признаков. Выбрано на validation: вес уровня {fro['structural']['weight_power']}, сжатие {fro['structural']['shrink']} | `_cache/StructDrift_*.npy` |
| LocalOnlyHGB | бустинг без национальных данных | абляция: те же признаки без национальных приоров | как StructHGB | `_cache/LocalOnlyDrift_*.npy` |
| ChronosBolt | фундаментальная, zero-shot | 12 096 рядов × 19 origin, без дообучения | `{fm['bolt']['checkpoint']}`, ревизия `{fm['bolt']['revision'][:12]}` | `_foundation/bolt_*` |
| Chronos2 | фундаментальная, zero-shot | 12 096 × 19 | `{fm['chronos2']['checkpoint']}`, ревизия `{fm['chronos2']['revision'][:12]}` | `_foundation/chronos2_*` |
| Chronos2_NatCov | фундаментальная + ковариата | то же, национальный путь как known-future ковариата | та же ревизия | `_foundation/chronos2_nat_*` |
| TimesFM | фундаментальная, zero-shot | только 200 случайных МО (1 200 рядов) × 19 origin | `{fm['timesfm_sample200']['checkpoint']}`, ревизия `{fm['timesfm_sample200']['revision'][:12]}` | `_foundation/timesfm_sample200_*` |
| Ensemble | взвешенная смесь | веса — линейная программа (минимум MAE на validation, симплекс), отдельно по горизонтам; h=12 использует пары h=6 и h=12 | кандидаты: {', '.join(fro['ensemble_candidates'])} | `frozen_selection.json`, `ensemble_weights.csv` |
| Интервалы | split-conformal | относительные ошибки на validation, по (горизонт, категория) | покрытие 0.9 | `intervals_test.csv` |

Веса ансамбля (h=1 / 3 / 6 / 12): """)
    w = pd.read_csv(run / "ensemble_weights.csv").pivot(index="model", columns="h", values="weight").fillna(0)
    w = w[w.sum(axis=1) > 0.001].reset_index().rename(columns={"model": "Модель"})
    out.append("\n" + md(w, ".2f"))

    t = pd.read_csv(run / "metrics_test.csv")
    v = pd.read_csv(run / "metrics_validation.csv")
    out.append("\n## 2. Муниципальные метрики\n\n### Test (июль–декабрь 2024), MAE и дополнительные метрики\n")
    mae = t.pivot(index="model", columns="h", values="mae")[H]
    mae["среднее"] = mae.mean(axis=1)
    r2c = t.pivot(index="model", columns="h", values="r2_change")[H]
    r2 = t.pivot(index="model", columns="h", values="r2")[H]
    wape = t.pivot(index="model", columns="h", values="wape")[H]
    tab = pd.DataFrame({"Модель": mae.index})
    for h in H:
        tab[f"MAE h={h}"] = mae[h].to_numpy()
    tab["MAE среднее"] = mae["среднее"].to_numpy()
    base = mae.loc["Prophet", "среднее"]
    tab["Δ к Prophet, %"] = (100 * (base - mae["среднее"]) / base).to_numpy()
    for h in H:
        tab[f"R² изм. h={h}"] = r2c[h].to_numpy()
    tab = tab.sort_values("MAE среднее")
    out.append(md(tab, ".2f").replace(".00 ", " "))
    out.append("\nR² по изменению — доля объяснённой дисперсии изменения относительно origin; обычный R² по уровню 0.97–0.996 у всех моделей и малоинформативен (см. `metrics_test.csv`: колонки `r2`, `wape`, `coverage`). Сетка у всех моделей одна, покрытие 100%.")
    mv = v.pivot(index="model", columns="h", values="mae")[H]
    mv["среднее"] = mv.mean(axis=1)
    mv = mv.sort_values("среднее").reset_index().rename(columns={"model": "Модель"})
    out.append("\n### Validation (январь–июнь 2024, используется для выбора), MAE\n\n" + md(mv.round(1)))
    b = pd.read_csv(run / "bootstrap_vs_prophet_test.csv")
    be = b[b.model == "Ensemble"][["h", "mae_diff_vs_prophet", "mo_ci_low", "mo_ci_high", "month_ci_low", "month_ci_high"]].rename(
        columns={"h": "h", "mae_diff_vs_prophet": "Δ MAE (ансамбль − Prophet)", "mo_ci_low": "95% по МО, ниж.", "mo_ci_high": "95% по МО, верх.",
                 "month_ci_low": "95% по месяцам, ниж.", "month_ci_high": "95% по месяцам, верх."})
    out.append("\n### Доверительные интервалы разницы с Prophet (test)\n\n" + md(be))
    iv = pd.read_csv(run / "intervals_test.csv")
    out.append("\n### Интервалы 90%\n\n" + md(iv.assign(h=iv.h.astype(int)), ".3f"))
    sub = rd(str(run.relative_to(ROOT) / "subset200_timesfm_comparison.csv"))
    if sub is not None:
        s = sub[sub.stage == "test"].pivot(index="model", columns="h", values="mae")[H]
        s["среднее"] = s.mean(axis=1)
        out.append("\n### TimesFM против остальных на одной подвыборке (200 МО, test), MAE\n\n" + md(s.sort_values("среднее").reset_index().rename(columns={"model": "Модель"})))

    # --- детекторы
    out.append("\n## 3. Детекторы и раннее предупреждение\n")
    thr = json.loads((ROOT / "artifacts/shocks/chosen_thresholds.json").read_text())
    inj = pd.read_csv(ROOT / "artifacts/shocks/injection_runs.csv").groupby("method").mean(numeric_only=True)
    order = ["CUSUM", "PageHinkley", "BOCPD"]
    d = pd.DataFrame({"Метод": order, "Порог": [thr[m] for m in order], "Precision": [inj.loc[m, "event_precision"] for m in order],
                      "Recall": [inj.loc[m, "event_recall"] for m in order], "F1": [inj.loc[m, "event_f1"] for m in order],
                      "Задержка, мес.": [inj.loc[m, "median_delay_months"] for m in order],
                      "Ложные тревоги/100 ряд-мес.": [inj.loc[m, "false_alarms_per_100_series_months"] for m in order]})
    out.append("Остатки МО, инъекция сдвигов в 5% рядов (5 повторов), бюджет ложных тревог на калибровке 1/100:\n\n" + md(d, ".3f"))
    wk = rd("artifacts/weekly_shocks/detector_metrics_proxy_labels.csv")
    if wk is not None:
        wk = wk[(wk.panel == "weekly_growth") & (wk.penalty_multiplier == 16)][["method", "threshold", "n_proxy_events_test", "event_precision", "event_recall", "event_f1", "false_alarms_per_100_series_weeks", "median_delay_weeks"]]
        out.append("\nНедельные реальные ряды (46 категорий), proxy-метки offline-сегментации (штраф 16):\n\n" + md(wk, ".3f"))
    ew = rd("artifacts/weekly_shocks/early_warning_metrics.csv")
    if ew is not None:
        e = ew[(ew.status == "ok") & (ew.panel == "weekly_growth")][["k_weeks", "model", "test_prevalence", "pr_auc", "brier", "brier_baseline_prevalence", "row_precision", "row_recall"]]
        out.append("\nРаннее предупреждение (HGB и логистическая регрессия, обучены на недельных признаках; навык не подтверждён):\n\n" + md(e, ".3f"))

    # --- новости, nowcast
    news = rd("artifacts/news_ablation/ablation_summary.csv")
    if news is not None:
        n = news[news.stage == "test"].pivot_table(index="kind", columns="h", values="mean").reset_index().rename(columns={"kind": "Вариант"})
        out.append("\n## 4. Новостные признаки и nowcast\n\nАблация (быстрый HGB, 14 обучений: база, 3 реальных варианта, 10 плацебо), изменение MAE к базе на test, % (+ лучше):\n\n" + md(n, ".2f"))
    nc = rd("artifacts/nowcast/nowcast_vs_h1_holdout.csv")
    if nc is not None:
        out.append("\nНедельный nowcast (без обучения параметров; поправка смещения по прошлым месяцам), национальный holdout, MAE млрд руб.:\n\n" + md(nc.rename(columns={"method": "Метод"}), ".1f"))

    # --- национальный блок
    ncfg = json.loads((ROOT / "configs/national_forecast.json").read_text())
    nrun = ROOT / json.loads((ROOT / "artifacts/forecast_runs/latest.json").read_text())["directory"]
    nm = pd.read_csv(nrun / "metrics.csv")
    nh = nm[nm.stage == "holdout"].pivot(index="model", columns="horizon_months", values="macro_mae")
    nh["среднее"] = nh.mean(axis=1)
    nh = nh.sort_values("среднее").reset_index().rename(columns={"model": "Модель"})
    fam = pd.DataFrame([{"id": x["id"], "семейство": x["family"], "параметры": json.dumps(x["params"], ensure_ascii=False)} for x in ncfg["variants"]])
    out.append(f"\n## 5. Национальный блок (N01, run `{nrun.name}`)\n\nОбучено {len(ncfg['variants'])} вариантов моделей 8 семейств плюс ансамбль из трёх семейств; данные — национальные расходы, 5 категорий, 93 месяца; holdout сентябрь 2025 – август 2026. Метрика — MAE млрд руб. (среднее по 5 категориям).\n\n" + md(nh, ".2f") + "\n\nВарианты и параметры:\n\n" + md(fam))
    ad = rd("artifacts/adaptive_runs/adaptive_20261003T125544_858704Z/ranking.csv")
    if ad is not None:
        ad = ad.sort_values("development_validation").head(6)
        out.append("\nАдаптивная ветка N02 (25 моделей; holdout уже был просмотрен, улучшение не заявляется), 6 лучших по validation, MAE млрд руб.:\n\n" + md(ad.rename(columns={"model": "Модель"}), ".1f"))
    out.append("""
## 6. Замороженные прогнозы

- `prospective/forecast_2025_frozen.csv` — прогноз 2025 по МО, ансамбль и Prophet, 90% интервалы (SHA256 в `.sha256`);
- `prospective/national_2026-09_frozen.csv` — национальные h=1 (N01) и недельный nowcast на сентябрь 2026.

## 7. Что не обучалось и не оценено

Дообучение фундаментальных моделей; TimesFM на полной панели; Росстат по МО; текстовые новости; детекторы на верифицированных шоках.
""")
    (ROOT / "docs/MODELS_AND_METRICS_RU.md").write_text("\n".join(out))
    print("готово", len("\n".join(out)))


if __name__ == "__main__":
    main()
