"""Графики для отчёта и презентации из сохранённых результатов (ничего не пересчитывает).

Читает последние запуски: artifacts/municipal_runs/latest.json, artifacts/shocks,
artifacts/weekly_shocks, artifacts/nowcast, artifacts/news_ablation.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.mun_data import CATEGORIES, NationalPriors, load_municipal

OUT = ROOT / "artifacts/figures"
OUT.mkdir(parents=True, exist_ok=True)
PAL = {"Prophet": "#8a8d91", "Ensemble": "#1a6fb0", "StructHGB": "#2a9d8f", "NatPath_K3": "#e9a23b",
       "SeasonalNaive_NatGrowth": "#c4572f", "Chronos2": "#7b5ea7", "Chronos2_NatCov": "#a66bbe",
       "TimesFM": "#b5b36a", "ChronosBolt": "#6e8fa6", "LastValue": "#bbbbbb", "LocalOnlyHGB": "#d17a9b"}
H = [1, 3, 6, 12]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130, "savefig.bbox": "tight"})


def latest_run():
    p = json.loads((ROOT / "artifacts/municipal_runs/latest.json").read_text())
    return ROOT / p["directory"]


def fig_mae(run):
    t = pd.read_csv(run / "metrics_test.csv")
    models = [m for m in ["Prophet", "SeasonalNaive_NatGrowth", "NatPath_K3", "StructHGB", "Chronos2", "Chronos2_NatCov",
                          "TimesFM", "Ensemble"] if m in set(t.model)]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    w = 0.8 / len(models)
    for k, m in enumerate(models):
        v = [t[(t.model == m) & (t.h == h)].mae.iloc[0] for h in H]
        ax.bar(np.arange(4) + k * w, v, w, label=m, color=PAL.get(m))
    ax.set_xticks(np.arange(4) + 0.4 - w / 2)
    ax.set_xticklabels([f"{h} мес." for h in H])
    ax.set_ylabel("MAE, руб./жителя в месяц")
    ax.set_title("Тест (июль–декабрь 2024): MAE по горизонтам, 12 096 рядов")
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(OUT / "mae_by_model_test.png")
    plt.close(fig)


def fig_gain(run):
    b = pd.read_csv(run / "bootstrap_vs_prophet_test.csv")
    t = pd.read_csv(run / "metrics_test.csv")
    base = t[t.model == "Prophet"].set_index("h").mae
    fig, ax = plt.subplots(figsize=(8, 4))
    models = [m for m in ["Ensemble", "StructHGB", "NatPath_K3", "Chronos2_NatCov", "Chronos2", "TimesFM"] if m in set(b.model)]
    w = 0.8 / len(models)
    for k, m in enumerate(models):
        part = b[b.model == m].set_index("h").loc[H]
        gain = -100 * part.mae_diff_vs_prophet / base.loc[H].to_numpy()
        lo = -100 * part.mo_ci_high / base.loc[H].to_numpy()
        hi = -100 * part.mo_ci_low / base.loc[H].to_numpy()
        ax.bar(np.arange(4) + k * w, gain, w, yerr=[gain - lo, hi - gain], capsize=2, color=PAL.get(m), label=m)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(np.arange(4) + 0.4 - w / 2)
    ax.set_xticklabels([f"{h} мес." for h in H])
    ax.set_ylabel("Снижение MAE относительно Prophet, %")
    ax.set_title("Выигрыш у Prophet (95% bootstrap по МО)")
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(OUT / "gain_vs_prophet.png")
    plt.close(fig)


def fig_weights(run):
    w = pd.read_csv(run / "ensemble_weights.csv").pivot(index="model", columns="h", values="weight").fillna(0)
    w = w.loc[w.sum(1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bottom = np.zeros(len(w.columns))
    for m in w.index:
        ax.bar([f"{h} мес." for h in w.columns], w.loc[m], bottom=bottom, label=m, color=PAL.get(m))
        bottom += w.loc[m].to_numpy()
    ax.set_ylabel("Вес в ансамбле")
    ax.set_title("Веса ансамбля (выбраны по validation)")
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.savefig(OUT / "ensemble_weights.png")
    plt.close(fig)


def fig_groups(run):
    g = pd.read_csv(run / "metrics_by_group_test.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, grp in zip(axes, ["category", "size_tercile"]):
        part = g[g.group == grp]
        vals = list(part.value.unique())
        for k, m in enumerate(["Prophet", "Ensemble"]):
            v = [part[(part.model == m) & (part.value == x)].mae.mean() for x in vals]
            ax.bar(np.arange(len(vals)) + k * 0.4, v, 0.4, label=m, color=PAL[m])
        ax.set_xticks(np.arange(len(vals)) + 0.2)
        ax.set_xticklabels(vals, rotation=25 if grp == "category" else 0, ha="right" if grp == "category" else "center")
        ax.set_title("MAE по категориям (среднее по 4 горизонтам)" if grp == "category" else "MAE по размеру ряда (терцили уровня)")
    axes[0].legend()
    fig.savefig(OUT / "mae_by_group.png")
    plt.close(fig)


def fig_detectors():
    inj = pd.read_csv(ROOT / "artifacts/shocks/injection_recall_by_magnitude.csv")
    summ = pd.read_csv(ROOT / "artifacts/shocks/injection_runs.csv").groupby("method").mean(numeric_only=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for m, c in zip(["CUSUM", "PageHinkley", "BOCPD"], ["#c4572f", "#2a9d8f", "#1a6fb0"]):
        p = inj[inj.method == m]
        axes[0].plot(p.magnitude, p.recall, marker="o", label=m, color=c)
    axes[0].set_xlabel("Величина сдвига, σ")
    axes[0].set_ylabel("Recall (в пределах 3 мес.)")
    axes[0].set_title("Инъекция сдвигов в реальные остатки (МО)")
    axes[0].legend()
    x = np.arange(3)
    axes[1].bar(x - 0.2, summ.loc[["CUSUM", "PageHinkley", "BOCPD"]].event_f1, 0.4, label="event F1", color="#1a6fb0")
    ax2 = axes[1].twinx()
    ax2.bar(x + 0.2, summ.loc[["CUSUM", "PageHinkley", "BOCPD"]].false_alarms_per_100_series_months, 0.4,
            label="ложные тревоги / 100 ряд-мес.", color="#e9a23b")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["CUSUM", "PageHinkley", "BOCPD"])
    axes[1].set_title("F1 и ложные тревоги (5 повторов, бюджет 1/100)")
    axes[1].legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(False)
    fig.savefig(OUT / "detectors_injection.png")
    plt.close(fig)


def fig_weekly():
    m = pd.read_csv(ROOT / "artifacts/weekly_shocks/detector_metrics_proxy_labels.csv")
    m = m[m.panel == "weekly_growth"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for method, c in zip(["CUSUM", "PageHinkley", "BOCPD"], ["#c4572f", "#2a9d8f", "#1a6fb0"]):
        p = m[m.method == method].sort_values("penalty_multiplier")
        axes[0].plot(p.penalty_multiplier, p.event_f1, marker="o", color=c, label=method)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("Штраф сегментации (больше = меньше меток)")
    axes[0].set_ylabel("event F1 (proxy-метки)")
    axes[0].set_title("Недельные ряды: F1 и разметка", fontsize=10)
    axes[0].legend()
    e = pd.read_csv(ROOT / "artifacts/weekly_shocks/early_warning_metrics.csv")
    e = e[(e.status == "ok") & (e.panel == "weekly_growth")]
    labels = [f"k={int(r.k_weeks)} {r.model}" for r in e.itertuples()]
    axes[1].bar(np.arange(len(e)) - 0.2, e.pr_auc, 0.4, label="PR-AUC", color="#1a6fb0")
    axes[1].bar(np.arange(len(e)) + 0.2, e.pr_auc_baseline_prevalence, 0.4, label="базовая (доля событий)", color="#bbbbbb")
    axes[1].set_xticks(np.arange(len(e)))
    axes[1].set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    axes[1].set_title("Раннее предупреждение: PR-AUC", fontsize=10)
    axes[1].legend(fontsize=8)
    fig.savefig(OUT / "weekly_detectors_early_warning.png")
    plt.close(fig)


def fig_examples(run):
    """Три реальных примера: событие (Орск), типичное срабатывание, ложная тревога."""
    panel, _ = load_municipal(ROOT)
    z = np.load(ROOT / "artifacts/shocks/z.npy")
    chosen = json.loads((ROOT / "artifacts/shocks/chosen_thresholds.json").read_text())
    alarms = {m: np.load(ROOT / f"artifacts/shocks/alarms_{m}.npy") for m in chosen}
    meta = panel.meta
    months = panel.months
    det = pd.read_csv(ROOT / "artifacts/shocks/event_registry_detection.csv") if (ROOT / "artifacts/shocks/event_registry_detection.csv").exists() else pd.DataFrame()
    picks = []
    orsk = meta[(meta.mo_name == "Орск") & (meta.category == "Все категории")]
    ev_month = pd.Timestamp("2024-04-01")
    if len(orsk):
        i = int(orsk.index[0])
        hit = any(alarms[m][i, 15:19].any() for m in alarms)
        picks.append((i, "Реальное событие: паводок в Орске 5.04.2024 — " +
                      ("тревога в окне 3 мес." if hit else "ни один детектор не сработал в окне 3 мес.")))
    first = alarms["BOCPD"]
    false_alarm = sustained = None
    for i in np.argsort(-np.nanmax(np.abs(z[:, 18:21]), 1)):
        for t in np.where(first[i, 18:21])[0] + 18:
            nxt = z[i, t + 1:t + 4]
            if len(nxt) == 3 and np.isfinite(nxt).all():
                if false_alarm is None and abs(z[i, t]) > 3 and np.abs(nxt).mean() < 0.8:
                    false_alarm = (int(i), int(t))
                if sustained is None and abs(z[i, t]) > 3 and np.all(np.sign(nxt) == np.sign(z[i, t])) and np.abs(nxt).mean() > 1.5:
                    sustained = (int(i), int(t))
        if false_alarm and sustained:
            break
    if false_alarm:
        picks.append((false_alarm[0], f"Ложная тревога BOCPD ({months[false_alarm[1]]:%m.%Y}): всплеск без продолжения"))
    if sustained:
        picks.append((sustained[0], f"Срабатывание BOCPD ({months[sustained[1]]:%m.%Y}): отклонение сохраняется"))
    fig, axes = plt.subplots(len(picks), 1, figsize=(9, 3.2 * len(picks)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (i, title) in zip(axes, picks):
        ax.plot(months, panel.values[i], color="#1a6fb0", marker="o", ms=3)
        ax.set_title(f"{title}\n{meta.mo_name[i]} · {meta.category[i]}", fontsize=9, loc="left")
        ax.set_ylabel("руб./жителя")
        ax2 = ax.twinx()
        ax2.plot(months, z[i], color="#999", lw=0.8)
        ax2.set_ylabel("z остатка", color="#999")
        ax2.grid(False)
        colors = {"CUSUM": "#c4572f", "PageHinkley": "#2a9d8f", "BOCPD": "#e9a23b"}
        for k, (m, a) in enumerate(alarms.items()):
            for t in np.where(a[i])[0]:
                ax.axvline(months[t], color=colors[m], ls="--", lw=1.2, label=m if t == np.where(a[i])[0][0] else None)
        ax.legend(fontsize=7, loc="upper left")
    if len(orsk):
        axes[0].axvline(ev_month, color="k", ls=":", lw=1.2)
        axes[0].text(ev_month, axes[0].get_ylim()[1], " паводок", fontsize=8, va="top")
    fig.savefig(OUT / "detector_examples.png")
    plt.close(fig)


def fig_news():
    p = ROOT / "artifacts/news_ablation/ablation_summary.csv"
    if not p.exists():
        return
    s = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, stage in zip(axes, ["validation", "test"]):
        part = s[s.stage == stage]
        kinds = [k for k in ["real", "placebo", "placebo_events"] if k in set(part.kind)]
        for k, kind in enumerate(kinds):
            q = part[part.kind == kind].sort_values("h")
            ax.errorbar(np.arange(len(q)) + k * 0.15, q["mean"], yerr=[q["mean"] - q["min"], q["max"] - q["mean"]],
                        fmt="o", capsize=3, label={"real": "реальные признаки (3 варианта)", "placebo": "плацебо: ставка+события",
                                                   "placebo_events": "плацебо: события"}[kind])
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(4))
        ax.set_xticklabels([f"{h} мес." for h in H])
        ax.set_title(stage)
    axes[0].set_ylabel("Изменение MAE к базе без новостей, % (+ лучше)")
    axes[1].legend(fontsize=7)
    fig.savefig(OUT / "news_ablation.png")
    plt.close(fig)


def fig_nowcast():
    t = pd.read_csv(ROOT / "artifacts/nowcast/nowcast_vs_h1_holdout.csv")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    cols = ["#bbb" if m.startswith("h=1") else ("#2a9d8f" if "True" in m else "#e9a23b") for m in t.method]
    ax.barh(t.method, t.mae, color=cols)
    ax.set_xlabel("MAE, млрд руб. (12 тестовых месяцев × 5 категорий)")
    ax.set_title("Недельный nowcast против h=1 прогнозов (национальный уровень)")
    ax.invert_yaxis()
    fig.savefig(OUT / "nowcast.png")
    plt.close(fig)


def fig_check_2025(run):
    f = pd.read_csv(run / "forecast_2025.csv", parse_dates=["target_month"])
    panel, _ = load_municipal(ROOT)
    nat_cfg = json.loads((ROOT / "configs/national_forecast.json").read_text())
    nat = NationalPriors.from_file(ROOT / nat_cfg["input"])
    pair = {"Все категории": "Всего", "Продовольствие": "Продовольственные товары", "Общественное питание": "Общественное питание"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    months24 = panel.months
    for ax, (cat, ncol) in zip(axes, pair.items()):
        idx = np.where((panel.meta.category == cat).to_numpy())[0]
        hist = panel.values[idx].mean(0)
        pred = f[f.category == cat].groupby("target_month").y_pred_ensemble.mean()
        prophet = f[f.category == cat].groupby("target_month").y_pred_prophet.mean()
        ax.plot(months24, hist, color="#1a6fb0", label="факт МО (среднее по МО)")
        ax.plot(pred.index, pred.values, color="#c4572f", label="прогноз 2025, ансамбль")
        ax.plot(prophet.index, prophet.values, color="#8a8d91", ls="--", label="Prophet")
        nat_series = nat.panel[ncol]
        scale = hist[:12].mean() / nat_series["2023-01-01":"2023-12-01"].mean()
        nat_part = nat_series["2023-01-01":"2025-12-01"] * scale
        ax.plot(nat_part.index, nat_part.values, color="#2a9d8f", lw=1, label="национальный факт (масштаб 2023)")
        ax.set_title(cat, fontsize=10)
        ax.tick_params(axis="x", rotation=30)
    axes[0].legend(fontsize=7)
    fig.suptitle("Проверка согласованности: прогноз 2025 против национального факта (уровня МО за 2025 нет)", fontsize=10)
    fig.savefig(OUT / "forecast_2025_check.png")
    plt.close(fig)


def fig_map(run):
    try:
        import geopandas as gpd
    except ImportError:
        return
    g = gpd.read_file(ROOT / "t_dict_municipal/t_dict_municipal_districts_poly.gpkg")
    panel, _ = load_municipal(ROOT)
    from src.mun_eval import make_grid, pair_errors
    grid = make_grid()
    ens = np.load(run / "pred_Ensemble.npy")
    prop = np.load(run / "pred_Prophet.npy")
    gr, y, pe, _ = pair_errors(panel, ens, grid, "test")
    _, _, pp, _ = pair_errors(panel, prop, grid, "test")
    ae, ap = np.abs(y - pe).mean(0), np.abs(y - pp).mean(0)
    sel = (panel.meta.category == "Все категории").to_numpy()
    df = pd.DataFrame({"territory_id": panel.meta.territory_id[sel], "gain": 100 * (ap[sel] - ae[sel]) / ap[sel]})
    g["territory_id"] = pd.to_numeric(g.territory_id)
    df["territory_id"] = df.territory_id.astype(int)
    g = g.merge(df, on="territory_id", how="left")
    g = g.to_crs(epsg=3576) if g.crs else g
    fig, ax = plt.subplots(figsize=(10, 6))
    g.plot(column="gain", cmap="RdBu", vmin=-60, vmax=60, ax=ax, linewidth=0, missing_kwds={"color": "#e8e8e8"}, legend=True,
           legend_kwds={"label": "Снижение MAE ансамбля относительно Prophet, % (тест, «Все категории»)", "shrink": 0.6})
    ax.set_axis_off()
    fig.savefig(OUT / "map_gain_vs_prophet.png")
    plt.close(fig)


def main():
    run = latest_run()
    for f in (fig_mae, fig_gain, fig_weights, fig_groups, fig_check_2025, fig_map, fig_examples):
        try:
            f(run) if f.__code__.co_argcount else f()
            print("ok", f.__name__)
        except Exception as exc:
            print("FAILED", f.__name__, repr(exc)[:200])
    for f in (fig_detectors, fig_weekly, fig_news, fig_nowcast):
        try:
            f()
            print("ok", f.__name__)
        except Exception as exc:
            print("FAILED", f.__name__, repr(exc)[:200])


if __name__ == "__main__":
    main()
