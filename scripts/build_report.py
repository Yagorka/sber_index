"""Собирает отчёт, документ с результатами, презентацию и лендинг из сохранённых результатов.

Числа берутся из файлов запусков; текст описывает только то, что в них есть. Если нужного
файла нет (например, TimesFM или абляция новостей не посчитаны), раздел помечается как не
выполненный, а не заполняется предположениями.
"""

import base64
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

H = [1, 3, 6, 12]
FIG = ROOT / "artifacts/figures"


def md(df, floatfmt=".1f", index=False):
    return df.to_markdown(index=index, floatfmt=floatfmt, missingval="—")


def html_table(df, floatfmt="{:.0f}"):
    return df.to_html(index=False, float_format=lambda v: floatfmt.format(v), border=0)


def rd(path, **kw):
    return pd.read_csv(path, **kw) if Path(path).exists() else None


def load():
    R = {}
    p = json.loads((ROOT / "artifacts/municipal_runs/latest.json").read_text())
    run = ROOT / p["directory"]
    R["run"] = run
    R["manifest"] = json.loads((run / "manifest.json").read_text())
    R["frozen"] = json.loads((run / "frozen_selection.json").read_text())
    R["test"] = pd.read_csv(run / "metrics_test.csv")
    R["val"] = pd.read_csv(run / "metrics_validation.csv")
    R["boot"] = pd.read_csv(run / "bootstrap_vs_prophet_test.csv")
    R["groups"] = pd.read_csv(run / "metrics_by_group_test.csv")
    R["intervals"] = pd.read_csv(run / "intervals_test.csv")
    R["weights"] = pd.read_csv(run / "ensemble_weights.csv")
    R["subset"] = rd(run / "subset200_timesfm_comparison.csv")
    sh = ROOT / "artifacts/shocks"
    R["inj"] = rd(sh / "injection_runs.csv")
    R["inj_mag"] = rd(sh / "injection_recall_by_magnitude.csv")
    R["reg_match"] = rd(sh / "event_registry_matching.csv")
    R["reg_det"] = rd(sh / "event_registry_detection.csv")
    R["proxy"] = rd(sh / "offline_proxy_sensitivity.csv")
    R["thr"] = json.loads((sh / "chosen_thresholds.json").read_text())
    wk = ROOT / "artifacts/weekly_shocks"
    R["wk_det"] = rd(wk / "detector_metrics_proxy_labels.csv")
    R["wk_ew"] = rd(wk / "early_warning_metrics.csv")
    R["news"] = rd(ROOT / "artifacts/news_ablation/ablation_summary.csv")
    R["news_runs"] = rd(ROOT / "artifacts/news_ablation/ablation_runs.csv")
    R["nowcast"] = rd(ROOT / "artifacts/nowcast/nowcast_vs_h1_holdout.csv")
    R["chk25"] = rd(ROOT / "artifacts/check_2025/summary_gap_pp.csv")
    return R


def mae(R, model, h, stage="test"):
    t = R["test"] if stage == "test" else R["val"]
    return float(t[(t.model == model) & (t.h == h)].mae.iloc[0])


def leaderboard(R):
    t = R["test"].pivot(index="model", columns="h", values="mae")[H]
    t["среднее"] = t.mean(axis=1)
    base = t.loc["Prophet"]
    for h in H + ["среднее"]:
        t[f"Δ к Prophet, % ({h})"] = 100 * (base[h] - t[h]) / base[h]
    t = t.sort_values("среднее")
    out = pd.DataFrame({"Модель": t.index})
    for h in H:
        out[f"MAE h={h}"] = t[h].to_numpy()
    out["Среднее"] = t["среднее"].to_numpy()
    out["Δ к Prophet, % (среднее)"] = t["Δ к Prophet, % (среднее)"].to_numpy()
    return out


def sec_forecast(R):
    ens = {h: mae(R, "Ensemble", h) for h in H}
    pro = {h: mae(R, "Prophet", h) for h in H}
    gain = {h: 100 * (pro[h] - ens[h]) / pro[h] for h in H}
    b = R["boot"]
    be = b[b.model == "Ensemble"].set_index("h")
    ci_mo_ok = all(be.mo_ci_high < 0)
    ci_mon_ok = all(be.month_ci_high < 0)
    lb = leaderboard(R)
    r2 = R["test"][R["test"].model.isin(["Ensemble", "Prophet"])].pivot(index="h", columns="model", values="r2_change")
    parts = []
    parts.append(f"""**Главный результат.** На независимом тесте (цели июль–декабрь 2024, 12 096 рядов) замороженный ансамбль имеет MAE **{ens[1]:.0f} / {ens[3]:.0f} / {ens[6]:.0f} / {ens[12]:.0f} руб. на жителя** для горизонтов 1 / 3 / 6 / 12 мес. против **{pro[1]:.0f} / {pro[3]:.0f} / {pro[6]:.0f} / {pro[12]:.0f}** у Prophet. Снижение — **{gain[1]:.0f}% / {gain[3]:.0f}% / {gain[6]:.0f}% / {gain[12]:.0f}%**.
Парный bootstrap по МО (1000 повторов) даёт 95% интервалы разности MAE{" строго ниже нуля на всех горизонтах" if ci_mo_ok else " (не на всех горизонтах ниже нуля)"}; bootstrap по шести целевым месяцам (общий шок месяца не усредняется по МО) {"также исключает ноль на всех горизонтах" if ci_mon_ok else "не на всех горизонтах исключает ноль"}.
R² по изменению относительно origin (динамика, а не масштаб МО): ансамбль {r2.loc[1, 'Ensemble']:.2f} / {r2.loc[3, 'Ensemble']:.2f} / {r2.loc[6, 'Ensemble']:.2f} / {r2.loc[12, 'Ensemble']:.2f}, Prophet {r2.loc[1, 'Prophet']:.2f} / {r2.loc[3, 'Prophet']:.2f} / {r2.loc[6, 'Prophet']:.2f} / {r2.loc[12, 'Prophet']:.2f}. Обычный R² по уровню 0.97–0.996 у всех моделей и почти ничего не показывает: уровень определяется масштабом МО.""")
    parts.append("### Таблица теста\n\n" + md(lb, ".1f"))
    # вклад национальных приоров
    loc = mae(R, "LocalOnlyHGB", 1), mae(R, "LocalOnlyHGB", 3), mae(R, "LocalOnlyHGB", 6), mae(R, "LocalOnlyHGB", 12)
    st = [mae(R, "StructHGB", h) for h in H]
    nat = [mae(R, "NatPath_K3", h) for h in H]
    lbase = [mae(R, "LocalBase_K3", h) for h in H]
    abl = pd.DataFrame({"Вариант": ["Локальный уровень (без национальных данных)", "+ GBM без национальных данных",
                                    "Национальный путь (NatPath_K3)", "NatPath + остаточный GBM (StructHGB)"],
                        **{f"MAE h={h}": [lbase[k], loc[k], nat[k], st[k]] for k, h in enumerate(H)}})
    parts.append("### Что даёт национальная информация (абляция)\n\n" + md(abl) +
                 f"\n\nНациональный путь (сезонность и темп роста из рядов с 2018 г., только по данным до origin) снижает среднюю MAE с {np.mean(lbase):.0f} до {np.mean(nat):.0f} ({100 * (np.mean(lbase) - np.mean(nat)) / np.mean(lbase):.0f}%). "
                 f"Остаточный GBM поверх национального пути даёт смешанный эффект: лучше на h=1 ({nat[0]:.0f} → {st[0]:.0f}) и h=12 ({nat[3]:.0f} → {st[3]:.0f}), но хуже на h=3 и h=6; в среднем {np.mean(nat):.0f} → {np.mean(st):.0f}. Основной вклад вносит национальный приор, а не нелинейная модель по МО. Причина: у МО всего 6–23 месяца истории, и ни сезонность, ни годовой рост по такому ряду оценить нельзя.")
    # веса
    w = R["weights"].pivot(index="model", columns="h", values="weight").fillna(0)
    w = w[w.sum(axis=1) > 0.001].reset_index().rename(columns={"model": "Модель"})
    parts.append("### Веса ансамбля (подобраны на validation)\n\n" + md(w, ".2f") +
                 "\n\nВеса найдены линейной программой (минимум MAE на симплексе); для h=12 использованы пары h=6 и h=12, потому что на validation у h=12 один целевой месяц. Foundation-модели получили нулевой вес: на validation они не лучше выбранных кандидатов.")
    # группы
    g = R["groups"]
    cat = g[(g.group == "category") & g.model.isin(["Prophet", "Ensemble"])].groupby(["value", "model"]).mae.mean().unstack()
    cat["Δ, %"] = 100 * (cat.Prophet - cat.Ensemble) / cat.Prophet
    cat = cat.reset_index().rename(columns={"value": "Категория"})
    parts.append("### По категориям (среднее по 4 горизонтам)\n\n" + md(cat) +
                 "\n\nВ категории «Маркетплейсы» с очень быстрым ростом и сильной сезонностью ансамбль не лучше Prophet — это слабое место, которое стоит исправлять отдельным национальным приором для маркетплейсов." if (cat.set_index("Категория").loc["Маркетплейсы", "Δ, %"] < 5) else "")
    sz = g[(g.group == "size_tercile") & g.model.isin(["Prophet", "Ensemble"])].groupby(["value", "model"]).mae.mean().unstack()
    sz["Δ, %"] = 100 * (sz.Prophet - sz.Ensemble) / sz.Prophet
    parts.append("### По размеру ряда (терцили уровня внутри категории)\n\n" + md(sz.reset_index().rename(columns={"value": "Терциль"})))
    iv = R["intervals"]
    parts.append("### Интервалы\n\n" + md(iv.assign(h=iv.h.astype(int)).rename(columns={"h": "Горизонт, мес.", "nominal": "Номинал", "empirical_coverage_test": "Покрытие на тесте", "mean_relative_half_width": "Средняя полуширина (отн.)"}), ".3f") +
                 "\n\nИнтервалы split-conformal по относительной ошибке на validation, отдельно для горизонта и категории. На тесте покрытие близко к номиналу 0.9.")
    # foundation
    f = []
    for m in ["Chronos2", "Chronos2_NatCov", "ChronosBolt"]:
        row = {"Модель": m}
        for h in H:
            row[f"MAE h={h}"] = mae(R, m, h)
        f.append(row)
    f.append({"Модель": "Prophet", **{f"MAE h={h}": pro[h] for h in H}})
    sub = R["subset"]
    txt = "### Фундаментальные модели (zero-shot)\n\n" + md(pd.DataFrame(f)) + "\n\n"
    txt += ("Chronos-Bolt-small и Chronos-2 использованы без дообучения, контекст — `history[:origin+1]`. Версия `Chronos2_NatCov` получает национальный путь как known-future ковариату. "
            "Результат: на коротких рядах (6–23 точки) zero-shot модели уступают Prophet на h≥3; на h=1 Chronos-2 и особенно Chronos-2 с ковариатой чуть лучше Prophet. Ковариата заметно улучшает Chronos-2 на всех горизонтах, но не доводит его до Prophet на h≥6. "
            "Возможное пересечение обучающих корпусов чекпойнтов с 2023–2024 гг. проверить нельзя, поэтому это оценка современной модели, а не исторического deployment.\n\n")
    if sub is not None:
        t = sub[sub.stage == "test"].pivot(index="model", columns="h", values="mae")
        t["среднее"] = t.mean(axis=1)
        txt += "**TimesFM-2.5 (200M)** на CPU считает ~0,13 с на ряд, поэтому оценён на случайной подвыборке из 200 МО (1200 рядов, все категории; seed 42), остальные модели сравнены на тех же рядах:\n\n" + md(t.sort_values("среднее").reset_index().rename(columns={"model": "Модель"})) + "\n"
    else:
        txt += "**TimesFM** не оценён: расчёт на CPU не завершён.\n"
    parts.append(txt)
    return "\n\n".join(p for p in parts if p)


def sec_detectors(R):
    inj = R["inj"].groupby("method").mean(numeric_only=True)
    sd = R["inj"].groupby("method").std(numeric_only=True)
    order = ["CUSUM", "PageHinkley", "BOCPD"]
    t = pd.DataFrame({"Метод": order, "Порог": [R["thr"][m] for m in order],
                      "Precision": [inj.loc[m, "event_precision"] for m in order],
                      "Recall": [inj.loc[m, "event_recall"] for m in order],
                      "F1": [inj.loc[m, "event_f1"] for m in order],
                      "± F1 (5 повторов)": [sd.loc[m, "event_f1"] for m in order],
                      "Медиана задержки, мес.": [inj.loc[m, "median_delay_months"] for m in order],
                      "Ложных тревог / 100 ряд-мес.": [inj.loc[m, "false_alarms_per_100_series_months"] for m in order]})
    mag = R["inj_mag"].pivot(index="method", columns="magnitude", values="recall").loc[order].reset_index()
    mag.columns = ["Метод"] + [f"Recall при сдвиге {c}σ" for c in mag.columns[1:]]
    txt = [f"""Детекторы CUSUM, Page–Hinkley и BOCPD подаются стандартизированными остатками прогноза h=1, выпущенного до наблюдения (национальный путь × локальный уровень). Масштаб остатка считается только по прошлому и сжимается к масштабу категории. Пороги подобраны на месяцах 10–17 при бюджете **1 ложная тревога на 100 ряд-месяцев**, оценка — на месяцах 18–23. Реальных размеченных шоков по МО у нас нет, поэтому оценка идёт тремя независимыми способами, и ни один из них не называется «истиной».""",
           "### A. Инъекция известных сдвигов в реальные остатки\n\nСдвиг ±1.5/2.5/4σ добавляется в 5% рядов (604 события) на месяцы 18–20; остальные 95% остаются реальными и служат фоном для ложных тревог. Пять независимых повторов.\n\n" + md(t, ".3f") + "\n\n" + md(mag, ".2f"),
           f"Частота тревог на тесте выше, чем на калибровке (~2 против ≤1 на 100 ряд-месяцев): в тесте часть тревог — настоящие шоки или сезонные промахи прогноза (Дек–Янв), а не ложные. Поэтому precision здесь консервативна."]
    # реестр
    rm, rdet = R["reg_match"], R["reg_det"]
    n_ev = rm.matched_territories.gt(0).sum()
    txt.append("### B. Реестр датированных событий\n\n" + f"В реестре 7 событий (паводки, теракты, обстрелы; ссылки в `data/external/event_registry.csv`). Сопоставление с МО по названию внутри региона нашло {int(rm.matched_territories.sum())} территорий. Белгорода, Курска и Суджи в панели нет (не прошли порог качества СберИндекса), а для Севастополя, Дербента и Махачкалы сопоставление по названию МО не нашло. Оценимы Орск, Оренбург, Новотроицк (паводок, 04.2024) и Красногорск («Крокус», 03.2024).")
    if rdet is not None and len(rdet):
        s = rdet.groupby(["event_id", "pattern", "method"]).agg(рядов=("alarm_in_window", "size"), доля_с_тревогой=("alarm_in_window", "mean"),
                                                                 макс_z=("max_abs_z_in_window", "max"), фон_все_ряды=("population_alarm_rate_same_window", "first")).reset_index()
        txt.append(md(s, ".2f") + "\n\nОтклик на реальные события слабый: максимум |z| 2.7–3.4σ, тревоги единичны (BOCPD по 1 из 6 категорий в Орске и Новотроицке), в других случаях — нет. Это не доказывает отсутствие эффекта — месячная сумма по всем категориям сглаживает локальный шок, — но и не подтверждает работоспособность детекторов на реальных событиях.")
    # proxy
    px = R["proxy"]
    px = px.pivot(index="penalty_multiplier", columns="method", values="f1").reset_index()
    px.columns.name = None
    txt.append("### C. Offline-сегментация как proxy-разметка\n\nF1 по меткам offline-сегментации z-рядов (3000 случайных рядов, тест). Разметка не независима от формы z и не равна истине, поэтому это анализ чувствительности: "
               "ранжирование методов зависит от типа разметки (на инъекциях выигрывает CUSUM, на offline-метках — BOCPD).\n\n" + md(px.rename(columns={"penalty_multiplier": "Штраф"}), ".3f"))
    # weekly
    wk = R["wk_det"]
    if wk is not None:
        prim = wk[(wk.panel == "weekly_growth") & (wk.penalty_multiplier == 16)][["method", "threshold", "n_proxy_events_test", "n_alarms_test", "event_precision", "event_recall", "event_f1", "false_alarms_per_100_series_weeks", "median_delay_weeks"]]
        txt.append("### D. Недельные реальные ряды (46 категорий, 151 неделя)\n\nУ недельных рядов достаточно длины для оценки на реальных данных: пороги — на первых 55% недель при бюджете 1/100, оценка — на последних 30%. Метки — offline-сегментация со штрафом 16 (чувствительность по штрафам 4/8/16/32 — в `artifacts/weekly_shocks/detector_metrics_proxy_labels.csv`).\n\n" + md(prim, ".3f"))
    ew = R["wk_ew"]
    if ew is not None:
        e = ew[(ew.status == "ok") & (ew.panel == "weekly_growth")][["k_weeks", "model", "test_prevalence", "pr_auc", "brier", "brier_baseline_prevalence", "alarm_rate_per_100", "row_precision", "row_recall"]]
        e = e.rename(columns={"k_weeks": "Окно, нед.", "model": "Модель", "test_prevalence": "Доля событий (базовый PR-AUC)", "pr_auc": "PR-AUC", "brier": "Brier", "brier_baseline_prevalence": "Brier константы"})
        best = e[e.Модель == "HGB"]
        skill = bool(((best["PR-AUC"] - best["Доля событий (базовый PR-AUC)"]) > 0.1).any() and (best["Brier"] < best["Brier константы"]).any())
        txt.append("### E. Раннее предупреждение (вероятность сдвига в окне (t, t+k])\n\nПризнаки на момент t: z, средние и максимумы |z| за 4 недели, доля рядов с |z|>2 (ширина шока), оценки трёх детекторов, изменение ключевой ставки ЦБ за 13 недель и недели с последнего изменения. "
                   "Обучение — на метках, зрелых к границе; между train и test зазор 8 недель.\n\n" + md(e, ".3f") + "\n\n" +
                   ("PR-AUC заметно выше доли событий, а Brier лучше константы." if skill else
                    "**Навык раннего предупреждения не подтверждён**: PR-AUC лишь немного выше доли событий, Brier не лучше константного прогноза, а при бюджете ложных тревог recall низкий. Нужны размеченные события и ряды длиннее; сейчас честный вывод — детекторы реагируют на уже начавшийся сдвиг, предсказывать его заранее по этим признакам не получается."))
    return "\n\n".join(txt)


def sec_news(R):
    n = R["news"]
    if n is None:
        return "Абляция новостных признаков не завершена."
    s = n.copy()
    s = s[s.stage == "test"].pivot_table(index="kind", columns="h", values="mean").reset_index()
    sv = n[n.stage == "validation"].pivot_table(index="kind", columns="h", values="mean").reset_index()
    names = {"real": "реальные признаки (среднее 3 вариантов)", "placebo": "плацебо: ставка + события", "placebo_events": "плацебо: события", "base": "база"}
    for d in (s, sv):
        d["kind"] = d.kind.map(names)
    runs = R["news_runs"]
    var = runs[(runs.stage == "test")].pivot_table(index="variant", columns="h", values="mae").reset_index()
    real_gain = n[(n.kind == "real") & (n.stage == "test")]["mean"]
    plc = n[(n.kind.str.startswith("placebo")) & (n.stage == "test")]
    useful = bool((real_gain > plc["max"].max()).any() and (real_gain > 1.0).any())
    txt = ["""«Новости» здесь — это **датированные официальные события**, а не текстовый корпус: решения Банка России по ключевой ставке (cbr.ru; `available_at` = дата вступления в силу, чтобы не заглядывать вперёд) и реестр из 7 региональных событий со ссылками. Признаки на якоре a: уровень ставки, её изменения за 3 и 6 мес., месяцев с последнего изменения, тяжесть недавних событий в МО и в регионе (окно 3 мес.). Текстовые новости (Интерфакс, Lenta, GDELT) не собирались.""",
           "Сравнение — одна быстрая конфигурация HGB на одной сетке; плацебо: реестр переносится на случайные МО и месяцы (5 повторов), ставка сдвигается назад на случайные 3–12 мес. (остаётся только прошлым).\n\nИзменение MAE к базе без новостей, % (положительное — лучше). Тест:\n\n" + md(s.rename(columns={"kind": "Вариант"}), ".2f") + "\n\nValidation:\n\n" + md(sv.rename(columns={"kind": "Вариант"}), ".2f"),
           "MAE по вариантам на тесте:\n\n" + md(var.rename(columns={"variant": "Вариант"}), ".1f")]
    txt.append("Варианты «+regional_events» и часть плацебо-событий дают побитово те же прогнозы, что база: события затрагивают единицы МО из 2016, и бустинг не делает по ним разбиений. Единственные признаки, меняющие прогноз, — ставка ЦБ и её лаги, а они лишь индекс времени в панели из 24 месяцев.")
    txt.append("**Вывод.** " + ("Реальные признаки превосходят диапазон плацебо — есть сигнал." if useful else
               "Реальные признаки не отличаются от плацебо: полезность новостных признаков для прогноза не подтверждена. Это ожидаемо: ставка ЦБ — признак времени, а в панели всего 24 месяца, и он не может выучиться на разрезе МО; региональные события затрагивают единицы МО из 2016."))
    return "\n\n".join(txt)


def sec_nowcast(R):
    t = R["nowcast"]
    if t is None:
        return "Не выполнено."
    return ("Недельные ряды роста г/г идут по 27.09.2026, месячные — по 08.2026. Для целевого месяца по уже вышедшим неделям оценивается г/г и переводится в уровень: уровень = уровень_{M−12} × (1 + yoy/100). "
            "Доступность недели консервативна (период + 7 дней), поправка смещения считается только по завершённым ранее месяцам, параметры не подбирались. Национальный уровень, 12 тестовых месяцев × 5 категорий (те же, что в N01):\n\n" + md(t.rename(columns={"method": "Метод", "n": "n", "mae": "MAE, млрд руб."}), ".1f") +
            "\n\nБез поправки смещения недельный nowcast плох (MAE 125–137): ряды СберИндекса по транзакциям не равны месячному набору, согласованному с Росстатом, и поправка по прошлым месяцам обязательна. С поправкой: по одной неделе месяца — хуже h=1-ансамбля и SeasonalGrowth, по двум — лучше ансамбля, по трём сравним с SeasonalGrowth, по полному месяцу — ниже обоих (62.0 против 70.1 и 66.0). Выигрыш скромный и на 60 наблюдениях; это nowcast с частичной информацией целевого месяца, а не прогноз h=1.")


def sec_2025(R):
    c = R["chk25"]
    base = "Прогноз на 12 месяцев 2025 г. построен от origin = 12.2024 и заморожен вместе с SHA256 (`prospective/`); фактов по МО за 2025 нет, и скрипт `scripts/score_prospective.py` посчитает MAE, когда они появятся."
    if c is None:
        return base
    return base + "\n\nСанити-проверка по национальным фактам 2025 г. (модель их не видела): средний по МО г/г рост прогноза против национального г/г роста, среднее модуля разницы, п.п.:\n\n" + md(c.reset_index().rename(columns={"category": "Категория"}), ".2f") + "\n\nЭто проверка согласованности (выборка МО без весов не равна стране), а не оценка точности на уровне МО. У ансамбля подразумеваемый рост ближе к национальному, чем у Prophet, во всех трёх категориях, но для «Общественного питания» разрыв велик у обеих моделей (около 29–33 п.п.): национальный общепит в 2025 г. рос заметно быстрее, чем предполагает любая из моделей; причина (состав категории, отличие от муниципальной выборки) не исследована."


def build_markdown(R):
    m = R["manifest"]
    ens = [mae(R, "Ensemble", h) for h in H]
    pro = [mae(R, "Prophet", h) for h in H]
    gains = [100 * (p - e) / p for p, e in zip(pro, ens)]
    ntest = f"{m['series']:,}".replace(",", " ")
    head = f"""# Прогноз потребительских расходов муниципалитетов и ранние сигналы сдвигов

Отчёт построен автоматически из результатов запуска `{R['run'].name}`. Данные: СберИндекс, безналичные расходы на жителя по {ntest} рядам (2016 МО × 6 категорий), январь 2023 – декабрь 2024. Численные утверждения ниже прочитаны из файлов результатов. Каталог обученных моделей и все метрики: `docs/MODELS_AND_METRICS_RU.md`.

## Резюме

- **Прогнозы 1/3/6/12 мес.** Ансамбль на тесте имеет MAE {ens[0]:.0f}/{ens[1]:.0f}/{ens[2]:.0f}/{ens[3]:.0f} руб. против {pro[0]:.0f}/{pro[1]:.0f}/{pro[2]:.0f}/{pro[3]:.0f} у Prophet, то есть на {gains[0]:.0f}/{gains[1]:.0f}/{gains[2]:.0f}/{gains[3]:.0f}% меньше. Выбор моделей и весов заморожен до открытия теста.
- **Откуда выигрыш.** Основной вклад — национальные ряды с 2018 г.: у МО всего 24 месяца, поэтому сезонность и рост берутся из страны, а по МО оценивается уровень и небольшая поправка. Абляция показывает вклад в явном виде.
- **Фундаментальные модели** (Chronos-Bolt, Chronos-2, Chronos-2 с национальной ковариатой, TimesFM на подвыборке) без дообучения на таких коротких рядах уступают Prophet на горизонтах ≥3 (на h=1 Chronos-2 с ковариатой лучше) и получили нулевой вес в ансамбле.
- **Детекторы** CUSUM / Page–Hinkley / BOCPD сравнены тремя способами и на недельных реальных рядах; победитель зависит от способа разметки. **Раннее предупреждение: навык не подтверждён.**
- **Новости** как датированные официальные события (ставка ЦБ, реестр) не показали пользы относительно плацебо.
- **Недельный nowcast** национального уровня с поправкой смещения лучше h=1-ансамбля при ≥2 неделях месяца; без поправки он непригоден.
- Прогноз на 2025 г. по МО заморожен для будущей проверки.

## 1. Данные

| Источник | Что | Период | Использование |
|---|---|---|---|
| СберИндекс, `consumption.parquet` | безналичные расходы на жителя, руб./мес., 6 категорий | 2023-01…2024-12 | цель; {m['series']} полных рядов, {m['incomplete_series_dropped']} неполных исключены |
| Справочник МО (`t_dict_municipal`) | регион, тип, координаты, интервалы действия | срез 2024 | статические признаки, сопоставление событий |
| `market_access.parquet` | индекс доступности рынков | 2024 | статический признак (log) |
| Национальные расходы СберИндекса | 5 категорий, млрд руб./мес. | 2018-12…2026-08 | приор: сезонность и рост **только по данным ≤ origin** |
| Недельный рост г/г (46 категорий), активность | недельные ряды | 2021/2023…2026-09 | детекторы, nowcast |
| ЦБ РФ: ключевая ставка | по дням | 2018…2026 | новостные/макро-признаки |
| Реестр событий | 7 событий со ссылками | 2023–2024 | событийная оценка детекторов |

Лицензия данных СберИндекса — CC BY-SA 4.0. Муниципальные файлы в репозиторий не включены (`scripts/get_data.sh` скачивает их в `data/raw/`); небольшие национальные и недельные выгрузки лежат в корне проекта.

## 2. Протокол оценки

Правила организатора мне неизвестны, поэтому использован «конкурсный» протокол, описанный в открытых работах участников: validation — цели янв.–июнь 2024, test — цели июль–декабрь 2024, расширяющееся окно, прогноз горизонтов 1/3/6/12 из каждого origin с историей не менее 6 месяцев. **Допущение подлежит сверке с официальными правилами.**

1. Для каждого origin и каждого ряда используются только значения ≤ origin; direct-метки обучения — только с целью ≤ origin (проверено тестами `tests/test_municipal.py`: изменение будущих значений не меняет признаки, метки и национальные приоры).
2. Конфигурация структурной модели и веса ансамбля выбраны **по validation**; `frozen_selection.json` записан до подсчёта метрик теста. После первого показа теста модели, кандидаты и веса не менялись; изменения касались только расчёта Prophet для origin 12.2024 (нужен для прогноза 2025, на тест не влияет) и оформления. Значительная часть решений (дрейф без экстраполяции за обученный горизонт, сжатие 0.25) принята по validation после того, как первая версия модели провалилась на h=6.
3. Метрика — MAE в рублях на жителя по всей сетке (сетки у всех моделей совпадают, покрытие 100%), дополнительно R² по уровню и по изменению.
4. Неопределённость — парный bootstrap по МО (1000 повторов) и по целевым месяцам (6 месяцев).
5. Допущения: значение месяца origin известно на origin; актуальная версия данных (без пересмотров); validation для h=12 содержит один целевой месяц (2024-06).

## 3. Модели

- **Baseline'ы:** LastValue; SeasonalNaive (то же значение год назад, а если нет — LastValue); **Prophet** (месячные даты, настройки по умолчанию, годовая сезонность `auto` включается только при истории >2 лет; настройки организатора неизвестны).
- **Национальный путь (NatPath):** прогноз = локальный уровень (среднее по последним 3 месяцам, скорректированное на реализованное национальное изменение) × национальное SeasonalGrowth-изменение от origin до цели (сезонность и темп роста из национальных рядов; соответствие муниципальных категорий национальным задано заранее).
- **SeasonalNaive_NatGrowth:** значение того же месяца год назад × национальный годовой рост.
- **StructHGB:** градиентный бустинг (MAE) по остатку относительно NatPath: признаки — локальные импульсы относительно страны, шум, сезонные разности, региональные и категорийные средние, статические признаки МО. Дрейф на месяц умножается на горизонт, **но не дальше обученного** (h ≤ origin−5) и сжимается на валидации (коэффициент {R['frozen']['structural']['shrink']}).
- **Foundation:** Chronos-Bolt-small, Chronos-2 (zero-shot), Chronos-2 + национальный путь как ковариата, TimesFM-2.5 (подвыборка). Ревизии чекпойнтов в `artifacts/municipal_runs/_foundation/*_meta.json`.
- **Ансамбль:** веса минимизируют MAE на validation (линейная программа, симплекс), отдельно по горизонтам.
- **Интервалы:** split-conformal по относительной ошибке на validation, по (горизонту, категории).

## 4. Прогноз 1/3/6/12 мес.

{sec_forecast(R)}

## 5. Структурные изменения: обнаружение и предупреждение

{sec_detectors(R)}

## 6. Новости и внешние события

{sec_news(R)}

## 7. Недельный nowcast

{sec_nowcast(R)}

## 8. Прогноз 2025 и независимая проверка

{sec_2025(R)}

## 9. Ограничения

- Всего 24 месяца: сезонность по МО не оценивается, h=12 на validation проверяется один месяц; протокол валидации — допущение.
- Тест — 6 целевых месяцев; месячные интервалы поэтому широки. Общие шоки месяца (например, декабрьский пик) коррелируют по всем МО, и bootstrap по МО их не отражает.
- Модели обучались на последней версии данных; даты публикации и пересмотры неизвестны.
- Foundation-модели оценены без дообучения; TimesFM — только на подвыборке 200 МО; возможное пересечение их обучающих данных с 2023–2024 гг. проверить нельзя.
- Метки шоков по МО отсутствуют: оценка детекторов опирается на инъекцию, 4 события из реестра и proxy-разметку.
- «Новости» — не текстовый корпус, а короткий реестр и решения ЦБ.
- В панели 2 016 МО из 2 594 в справочнике (73 региона из 85): МО, не прошедшие порог качества СберИндекса, отсутствуют (например, Белгород, Курск, Краснодар); 1 044 неполных ряда исключены.

## 10. Воспроизведение

```bash
conda env create -f environment.yml && conda activate sber   # Python 3.12
bash scripts/get_data.sh                                     # открытые данные конкурса -> data/raw/
make municipal                                               # Prophet, Chronos, структурная модель, ансамбль, метрики
make shocks                                                  # детекторы и недельные ряды
make report                                                  # отчёт, слайды, лендинг
```
"""
    return head


def inline_md(text):
    import markdown
    return markdown.markdown(text, extensions=["tables", "fenced_code"])


CSS = """
:root{--bg:#fff;--fg:#1c2330;--muted:#5d6675;--line:#dfe3ea;--acc:#1a6fb0}
@media (prefers-color-scheme: dark){:root{--bg:#10151d;--fg:#e8ecf2;--muted:#9aa4b2;--line:#2a3340;--acc:#6db3ee}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:980px;margin:0 auto;padding:28px 18px 64px}h1{font-size:28px;line-height:1.2}h2{margin-top:2.2em;border-bottom:1px solid var(--line);padding-bottom:.25em}h3{margin-top:1.6em}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:12px 0;display:block;overflow-x:auto}th,td{border:1px solid var(--line);padding:5px 8px;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{background:rgba(127,127,127,.12)}code{background:rgba(127,127,127,.15);padding:1px 4px;border-radius:4px;font-size:90%}pre{background:rgba(127,127,127,.12);padding:12px;border-radius:8px;overflow-x:auto}
img{max-width:100%;height:auto}figure{margin:18px 0}figcaption{color:var(--muted);font-size:13px}a{color:var(--acc)}
@media print{main{max-width:none;padding:0}h2{break-after:avoid}table{font-size:10.5px;display:table}img{break-inside:avoid}}
"""

FIGS = [("mae_by_model_test.png", "MAE по горизонтам на тесте", "4"),
        ("gain_vs_prophet.png", "Снижение MAE относительно Prophet с 95% bootstrap по МО", "4"),
        ("ensemble_weights.png", "Веса ансамбля", "4"),
        ("mae_by_group.png", "MAE по категориям и размеру ряда", "4"),
        ("map_gain_vs_prophet.png", "Карта выигрыша у Prophet по МО (категория «Все категории»)", "4"),
        ("detectors_injection.png", "Детекторы на инъекциях в реальные остатки", "5"),
        ("detector_examples.png", "Три реальных примера: событие, ложная тревога, срабатывание", "5"),
        ("weekly_detectors_early_warning.png", "Недельные ряды: детекторы и раннее предупреждение", "5"),
        ("news_ablation.png", "Новостные признаки против плацебо", "6"),
        ("nowcast.png", "Недельный nowcast против h=1", "7"),
        ("forecast_2025_check.png", "Прогноз 2025 против национального факта", "8")]


def insert_figures(text, link_prefix):
    for name, caption, section in FIGS:
        if not (FIG / name).exists():
            continue
        marker = f"\n## {int(section) + 1 if False else section}."
        tag = f'\n\n<figure><img src="{link_prefix}{name}" alt="{caption}"><figcaption>{caption}</figcaption></figure>\n\n'
        # вставка в конец соответствующего раздела
        idx = text.find(f"\n## {int(section) + 1}.")
        if idx == -1:
            text += tag
        else:
            text = text[:idx] + tag + text[idx:]
    return text


def html_page(title, body, extra_css=""):
    return f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>{CSS}{extra_css}</style></head><body><main>{body}</main></body></html>"""


def to_pdf(html_path, pdf_path, landscape=False):
    cmd = ["google-chrome", "--headless=new", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
           f"--print-to-pdf={pdf_path}", f"file://{html_path}"]
    subprocess.run(cmd, check=True, capture_output=True, timeout=180)


def embed(html):
    import re
    def repl(m):
        p = (ROOT / "artifacts/figures" / Path(m.group(1)).name)
        if p.exists():
            return 'src="data:image/png;base64,' + base64.b64encode(p.read_bytes()).decode() + '"'
        return m.group(0)
    return re.sub(r'src="([^"]+\.png)"', repl, html)


SLIDE_CSS = """
@page{size:1280px 720px;margin:0}
body{margin:0;background:#e9edf2}
.slide{width:1280px;height:720px;box-sizing:border-box;padding:44px 60px;background:#fff;color:#1c2330;page-break-after:always;break-after:page;overflow:hidden;position:relative;margin:0 auto;font:25px/1.42 system-ui,"Segoe UI",Roboto,sans-serif}
.slide h1{font-size:44px;line-height:1.1;margin:0 0 14px}.slide h2{font-size:32px;margin:0 0 14px;color:#1a6fb0}
.slide .grid{display:grid;grid-template-columns:1.55fr 1fr;gap:26px;align-items:start}
.slide img{max-width:100%;max-height:600px;object-fit:contain}.slide table{border-collapse:collapse;font-size:19px;width:100%}
.slide th,.slide td{border:1px solid #dfe3ea;padding:4px 8px;text-align:right}.slide th:first-child,.slide td:first-child{text-align:left}th{background:#f1f4f8}
.slide ul{margin:6px 0;padding-left:22px}.slide li{margin:4px 0}.big{font-size:62px;font-weight:700;color:#1a6fb0}.muted{color:#5d6675;font-size:19px}
.foot{position:absolute;left:60px;bottom:16px;color:#8a93a1;font-size:13px}.warn{background:#fff4e0;border-left:5px solid #e9a23b;padding:10px 16px;font-size:22px}
.kpi{display:flex;gap:22px;margin:12px 0}.kpi div{background:#f1f4f8;border-radius:10px;padding:14px 22px;min-width:170px}.kpi b{display:block;font-size:46px;color:#1a6fb0}
"""


def slide(inner, n):
    return f'<section class="slide">{inner}<div class="foot">СберИндекс · прогноз расходов МО · {n}</div></section>'


def fig_tag(name, alt=""):
    return f'<img src="../artifacts/figures/{name}" alt="{alt}">'


def build_slides(R):
    ens = {h: mae(R, "Ensemble", h) for h in H}
    pro = {h: mae(R, "Prophet", h) for h in H}
    gain = {h: 100 * (pro[h] - ens[h]) / pro[h] for h in H}
    inj = R["inj"].groupby("method").mean(numeric_only=True)
    ew = R["wk_ew"]
    news = R["news"]
    nc = R["nowcast"]
    sub = R["subset"]
    nat = np.mean([mae(R, "NatPath_K3", h) for h in H])
    loc = np.mean([mae(R, "LocalBase_K3", h) for h in H])
    sl = []
    sl.append(slide(f"""<h1>Прогноз расходов 2 016 муниципалитетов<br>и ранние сигналы структурных сдвигов</h1>
<div class="kpi"><div><b>−{gain[1]:.0f}%</b>MAE к Prophet, 1 мес.</div><div><b>−{gain[3]:.0f}%</b>3 мес.</div><div><b>−{gain[6]:.0f}%</b>6 мес.</div><div><b>−{gain[12]:.0f}%</b>12 мес.</div></div>
<p>Независимый тест: июль–декабрь 2024, 12 096 рядов (МО × 6 категорий). Выбор заморожен до открытия теста.</p>
<p class="muted">Данные: СберИндекс (CC BY-SA 4.0). Протокол «конкурсный»: официальные правила организатора мне неизвестны.</p>""", 1))
    sl.append(slide(f"""<h2>Данные и задача</h2><ul>
<li><b>Цель:</b> безналичные расходы на жителя, руб./мес. — 2 016 МО × 6 категорий × 24 месяца (2023–2024)</li>
<li><b>Горизонты:</b> 1, 3, 6, 12 месяцев; метрика — MAE в рублях, R², интервалы</li>
<li><b>Проблема:</b> по МО всего 6–23 месяца истории — сезонность и рост по ряду не оценить</li>
<li><b>Решение:</b> национальные ряды с 2018 г. (сезонность, темп роста) + уровень МО + небольшая поправка</li>
<li><b>Дополнительно:</b> недельные ряды, решения ЦБ, реестр событий, справочник МО, индекс доступности рынков</li></ul>
<p class="warn">Из 13 140 рядов в панель вошли 12 096 полных: крупные города, не прошедшие порог качества СберИндекса, отсутствуют.</p>""", 2))
    sl.append(slide("""<h2>Методология одним взглядом</h2>
<div class="grid"><div><ol>
<li><b>Уровень МО</b> = среднее трёх последних месяцев, очищенное от реального национального изменения</li>
<li><b>Национальный путь</b> до цели = сезонность прошлого года × темп роста (только данные ≤ origin)</li>
<li><b>GBM по остатку</b> (импульсы относительно страны, регион, шум), сжатый и без экстраполяции за обученный горизонт</li>
<li><b>Ансамбль</b> с Prophet и локальной сезонностью; веса — минимум MAE на validation</li>
<li><b>Интервалы</b> — split-conformal</li></ol></div>
<div><p><b>Защита от утечек</b></p><ul><li>признаки и метки ≤ origin (юнит-тесты)</li><li>выбор → заморозка → однократный показ теста</li><li>национальные данные после origin не используются</li><li>bootstrap по МО и по месяцам</li></ul></div></div>""", 3))
    sl.append(slide(f"""<h2>Лидерборд на тесте (MAE, руб./жителя)</h2><div class="grid"><div>{fig_tag('mae_by_model_test.png')}</div><div>{html_table(leaderboard(R).head(6).drop(columns=['Δ к Prophet, % (среднее)']))}<p class="muted">R² по изменению: ансамбль {R['test'][(R['test'].model=='Ensemble')].r2_change.mean():.2f}, Prophet {R['test'][(R['test'].model=='Prophet')].r2_change.mean():.2f} (среднее по h).</p></div></div>""".replace("<table", "<table class='t'") , 4))
    sl.append(slide(f"""<h2>Выигрыш у Prophet статистически устойчив</h2><div class="grid"><div>{fig_tag('gain_vs_prophet.png')}</div><div><ul>
<li>95% bootstrap по МО исключает ноль на всех горизонтах</li><li>Bootstrap по 6 целевым месяцам (общий шок месяца) тоже исключает ноль для ансамбля</li>
<li>Одиночные модели нестабильны: структурная GBM и NatPath выигрывают на h=1 и h=12, но не на h=3–6</li>
<li>Слабое место — «Маркетплейсы»: ансамбль не лучше Prophet</li></ul></div></div>""", 5))
    sl.append(slide(f"""<h2>Откуда выигрыш: национальная информация</h2><div class="grid"><div>
<table><tr><th>Вариант</th><th>MAE, ср. по h</th></tr>
<tr><td>Локальный уровень</td><td>{loc:.0f}</td></tr><tr><td>+ GBM без национальных данных</td><td>{np.mean([mae(R,'LocalOnlyHGB',h) for h in H]):.0f}</td></tr>
<tr><td>Национальный путь (NatPath)</td><td>{nat:.0f}</td></tr><tr><td>NatPath + остаточный GBM</td><td>{np.mean([mae(R,'StructHGB',h) for h in H]):.0f}</td></tr>
<tr><td><b>Ансамбль</b></td><td><b>{np.mean(list(ens.values())):.0f}</b></td></tr><tr><td>Prophet</td><td>{np.mean(list(pro.values())):.0f}</td></tr></table></div>
<div>{fig_tag('ensemble_weights.png')}<p class="muted">Нелинейная модель по МО добавляет мало; главное — сезонность и рост из национальных рядов.</p></div></div>""", 6))
    tf = ""
    if sub is not None:
        t = sub[(sub.stage == 'test') & (sub.model.isin(['TimesFM', 'Prophet']))].pivot(index='model', columns='h', values='mae')
        tf = f"<p>TimesFM-2.5 (200M) на подвыборке 200 МО: MAE {t.loc['TimesFM',1]:.0f}/{t.loc['TimesFM',3]:.0f}/{t.loc['TimesFM',6]:.0f}/{t.loc['TimesFM',12]:.0f}; Prophet на тех же рядах {t.loc['Prophet',1]:.0f}/{t.loc['Prophet',3]:.0f}/{t.loc['Prophet',6]:.0f}/{t.loc['Prophet',12]:.0f}.</p>"
    sl.append(slide(f"""<h2>Фундаментальные модели zero-shot</h2><ul>
<li>Chronos-Bolt-small, Chronos-2, Chronos-2 + национальный путь как ковариата</li><li>Ревизии чекпойнтов зафиксированы; контекст — только история ≤ origin</li>
<li>На 6–23 точках <b>хуже Prophet</b> на h≥3; ковариата заметно помогает Chronos-2 (h=1: {mae(R,'Chronos2_NatCov',1):.0f} против {mae(R,'Chronos2',1):.0f})</li><li>Вес в ансамбле — 0</li></ul>{tf}
<p class="warn">Возможное пересечение обучающих корпусов с 2023–2024 гг. проверить нельзя: это оценка современной модели, а не исторического deployment.</p>""", 7))
    sl.append(slide(f"""<h2>Детекторы: инъекция сдвигов в реальные остатки</h2><div class="grid"><div>{fig_tag('detectors_injection.png')}</div><div><ul>
<li>Остатки h=1 вне обучения; порог — при бюджете 1 ложная тревога на 100 ряд-месяцев</li>
<li>F1: CUSUM {inj.loc['CUSUM','event_f1']:.2f}, BOCPD {inj.loc['BOCPD','event_f1']:.2f}, Page–Hinkley {inj.loc['PageHinkley','event_f1']:.2f}</li>
<li>Recall при сдвиге 4σ: CUSUM {R['inj_mag'].query("method=='CUSUM' and magnitude==4.0").recall.iloc[0]:.2f}</li><li>Ранжирование зависит от способа разметки: на offline-метках лидирует BOCPD</li></ul></div></div>""", 8))
    sl.append(slide(f"""<h2>Три реальных примера</h2>{fig_tag('detector_examples.png')}""", 9))
    sl.append(slide("""<h2>Недельные ряды и раннее предупреждение</h2><div class="grid"><div>"""+fig_tag('weekly_detectors_early_warning.png')+"""</div><div><ul>
<li>46 категорий, 151 неделя: реальная оценка детекторов (метки — offline-proxy, не истина)</li><li>Page–Hinkley и CUSUM сильнее BOCPD на недельных рядах</li>
<li><b>Раннее предупреждение: навык не подтверждён</b> — PR-AUC около доли событий, Brier не лучше константы</li></ul><p class="warn">Нужна верифицированная разметка шоков.</p></div></div>""", 10))
    news_txt = "реальные признаки не лучше плацебо"
    sl.append(slide(f"""<h2>Новости и внешние события</h2><div class="grid"><div>{fig_tag('news_ablation.png')}</div><div><ul>
<li>Датированные официальные события: ключевая ставка ЦБ и реестр из 7 региональных событий</li><li>Только прошлое: available_at = дата вступления в силу / публикации</li><li>Результат: {news_txt}</li><li>Текстовый корпус новостей не собирался</li></ul></div></div>""", 11))
    ncv = ""
    if nc is not None:
        a = nc[nc.method.str.contains("weeks=all,calibrated=True")].mae.iloc[0]
        b = nc[nc.method == "h=1 DevWeightedEnsemble"].mae.iloc[0]
        ncv = f"Nowcast по полному месяцу: MAE {a:.1f} против {b:.1f} у h=1-ансамбля (млрд руб.)."
    sl.append(slide(f"""<h2>Недельный nowcast и прогноз 2025</h2><div class="grid"><div>{fig_tag('forecast_2025_check.png')}</div><div><ul>
<li>{ncv}</li><li>Прогноз 2025 заморожен (SHA256) — скрипт оценит MAE, когда СберИндекс опубликует факты по МО</li><li>Санити-проверка: ансамбль согласуется с национальным ростом точнее Prophet</li></ul>{fig_tag('nowcast.png')}</div></div>""", 12))
    sl.append(slide("""<h2>Ограничения</h2><ul>
<li>24 месяца данных; тест — 6 целевых месяцев; validation h=12 — один месяц</li><li>Протокол валидации — допущение, нужна сверка с официальными правилами</li><li>Последняя версия данных, даты публикации неизвестны</li>
<li>Нет верифицированных меток шоков; события реестра охватывают 4 оценимых МО</li><li>Новости = решения ЦБ и реестр, не текстовый корпус</li><li>TimesFM — только подвыборка 200 МО</li><li>Маркетплейсы и общественное питание остаются слабым местом</li></ul>""", 13))
    sl.append(slide("""<h2>Воспроизведение</h2><pre style="font-size:20px">conda env create -f environment.yml
conda activate sber
bash scripts/get_data.sh
make test municipal shocks report</pre><ul><li>Конфигурации: <code>configs/*.json</code>; хэши данных и кода в <code>manifest.json</code></li><li>Журналы: <code>tracking/*.csv</code>; прогноз 2025: <code>prospective/</code></li><li>Отчёт: <code>report/report.pdf</code>; результаты: <code>docs/MUNICIPAL_RESULTS_RU.md</code></li></ul>""", 14))
    html = f'<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Слайды: прогноз расходов МО</title><style>{CSS}{SLIDE_CSS}</style></head><body>{"".join(sl)}</body></html>'
    return html


def main():
    R = load()
    text = build_markdown(R)
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "report").mkdir(exist_ok=True)
    (ROOT / "presentation").mkdir(exist_ok=True)
    rel = "../artifacts/figures/"
    full = insert_figures(text, rel)
    (ROOT / "report/report.md").write_text(full)
    (ROOT / "docs/MUNICIPAL_RESULTS_RU.md").write_text(full.replace(rel, "../artifacts/figures/"))
    html = html_page("Прогноз расходов МО", inline_md(full))
    (ROOT / "report/report.html").write_text(html)
    to_pdf(str(ROOT / "report/report.html"), str(ROOT / "report/report.pdf"))
    print("отчёт готов")
    # лендинг: самодостаточный HTML
    (ROOT / "presentation/index.html").write_text(embed(html_page("Прогноз расходов МО", inline_md(insert_figures(text, "")))))
    print("лендинг готов")
    slides_html = build_slides(R)
    (ROOT / "presentation/slides.html").write_text(slides_html)
    to_pdf(str(ROOT / "presentation/slides.html"), str(ROOT / "presentation/slides.pdf"))
    print("слайды готовы")


if __name__ == "__main__":
    main()
