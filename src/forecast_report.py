"""Графики и русскоязычный отчёт по сохранённым прогнозам, без переобучения."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages


def markdown_table(df):
    # Без необязательной зависимости tabulate.
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *rows])


def render_report(root, run):
    manifest = json.loads((run / "manifest.json").read_text())
    selection = json.loads((run / "frozen_selection.json").read_text())
    metrics = pd.read_csv(run / "metrics.csv")
    per_series = pd.read_csv(run / "per_series_metrics.csv")
    predictions = pd.read_csv(run / "predictions.csv", parse_dates=["origin", "target_date"])
    future = pd.read_csv(run / "future_forecasts.csv", parse_dates=["origin", "target_date"])
    panel = pd.read_csv(run / "target_panel.csv", index_col="period", parse_dates=True)
    split = pd.read_csv(run / "split_manifest.csv", parse_dates=["origin", "target_date"])
    bootstrap = pd.read_csv(run / "paired_bootstrap.csv")
    selected = selection["selected_model_on_development"]
    dev_rank = metrics[metrics.stage == "development"].groupby("model").macro_mae.mean().sort_values()
    test_rank = metrics[metrics.stage == "holdout"].groupby("model").macro_mae.mean().sort_values()
    models = dev_rank.index.tolist()
    horizons = sorted(metrics.horizon_months.unique())
    figures = run / "figures"
    figures.mkdir(exist_ok=True)
    figure_names = []
    pdf_path = run / "forecast_comparison.pdf"
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.text(.07, .9, "Прогноз расходов СберИндекса: сравнение моделей", fontsize=20)
        lines = [f"Россия; {len(panel.columns)} расходных категорий × {len(panel)} месяцев; единицы: млрд руб.",
                 f"Данные: {panel.index.min():%m.%Y}–{panel.index.max():%m.%Y}",
                 f"34 development origins; тест: {manifest['holdout_start']}–{manifest['last_month']}",
                 f"Выбрана до теста: {selected}",
                 f"Средний MAE выбранной модели на тесте: {test_rank[selected]:.2f} млрд руб.",
                 "MAE — среднее по пяти категориям, затем по горизонтам 1 / 3 / 6 / 12.",
                 "Параметры заморожены до теста; обучение обновляется на каждом origin.",
                 "Одна текущая версия истории; реальные задержки публикации неизвестны.",
                 "Это национальный эксперимент, не прогноз муниципальных расходов.",
                 "Prophet / Chronos / TimesFM не запущены: зависимости/веса недоступны.",
                 "Детекторы шоков и новости в данном прогнозном запуске не оценивались."]
        for i, line in enumerate(lines):
            fig.text(.07, .79 - .052 * i, line, fontsize=12)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        def save(fig, name):
            fig.savefig(figures / f"{name}.png", dpi=140, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            figure_names.append(name)

        fig, ax = plt.subplots(figsize=(13, 4.5))
        segments = [
            ("Начальная train-история", panel.index[0], panel.index[35], "#64748b"),
            ("Dev origins", split[split.stage == "development"].origin.min(), split[split.stage == "development"].origin.max(), "#2563eb"),
            ("Dev цели (все h)", split[split.stage == "development"].target_date.min(), split[split.stage == "development"].target_date.max(), "#60a5fa"),
            ("Holdout цели", split[split.stage == "holdout"].target_date.min(), split[split.stage == "holdout"].target_date.max(), "#ea580c"),
            ("Holdout origins для h=12", split[(split.stage == "holdout") & (split.horizon_months == 12)].origin.min(), split[(split.stage == "holdout") & (split.horizon_months == 12)].origin.max(), "#fbbf24")]
        for i, (label, start, end, color) in enumerate(segments):
            stop = end + pd.DateOffset(months=1)
            ax.barh(i, mdates.date2num(stop) - mdates.date2num(start), left=mdates.date2num(start), color=color, height=.6)
            ax.text(mdates.date2num(start) + 10, i, f"{start:%m.%Y}–{end:%m.%Y}", va="center", fontsize=9)
        ax.set_yticks(range(len(segments)), [s[0] for s in segments])
        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_title("Календарная валидация: цели dev не пересекают holdout")
        ax.grid(axis="x", alpha=.2)
        fig.tight_layout()
        save(fig, "validation_timeline")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        lower, upper = metrics.macro_mae.min(), metrics.macro_mae.max()
        for stage, ax, title in zip(["development", "holdout"], axes, ["Подбор параметров: development", "Финальный тест: holdout"]):
            table = metrics[metrics.stage == stage].pivot(index="model", columns="horizon_months", values="macro_mae").reindex(models)
            im = ax.imshow(table.to_numpy(), cmap="YlOrRd", aspect="auto", vmin=lower, vmax=upper)
            for i in range(len(models)):
                for j in range(len(horizons)):
                    color = "white" if table.iloc[i,j] > lower + .7 * (upper - lower) else "black"
                    ax.text(j, i, f"{table.iloc[i,j]:.1f}", ha="center", va="center", fontsize=9, color=color)
            ax.set_xticks(range(len(horizons)), horizons)
            ax.set_yticks(range(len(models)), [m + (" ★" if m == selected else "") for m in models])
            ax.set(xlabel="Горизонт, месяцев", title=title)
            fig.colorbar(im, ax=ax, label="MAE, млрд руб.", shrink=.8)
        fig.suptitle("MAE: среднее по категориям. ★ Модель выбрана только на dev", fontsize=13)
        fig.tight_layout()
        save(fig, "mae_heatmap")

        fig, ax = plt.subplots(figsize=(12, 6))
        rank = pd.DataFrame({"Development": dev_rank, "Holdout": test_rank}).reindex(models)
        rank.plot.barh(ax=ax, color=["#2563eb", "#ea580c"])
        ax.set(xlabel="Средний MAE по 4 горизонтам, млрд руб.", title="Сравнение качества до и после фиксации параметров")
        ax.grid(axis="x", alpha=.2)
        fig.tight_layout()
        save(fig, "model_ranking")

        fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharey=True)
        colors = {selected: "#2563eb", "SeasonalNaive": "#64748b", "SeasonalGrowth": "#ea580c"}
        compare_models = list(dict.fromkeys([selected, "SeasonalNaive", "SeasonalGrowth"]))
        for h, ax in zip(horizons, axes.flat):
            actual = panel["Всего"].iloc[-24:]
            ax.plot(actual.index, actual, color="black", lw=2, label="Факт")
            for model in compare_models:
                part = predictions[(predictions.stage == "holdout") & (predictions.series_id == "Всего") &
                                   (predictions.horizon_months == h) & (predictions.model == model)].sort_values("target_date")
                ax.plot(part.target_date, part.y_pred, marker="o", ms=3, color=colors[model], label=model)
            ax.axvline(pd.Timestamp(manifest["holdout_start"]), color="black", linestyle="--", alpha=.4)
            ax.set(title=f"Всего: горизонт {h} мес.", ylabel="Млрд руб.")
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m.%y"))
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
        fig.suptitle("Holdout: каждая точка — прогноз из своего origin, не одна траектория", fontsize=13)
        fig.tight_layout()
        save(fig, "holdout_total_forecasts")

        fig, axes = plt.subplots(3, 2, figsize=(14, 11))
        for series, ax in zip(panel.columns, axes.flat):
            for model, color in [(selected, "#2563eb"), ("SeasonalNaive", "#64748b")]:
                part = predictions[(predictions.stage == "holdout") & (predictions.series_id == series) &
                                   (predictions.horizon_months == 3) & (predictions.model == model)].sort_values("target_date")
                ax.plot(part.target_date, part.y_pred, label=model, color=color, marker="o", ms=3)
            actual = panel[series].iloc[-12:]
            ax.plot(actual.index, actual, color="black", label="Факт", lw=2)
            ax.set(title=f"{series}: h=3", ylabel="Млрд руб.")
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m.%y"))
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
        axes.flat[-1].axis("off")
        fig.tight_layout()
        save(fig, "holdout_categories")

        fig, ax = plt.subplots(figsize=(12, 6))
        r2 = per_series[per_series.stage == "holdout"].groupby(["model", "series_id"]).r2.mean().unstack().reindex(models)
        ax.imshow(r2.to_numpy(), cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)
        for i in range(len(models)):
            for j in range(len(r2.columns)):
                ax.text(j, i, f"{r2.iloc[i,j]:.2f}", ha="center", va="center", fontsize=9)
        ax.set_xticks(range(len(r2.columns)), r2.columns, rotation=15, ha="right")
        ax.set_yticks(range(len(models)), models)
        ax.set_title("R² внутри каждой категории: среднее по 4 горизонтам (holdout)")
        fig.tight_layout()
        save(fig, "r2_by_category")

        fig, ax = plt.subplots(figsize=(13, 5))
        actual = panel["Всего"].iloc[-36:]
        ax.plot(actual.index, actual, color="black", label="Факт", lw=2)
        part = future[(future.series_id == "Всего") & (future.model == selected)].sort_values("target_date")
        ax.scatter(part.target_date, part.y_pred, color="#2563eb", s=60, label=f"Прогноз {selected}")
        for row in part.itertuples():
            ax.annotate(f"h={row.horizon_months}\n{row.y_pred:.0f}", (row.target_date, row.y_pred), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=9)
        ax.axvline(panel.index[-1], linestyle="--", color="gray")
        ax.set(title="Всего: прогнозы от последнего контекста (август 2026)", ylabel="Млрд руб.")
        ax.legend()
        ax.grid(alpha=.2)
        fig.tight_layout()
        save(fig, "future_forecast")

    holdout_table = metrics[metrics.stage == "holdout"].pivot(index="model", columns="horizon_months", values="macro_mae").reindex(models)
    holdout_table["Средний MAE"] = test_rank
    holdout_table = holdout_table.round(2).reset_index()
    holdout_table.columns = ["Модель", "1 мес.", "3 мес.", "6 мес.", "12 мес.", "Средний MAE"]
    selected_total = per_series[(per_series.stage == "holdout") & (per_series.model == selected) & (per_series.series_id == "Всего")][["horizon_months", "mae", "r2"]].sort_values("horizon_months").round(3)
    selected_total.columns = ["Горизонт, мес.", "MAE Всего, млрд руб.", "R² Всего"]
    improvement = 100 * (test_rank["SeasonalNaive"] - test_rank[selected]) / test_rank["SeasonalNaive"]
    best_test = test_rank.index[0]
    growth_improvement = 100 * (test_rank["SeasonalGrowth"] - test_rank[selected]) / test_rank["SeasonalGrowth"]
    report = f"""# Результаты обучения и сравнения прогнозных моделей

Запуск `{manifest['run_id']}`. Обучены 8 семейств моделей и построен ансамбль из трёх семейств. Данные — реальные месячные абсолютные расходы СберИндекса: Россия, пять расходных категорий, 93 месяца с декабря 2018 по август 2026. Единицы MAE — млрд рублей. Это общероссийский эксперимент, а не выполнение муниципальной части задания.

## Результат выбора

По development выбран **{selected}**. Его средний MAE на holdout — **{test_rank[selected]:.2f} млрд руб.**: на **{improvement:.1f}% меньше SeasonalNaive** и на **{growth_improvement:.1f}% меньше более сильного SeasonalGrowth**. Фактический минимум на holdout у **{best_test}** ({test_rank.iloc[0]:.2f}), но выбор решения после просмотра теста не менялся. Для выбора усредняли MAE пяти категорий, затем равновесно четыре горизонта. Категории различаются по масштабу; «Всего» отдельно показано ниже и не суммируется с детализацией.

## MAE на независимом holdout

{markdown_table(holdout_table)}

Здесь все модели дают прогнозы на одинаковых 12 целевых месяцах × 5 категориях для каждого горизонта: 60 наблюдений на h. Разница MAE на разных горизонтах показана явно; средняя метрика не означает победу на каждом горизонте. На текущем тесте минимальный MAE для h=1 у CatBoost, для h=3 у SeasonalGrowth, для h=6/12 у ансамбля. Это разбор уже открытого теста: не используем его для переопределения модели по горизонтам.

## Детализация выбранной модели для «Всего»

{markdown_table(selected_total)}

R² внутри категории измеряет изменение временного ряда. Pooled R² по пяти категориям также сохранён в CSV, но может быть высоким благодаря различию масштаба категорий, поэтому сам по себе недостаточен для оценки динамики. Отрицательные R² сохранены без обрезки.

## Как проверяли

Train начинаетcя в декабре 2018; минимальный контекст 36 месяцев. 34 общих development origins: ноябрь 2021–август 2024. Все dev-цели находятся раньше сентября 2025. Последние 12 месяцев целей, сентябрь 2025–август 2026, оставлены для holdout. Для h=12 origins теста начинаются в сентябре 2024; это корректно, потому что модель прогнозирует вперёд из доступной на тот origin истории.

Параметры каждого семейства и веса ансамбля зафиксированы в frozen_selection.json до расчёта holdout. На тесте модель переобучается при продвижении origin, включая только уже наблюдённый контекст; гиперпараметры и состав ансамбля не меняются. Для direct-примеров выполняется anchor+h <= origin, что проверено тестами и training_audit CSV. Значения будущих месяцев не используются ни для масштаба, ни для скользящих признаков. Изменение всех значений после origin в проверке не меняет признаки и Ridge-прогнозы.

Последняя версия данных не содержит available_at и исторических vintages. Принято допущение, что контекстный месяц известен на origin. Это latest-vintage evaluation; неизвестные задержки публикации и позднейшие пересмотры ограничивают переносимость результатов на реальный ранний прогноз. Цель holdout не участвовала в выборе параметров, но исследовательские графики и последующий просмотр теста не превращают его в новый development. Повторный тюнинг по этому тесту потребует другого независимого периода.

## Что обучено

- LastValue: последнее известное значение.
- SeasonalNaive: тот же месяц прошлого года.
- SeasonalGrowth: прошлогодний месяц, умноженный на отношение последних и предыдущих 12-месячных средних.
- HoltWinters: настоящая statsmodels ExponentialSmoothing с годовой сезонностью и трендом; additive или multiplicative+damped вариант выбран на dev. Масштабирование делается внутри train; warnings и fallback записаны.
- FourierRidge: логарифм расходов, линейный тренд и три годовые Fourier-гармоники; локальная модель каждой категории, выбор окна 36/60 на dev.
- RidgeDirect, RandomForestDirect, CatBoostDirect: общая модель категорий с отдельной головой для каждого h. Используются только прошлые лаги, trailing-признаки, календарь и известная сезонная база. Цель — относительное отклонение от сезонной базы; обратно получаем прогноз в млрд руб.
- DevWeightedEnsemble: три лучших семейства по dev; веса обратно пропорциональны их dev-MAE на каждом h. Ансамбль не обучался на holdout.

Параметры и небольшой заранее заданный набор вариантов сохранены в configs/national_forecast.json. Dev-метрики кандидатов после подбора оптимистичны; окончательную оценку отбора даёт holdout. Никакие модели не заменяют Prophet: он отсутствует, установить его в текущем окружении не удалось. Chronos и TimesFM также не запускались из-за недоступных зависимостей/checkpoints. Победа над Prophet и преимущество фундаментальных моделей не проверены.

## Неопределённость и ограничения

{markdown_table(bootstrap.round(3))}

Разница определяется как MAE выбранной модели минус MAE baseline: отрицательная лучше. Парный круговой блочный bootstrap использует блоки по 3 целевых месяца, сохраняет вместе категории и горизонты, 1000 повторений. Всего 12 тестовых месяцев — интервалы нестабильны; они не заменяют проверку на других периодах. Если интервал включает ноль, убедительного выигрыша по этой проверке нет.

Модель будущего дообучена на всей доступной истории с теми же параметрами. Прогнозы для 1/3/6/12 месяцев от августа 2026 сохранены отдельно; это модельные оценки, а не фактические будущие расходы и не проверенные интервалы неопределённости.

## Соответствие заданию и следующий этап

Выполнены использование СберИндекса, обучение разных прогнозных моделей, временное сравнение на четырёх горизонтах, обязательная MAE, R², сохранение конфигурации, моделей, прогнозов, журналов и визуализаций. Не завершены муниципальные прогнозы (нет целевых расходов МО), сравнение с Prophet, фундаментальные модели, новости и оценка детекторов/предупреждений шоков. Эти пункты нельзя оценивать по текущему национальному прогнозу. Настройки baseline организатора также не предоставлены.

Новости и одновременные реальные индексы/приросты не вводились: фактический прирост прогнозируемого месяца зависит от его расходов и дал бы утечку. После появления дат публикаций возможен отдельный as-of эксперимент внешних признаков. Для шоков нужна фиксированная разметка событий и отдельные event-метрики, а не только визуальные отклонения прогноза.

## Воспроизведение и артефакты

Запуск: python scripts/train_forecasts.py. В текущей среде основной интерпретатор — /home/bugor/miniconda3/envs/algo/bin/python; Holt–Winters запускается в найденной среде trend. На другой машине установите requirements-models.txt в одну среду или задайте statsmodels_python в JSON. Все входные, конфигурационные и кодовые хеши, версии пакетов, временная сетка и параметры выбора сохранены в каталоге {run.relative_to(root)}. Полные fitted-модели финального контекста доступны в models; baseline воспроизводится из config и target_panel.

Ноутбук: notebooks/03_forecast_model_comparison.ipynb. PDF: {pdf_path.relative_to(root)}. Численные журналы: tracking/national_forecast_metrics.csv. Показатели национального опыта сохраняются отдельно от муниципального leaderboard.

Первичные описания инструментов: [Holt–Winters](https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html), [CatBoostRegressor](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor), [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html), [Prophet: временные отсечения](https://facebook.github.io/prophet/docs/diagnostics.html).
"""
    path = root / "docs/NATIONAL_FORECAST_RESULTS_RU.md"
    path.write_text(report, encoding="utf-8")
    (run / "results_report.md").write_text(report, encoding="utf-8")
    return {"selected_model": selected, "mean_test_mae": float(test_rank[selected]),
            "improvement_vs_seasonal_naive_pct": float(improvement), "figures": figure_names,
            "pdf": str(pdf_path.relative_to(root)), "report": str(path.relative_to(root))}
