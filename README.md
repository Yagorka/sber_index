# Прогноз расходов муниципалитетов и ранние сигналы структурных сдвигов (СберИндекс)

Прогноз средних безналичных расходов жителя 2 016 муниципальных образований по 6 категориям на 1/3/6/12 месяцев, сравнение детекторов структурных сдвигов, фундаментальные модели, «новостные» признаки, недельный nowcast и заморозка прогноза на 2025 год. Основа — открытые данные конкурса СберИндекса (CC BY-SA 4.0). Всё воспроизводится одной средой.

## Главный результат

Независимый тест: цели июль–декабрь 2024, 12 096 рядов (2 016 МО × 6 категорий). Выбор моделей и весов заморожен до открытия теста (`frozen_selection.json`).

| MAE, руб. на жителя в месяц | 1 мес. | 3 мес. | 6 мес. | 12 мес. |
|---|---:|---:|---:|---:|
| **Ансамбль (замороженный)** | **440** | **514** | **466** | **583** |
| Prophet (настройки по умолчанию) | 640 | 719 | 736 | 1 093 |
| Снижение MAE | −31% | −28% | −37% | −47% |

95% интервалы разности MAE (bootstrap по МО и по целевым месяцам) на всех горизонтах лежат ниже нуля. Откуда выигрыш: у МО всего 24 месяца, поэтому сезонность и рост берутся из национальных рядов с 2018 г. (абляция: −33% MAE только за счёт национального пути). Нелинейная модель по МО даёт мало. Полный разбор и все оговорки — в [отчёте](report/report.pdf) и [`docs/MUNICIPAL_RESULTS_RU.md`](docs/MUNICIPAL_RESULTS_RU.md).

**Что не получилось (и это записано в отчёте):**
- Фундаментальные модели (Chronos-Bolt, Chronos-2, Chronos-2 + национальная ковариата, TimesFM на подвыборке 200 МО) без дообучения уступают Prophet на h≥3; в ансамбле их вес 0.
- Раннее предупреждение сдвигов: навык не подтверждён (PR-AUC около доли событий, Brier не лучше константы). Детекторы реагируют на уже начавшийся сдвиг. Победитель CUSUM / Page–Hinkley / BOCPD зависит от способа разметки.
- «Новости» (решения ЦБ + реестр из 7 событий) не лучше плацебо. Текстовый корпус новостей не собирался.
- В категории «Маркетплейсы» ансамбль не лучше Prophet.

**Главное допущение.** Официальные правила и baseline организатора мне неизвестны. Использован «конкурсный» протокол из открытых работ участников: validation — цели янв.–июнь 2024, test — июль–декабрь 2024, минимальная история 6 месяцев. Его нужно сверить с правилами; Prophet здесь — «из коробки», без настроек организатора.

## Что обучено и какие метрики

Полный каталог (параметры, ревизии чекпойнтов, число подгонок, все метрики validation/test, детекторы, новости, национальный блок) — [`docs/MODELS_AND_METRICS_RU.md`](docs/MODELS_AND_METRICS_RU.md). Кратко, test (июль–декабрь 2024), MAE руб. на жителя:

| Модель | Что обучено | h=1 | h=3 | h=6 | h=12 | среднее |
|---|---|---:|---:|---:|---:|---:|
| **Ensemble** | LAD-веса на validation: Prophet + SeasonalNaive_NatGrowth + StructHGB | **440** | **514** | **466** | **583** | **501** |
| SeasonalNaive_NatGrowth | правило: прошлый год × национальный рост | 547 | 543 | 529 | 612 | 558 |
| NatPath_K3 | правило: локальный уровень × национальный путь | 526 | 758 | 757 | 687 | 682 |
| StructHGB | HistGradientBoosting по остатку (18 моделей на вариант веса) | 500 | 736 | 883 | 606 | 681 |
| Prophet | 229 824 подгонки, настройки по умолчанию | 640 | 719 | 736 | 1 093 | 797 |
| LocalOnlyHGB | абляция без национальных данных | 666 | 775 | 930 | 1 165 | 884 |
| Chronos-2 + национальная ковариата | zero-shot, 12 096 × 19 | 506 | 722 | 970 | 1 533 | 933 |
| Chronos-2 | zero-shot | 613 | 877 | 1 361 | 1 817 | 1 167 |
| Chronos-Bolt-small | zero-shot | 707 | 1 024 | 1 603 | 1 986 | 1 330 |
| TimesFM-2.5 | zero-shot, **только 200 МО** (сравнение на той же подвыборке: 549 / 786 / 1 245 / 1 630 против Prophet 622 / 695 / 709 / 1 078) | | | | | |

Остальные метрики:
- **R² по изменению** (test, h=1/3/6/12): ансамбль 0.67 / 0.74 / 0.84 / 0.69, Prophet 0.13 / 0.42 / 0.53 / −0.44; интервалы 90% покрывают 0.89 / 0.87 / 0.94 / 0.94.
- **Детекторы** (инъекции, бюджет 1/100): F1 CUSUM 0.24, BOCPD 0.19, Page–Hinkley 0.14. Недельные реальные ряды (proxy-метки): Page–Hinkley 0.39, CUSUM 0.31, BOCPD 0.14.
- **Раннее предупреждение** (недельные ряды): PR-AUC 0.16 при доле событий 0.12 (k=4), 0.39 при 0.36 (k=13): навыка нет.
- **Новости:** изменение MAE к базе на тесте −1.0 / −1.9 / −2.5 / −0.2% против плацебо от −2.6 до +1.2%.
- **Nowcast** (национальный уровень, млрд руб.): 62.0 по полному месяцу против 70.1 у h=1-ансамбля N01.
- **Национальный N01** (holdout, млрд руб.): ансамбль 74.66, CatBoost 78.91, SeasonalGrowth 86.99, SeasonalNaive 346.62.

## Материалы

| Что | Где |
|---|---|
| Методологический отчёт (PDF / HTML / MD) | [`report/report.pdf`](report/report.pdf), [`report/report.html`](report/report.html), [`report/report.md`](report/report.md) |
| Презентация и лендинг | [`presentation/slides.pdf`](presentation/slides.pdf), [`presentation/index.html`](presentation/index.html) (самодостаточный) |
| Что обучено и все метрики | [`docs/MODELS_AND_METRICS_RU.md`](docs/MODELS_AND_METRICS_RU.md) |
| Результаты муниципального блока | [`docs/MUNICIPAL_RESULTS_RU.md`](docs/MUNICIPAL_RESULTS_RU.md), `artifacts/municipal_runs/<run>/` |
| Детекторы и раннее предупреждение | `artifacts/shocks/` (МО), `artifacts/weekly_shocks/` (недельные ряды) |
| Новости / события | [`data/external/`](data/external/README.md), `artifacts/news_ablation/` |
| Недельный nowcast | `artifacts/nowcast/` |
| Замороженные прогнозы для будущей проверки | [`prospective/`](prospective/README.md): 2025 по МО, 09.2026 национальные |
| Журналы экспериментов и метрик | `tracking/*.csv` |
| Национальный эксперимент N01 | [`docs/NATIONAL_FORECAST_RESULTS_RU.md`](docs/NATIONAL_FORECAST_RESULTS_RU.md), ансамбль MAE 74.66 млрд руб. |
| Адаптивная ветка N02 (не улучшение, holdout не независим) | [`docs/ADAPTIVE_BRANCH_RU.md`](docs/ADAPTIVE_BRANCH_RU.md) |
| План моделирования | [`docs/MODELING_PLAN_RU.md`](docs/MODELING_PLAN_RU.md) |
| Аудит справочника МО и потребительских выгрузок | `notebooks/01…02`, `artifacts/eda/`, `artifacts/consumer_audit/` |

## Запуск

```bash
conda env create -f environment.yml && conda activate sber     # Python 3.12; проверено на версиях из requirements.lock.txt
bash scripts/get_data.sh                                       # открытые данные конкурса -> data/raw/ (в git не входят)
make test                                                      # юнит-тесты, включая проверки утечек
make foundation                                                # Prophet, Chronos, TimesFM: ~2 ч на 12 ядрах CPU (TimesFM — подвыборка)
make municipal shocks news nowcast check2025 report            # прогнозы, детекторы, абляция, nowcast, отчёт
```

Воспроизведение национального N01 в единой среде дало те же метрики (`tracking/reproduction_check_N01.csv`). Каждый запуск пишет свой каталог с `config.json`, SHA256 данных и `manifest.json`; предыдущие запуски не перезаписываются.

## Проверка вперёд (out-of-time)

- `prospective/forecast_2025_frozen.csv` — прогноз по МО на 2025 г. с интервалами и SHA256. Когда СберИндекс опубликует факты 2025 г. по МО: `python scripts/score_prospective.py`.
- `prospective/national_2026-09_frozen.csv` — национальные прогнозы и недельный nowcast на сентябрь 2026 (заморожены до выхода месячных данных): `python scripts/score_prospective_national.py`.
- Чтобы заморозка имела независимую временную метку, закоммитьте эти файлы до публикации новых данных.

## Структура

```
src/            mun_data / mun_models / mun_eval / mun_ensemble / mun_shocks / mun_news — муниципальный блок;
                forecasting / adaptive / transitions / evaluation — национальный блок и детекторы
scripts/        run_municipal, run_shocks, run_weekly_shocks, run_news_ablation, nowcast_weekly, mun_prophet,
                mun_foundation, check_2025_vs_national, make_figures, build_report ...
configs/        municipal.json, shocks.json, national_forecast.json, adaptive_forecast.json ...
tests/          test_municipal.py (утечки, метрики, детекторы), test_forecasting.py
data/external/  ставка ЦБ, реестр событий (с источниками)
```

## Данные и лицензия

Муниципальные расходы, справочник и индекс доступности рынков — СберИндекс, CC BY-SA 4.0 («Потребительские безналичные расходы на уровне муниципальных образований по категориям трат. СберИндекс», данные скачаны 04.10.2026, [описание](https://sberindex.ru/ru/research/data-sense-opisanie-nabora-dannikh-khakatona-sberindeksa-po-munitsipalnim-dannim)). Справочник границ: [СберИндекс](https://sberindex.ru/ru/research/dataset-borders-and-changes-of-municipalities), метаданные — `metadata_municipal_dict_sberindex_2.pdf`. Национальные и недельные выгрузки лежат в корне проекта. Ключевая ставка — [cbr.ru](https://www.cbr.ru/hd_base/KeyRate/).

Ограничения справочника (срезы на 1 января 2018–2024, центры по состоянию на 2018 г., нет координат городов федерального значения, Чечня) описаны в `configs/municipal_metadata.json` и `docs/MODELING_PLAN_RU.md`. Панель расходов покрывает 2 016 из 2 594 МО справочника (73 региона из 85).
