"""Обновлённое исследование и проверка критериев из реально рассчитанных артефактов."""
import base64
import html
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scripts.build_report import html_page, inline_md, SLIDE_CSS, slide, html_table


def table(df):
    return df.to_markdown(index=False,floatfmt=".3f",missingval="—")


def criteria(run):
    rubric=json.loads((ROOT/"configs/contest_criteria.json").read_text())
    evidence={
        "methodology":("выполнено","docs/METHODOLOGY_ONE_PAGE_RU.md; report/report.md; presentation/slides.pdf",
                       "краткое объяснение отделено от технического аудита; доступность месячных данных остаётся допущением"),
        "forecast":("есть; официальный протокол не подтверждён","metrics.csv; paired_comparisons.csv; prophet_same_subset.csv",
                    "новые сравнения после просмотра test; сезонный Prophet только 25 МО; официальный baseline отсутствует"),
        "detectors":("сравнение выполнено; выбор условный","docs/DETECTOR_CHOICE_RU.md; artifacts/shocks/injection_runs.csv; artifacts/weekly_shocks/; verified_event_detection.csv",
                     "CUSUM выбран для основного муниципального инъекционного опыта; на недельных proxy-метках лидирует PageHinkley; универсальный победитель не доказан"),
        "foundation":("есть, с ограничениями","docs/MODELS_AND_METRICS_RU.md; artifacts/municipal_runs/_foundation/*_meta.json",
                       "zero-shot; TimesFM на 200 МО; историческая доступность чекпойнтов не подтверждена"),
        "news":("частично — главный пробел","src/mun_news.py; artifacts/news_ablation/; data/inputs/verified_event_cases.csv; docs/NEWS_ALIGNMENT_RU.md",
                 "есть событийные признаки для прогноза и событийная диагностика детекторов; отдельный новостной вход детектора не проверен; недельные расходы не являются новостями; текстовый корпус сам по себе не обязателен"),
        "mae":("есть","metrics.csv; reproduction.csv",
                "руб./жителя; одинаковая сетка; итоговый способ агрегирования нужно сверить с правилами"),
        "reproduction":("есть, с ограничениями","requirements.lock.txt; tracking/research_verification.json; presentation/research_demo.html",
                        "проверена текущая среда и сохранённые метрики; установка с нуля, публичный релиз и внешняя временная отметка отсутствуют")}
    news_report=ROOT/"docs/REGIONAL_NEWS_IMPACT_RU.md"
    if news_report.exists():
        evidence["news"]=("интеграция проверена в обеих задачах; региональный пилот",
                          "docs/REGIONAL_NEWS_IMPACT_RU.md; configs/news_impact.json; artifacts/regional_news_runs/; src/news_impact.py",
                          "МЧС и датированный архив регионального СМИ; полнота всех новостей региона неизвестна; доступность по публикации — допущение; метки детекторов — финансовые proxy")
    rows=[]
    for c in rubric["criteria"]:
        status,files,lim=evidence.get(c["id"],("нужна проверка","—","нет сопоставления с новой формулировкой"))
        rows.append({"Критерий":c["name"],"Вес из предоставленной шкалы" if rubric["verified"] else "Вес из локального плана":c["weight"],
                     "Состояние":status,"Свидетельство":files,"Ограничение":lim})
    text=("# Проверка конкурсных критериев\n\n"+
          ("Критерии и веса подтверждены пользователем. Проценты — доли итоговой оценки, а не уже полученные баллы. Подробной шкалы качества внутри пунктов нет, поэтому баллы жюри не присваиваются. " if rubric["verified"] else
           "**Официальная шкала не подтверждена.** Веса взяты из локального плана; это не присвоенные баллы и не прогноз оценки жюри. ")+
          "Источник: "+rubric["source"]+".\n\n"+table(pd.DataFrame(rows))+
          "\n\nРезультаты доработок: `"+str(run.relative_to(ROOT))+"`. Не подтверждены настройки baseline и правила валидации. Отсутствие MAE обнулило бы её критерий; в проекте MAE рассчитана на всех четырёх горизонтах, поэтому этот риск устранён. Успешность новостей или foundation-моделей не требуется формулировками сама по себе: честный отрицательный результат допустим, но интеграция и корректное сравнение должны быть показаны. Раннее предупреждение не является отдельным обязательным условием приведённого пункта о детекторах.\n")
    (ROOT/"docs/CRITERIA_AUDIT_RU.md").write_text(text)
    return text


def build_text(run):
    m=pd.read_csv(run/"metrics.csv")
    test=m[m.stage=="test"].pivot(index="model",columns="h",values="mae")
    chosen=["Ensemble","DevBlend_NoHGB","EqualBlend_NoHGB","PastOnlyBlend","SeasonalNaive_NatGrowth","Prophet","RidgeSeasonal_DevSelected","WeeklyMarketplacePrior"]
    lb=test.loc[chosen].reset_index()
    lb["Среднее по h"]=lb[[1,3,6,12]].mean(axis=1)
    lb=lb.rename(columns={"model":"Модель",1:"h=1",3:"h=3",6:"h=6",12:"h=12"})
    boot=pd.read_csv(run/"paired_comparisons.csv")
    b=boot[(boot.model=="Ensemble")&(boot.comparator=="SeasonalNaive_NatGrowth")]
    ci=b[["h","mae_difference","month_ci_low","month_ci_high"]]
    interval=pd.read_csv(run/"past_only_intervals.csv").groupby("h")[["availability","coverage","relative_half_width"]].mean().reset_index()
    source=ROOT/json.loads((run/"manifest.json").read_text())["source_run"]
    shared_original=pd.read_csv(source/"intervals_test.csv")
    groups=pd.read_csv(run/"group_metrics.csv")
    market=groups[(groups.group=="category")&(groups.value=="Маркетплейсы")].groupby(["model","h"]).mae.mean().unstack()
    # Category reference computed directly from saved baseline on identical rows.
    from src.mun_data import load_municipal
    from src.mun_eval import make_grid,pair_errors
    panel,_=load_municipal(ROOT)
    ref=np.load(source/"pred_SeasonalNaive_NatGrowth.npy")
    g,y,p,_=pair_errors(panel,ref,make_grid(),"test")
    ids=(panel.meta.category=="Маркетплейсы").to_numpy()
    base_market=[np.abs(y[(g.h==h).to_numpy()][:,ids]-p[(g.h==h).to_numpy()][:,ids]).mean() for h in (1,3,6,12)]
    market.loc["SeasonalNaive_NatGrowth"]=base_market
    market_table=market.loc[["Ensemble","PastOnlyBlend","WeeklyMarketplacePrior","SeasonalNaive_NatGrowth"]].reset_index()
    market_table=market_table.rename(columns={"model":"Модель"})
    market_table.to_csv(run/"marketplace_comparison.csv",index=False)
    cases=pd.read_csv(run/"verified_event_matching.csv")
    details=pd.read_csv(run/"verified_event_detection.csv")
    event_summary=details.groupby(["event_id","method"]).agg(series=("detected","size"),detected_share=("detected","mean"),
                                  median_delay=("delay_months","median")).reset_index()
    action=pd.read_csv(run/"shock_action.csv")
    action_summary=action.pivot(index="method",columns="h",values="delta_mae").reset_index()
    economic=pd.read_csv(run/"economic_diagnostics.csv")
    econ=economic.groupby("category")[["mean_local_yoy_pct","median_local_yoy_pct","national_proxy_yoy_pct"]].mean().reset_index()
    gap=pd.read_csv(run/"2025_aggregation_sensitivity.csv").groupby("category")[["mean_mo_yoy_pct","ratio_of_sums_yoy_pct","national_actual_yoy_pct"]].mean().reset_index()
    wins=pd.read_csv(run/"win_shares.csv")
    win=wins[(wins.model.isin(["Ensemble","PastOnlyBlend"]))&(wins.comparator=="SeasonalNaive_NatGrowth")]
    category=groups[(groups.group=="category") & (groups.model.isin(["Ensemble","PastOnlyBlend"]))].groupby(["model","value"])[["mae","wape","median_ape"]].mean().reset_index()
    regional=groups[(groups.group=="region")&(groups.model=="Ensemble")].groupby("value")[["mae","wape","median_ape"]].mean().sort_values("wape",ascending=False).head(10).reset_index()
    selected_delta=(test.loc["DevBlend_NoHGB"].mean()/test.loc["Ensemble"].mean()-1)*100
    text=f"""# Перенос национальной динамики на короткие муниципальные ряды

Дополнение от 07.10.2026, запуск `{run.name}`. **Все эксперименты этого дополнения выполнены после просмотра теста 2024H2.** Это анализ устойчивости и новые исследовательские результаты; они не превращают просмотренный тест в независимый. Исходные прогнозы и файлы `prospective/` сохранены.

## Исследовательский вопрос и вклад

Можно ли прогнозировать расходы муниципалитета с 6–24 месяцами истории, перенося сезонность и рост из длинных национальных рядов? Гипотеза: основную пользу приносит общий национальный путь, а локальная нелинейная коррекция даёт небольшую прибавку. Проверяем MAE в рублях, WAPE, медианную относительную ошибку и долю выигравших территорий на общей сетке.

Это прикладная проверка переноса информации между уровнями, а не заявка на новый алгоритм ансамблирования или conformal prediction. Общие и локальные модели обсуждаются в [Montero-Manso и Hyndman](https://arxiv.org/abs/2008.00444). Наше отличие — явное разложение на муниципальный уровень и национальную траекторию, проверка доступности параметров и разбор короткой истории в этой панели. Полноценного обзора всех решений конкурса нет.

## Сильные базы и цена усложнения

MAE на просмотренном тесте, полная панель 2 016 МО × 6 категорий:

{table(lb)}

DevBlend_NoHGB: Prophet + SeasonalNaive_NatGrowth + NatPath_K3, LAD-веса на validation. EqualBlend_NoHGB: те же три модели с фиксированными равными весами. Прибавка исходного ансамбля с HGB относительно DevBlend_NoHGB по средней MAE — **{selected_delta:.2f}%**. Уникальный вклад HGB мал; дополнительная сложность пока не обоснована большим выигрышем.

NatPath_K1 имеет MAE {test.loc['NatPath_K1',1]:.1f} на h=1, ниже {test.loc['Ensemble',1]:.1f} у исходного ансамбля. Исходный ансамбль не является лучшим на каждом горизонте; выбирать победителя по текущему тесту для финального утверждения нельзя.

Разность MAE исходного ансамбля и сильного SeasonalNaive_NatGrowth, 95% bootstrap по шести целевым месяцам (отрицательное — лучше ансамбль):

{table(ci)}

На h=3 и h=12 интервалы включают ноль. Шесть месяцев недостаточны для уверенного вывода об устойчивости по времени; пространственный размер панели не добавляет независимых временных режимов. Bootstrap по месяцам здесь диагностический: временная зависимость и малое число месяцев ограничивают его интерпретацию.

Явно сезонный Prophet проверен с Fourier order=2, двумя вариантами регуляризации, выбранными по validation. Все модели сравнены на **одних и тех же 25 случайных МО / 150 рядах**, seed=42; выполнено 5 700 подгонок. Это ограниченное исследование чувствительности, не официальный baseline и не полный поиск параметров.

"""
    if (run/"prophet_same_subset.csv").exists():
        subset=pd.read_csv(run/"prophet_same_subset.csv")
        text+=table(subset[subset.stage=="test"].pivot(index="model",columns="h",values="mae").reset_index())+"\n\n"
    text+=f"""## Доступность параметров во времени

Исходные веса выбирались по целям до июня 2024 включительно. На h=12 прогнозы тестовых целей июля–декабря 2024 имеют origin июль–декабрь 2023: выбранные позднее веса тогда были недоступны. Это ретроспективная проверка метода, отделённая от численного as-of backtest.

PastOnlyBlend подбирает веса только по ошибкам с target ≤ origin−2; последние два зрелых месяца оставлены калибровке. При менее двух месяцев подбора используются фиксированные равные веса. Для h=12 на данном тесте нет зрелых меток: используется именно fallback, а не параметры, обученные на будущих фактах. Случайная подвыборка подбора ограничена 3 000 строками, seed=42. Модели в этом сравнении не используют статические признаки среза 2024.

Сохраняются: `past_only_selection_audit.csv`, `calibration_audit.csv`, `original_selection_time_audit.csv`. Ограничения: последняя версия данных, неизвестные задержки публикаций месячных значений и выбор полных рядов по всей истории. «As-of» относится к численным входам и весам при допущении доступности месяца origin; историческую доступность современного кода или винтажей оно не доказывает.

## Интервалы: калибровка отделена от подбора

Последовательная калибровка использует ошибки ранее выданных прогнозов; её целевые месяцы исключены из текущего подбора весов. Из-за коррелированных территорий, короткой истории и изменения распределения гарантии обменного split-conformal не заявляются. [Adaptive Conformal Inference](https://arxiv.org/abs/2106.00170) показывает важность отдельного рассмотрения изменения распределения; адаптивный алгоритм этой статьи здесь не реализован.

Среднее эмпирическое покрытие по категориям и месяцам:

{table(interval)}

На h=1 и h=3 есть недопокрытие относительно 90%. На h=12 интервал недоступен: нулевую доступность нельзя заменять высоким покрытием. Полный разрез по категориям и месяцам — `past_only_intervals.csv`.

Исходные интервалы были рассчитаны по той же validation, где выбирались веса. Их старые значения сохранены как эмпирическая диагностика; стандартная гарантия split-conformal к ним неприменима.

## Внешние данные: маркетплейсы и инфляция

В имеющейся выгрузке найден недельный рост расходов на маркетплейсах с 12.11.2023. Новый prior использует медиану доступных недель последних 13 недель (минимум 4) и расход МО того же месяца год назад. Доступность недели = период + 7 дней; неизвестный или недоступный prior заменяется SeasonalNaive_NatGrowth. Чужая будущая неделя не используется. Для остальных категорий прогноз базы не меняется.

MAE только по маркетплейсам, одна сетка:

{table(market_table)}

Источник категории «Непродовольственные товары» слишком широк для маркетплейсов; недельная категория совпадает по названию, но равенство транзакционного охвата не подтверждено. Улучшение относительно прежнего сезонного правила есть на h=1/3/6, на h=12 его нет; замена всего исходного ансамбля этим правилом не обоснована. Такой prior теперь можно проверять на следующем независимом периоде.

Для интерпретации добавлен национальный ИПЦ декабря 2024 к декабрю 2023 из [Росстата](https://rosstat.gov.ru/storage/mediabank/1_15-01-2025.html), опубликованный 15.01.2025. Он не подаётся в исторические прогнозы 2024 года.

{table(pd.read_csv(run/'inflation_interpretation.csv').drop(columns='note'))}

Дефлированный показатель — диагностический proxy. Он не отделяет объём потребления от изменений структуры корзины, охвата платежей и состава клиентов. Для общепита, здоровья и маркетплейсов соответствующий точный дефлятор не найден; широкие индексы не подставляются как эквивалент.

## Экономическая интерпретация и устойчивость

Средний г/г рост по месяцам 2024: национальный proxy и муниципальная панель отличаются; соответствие категорий является допущением.

{table(econ)}

Прогноз 2025: чувствительность к агрегированию (среднее по 12 месяцам):

{table(gap)}

По общепиту переход от среднего роста МО к отношению сумм почти не устраняет разрыв с национальным фактом. **Различие весов этих двух способов не объясняет расхождение.** Причина остаётся неустановленной: нужны проверка определений категорий, охвата и возможных изменений измерения. Отношение сумм расходов на жителя не является суммой денежных расходов страны: для неё нужны веса населения.

Доля МО с улучшением против сильной сезонной базы:

{table(win.drop(columns=['series_win_share','median_series_relative_gain']))}

Категории: MAE, WAPE и медианная относительная ошибка; значения усреднены по месяцам и горизонтам:

{table(category)}

Десять регионов с наибольшим средним WAPE исходного ансамбля:

{table(regional)}

Результаты относятся к полной наблюдаемой когорте, а не ко всем МО России. В «Все категории» входят расходы подкатегорий; средняя метрика по шести категориям не является суммой экономической полезности. Итоговые веса категорий следует согласовать с регламентом.

## Официальные события и действие после тревоги

Добавлены три кейса с подтверждающими официальными публикациями. Дата события, дата публикации и дата проверки сохранены отдельно. Это внешние события, **не независимые подтверждённые метки сдвига расходов**.

{table(cases[['event_id','title','matched_territories','evaluable','period_role','reason']])}

{table(event_summary)}

Два оценимых события относятся к периоду настройки порогов: это описательные кейсы, а не независимый событийный тест. Уссурийск сопоставлен, но история для стандартизации слишком коротка. BOCPD реагирует на одну из шести категорий Орска; реакция остальных методов и на Красногорск слабая. Отсутствие тревоги не доказывает отсутствие экономического воздействия.

Добавлена база AbsZ: порог абсолютного стандартизованного остатка, подобранный по той же калибровке при бюджете 1 тревога на 100 ряд-месяцев; cooldown=3 месяца. Тревоги без события в коротком реестре сохранены в `unmatched_alarms.csv`; достоверно назвать их ложными нельзя.

Проверена связка «тревога → коррекция будущего прогноза». После сигнала используем только доступный остаток, сжатие 0.5, ограничение log-коррекции ±0.3 и затухание за 3 месяца. Параметры заданы для этого исследовательского опыта, по тесту не оптимизировались. Разность MAE относительно неизменённого NatPath_K3 (отрицательное — лучше):

{table(action_summary)}

Эффект небольшой и зависит от горизонта; раннее предупреждение будущего события этим не доказано. Действие начинается после наблюдения сигнала. Сравнение не заменяет существующую оценку детекторов на инъекциях и недельных proxy-метках.

## Сценарий использования

Пользователь — региональный аналитик спроса. Он выбирает МО и категорию, смотрит прогноз расходов и диапазон неопределённости, сопоставляет отклонение с общим национальным ростом. При тревоге проверяет качество данных и официальные события; решение о коррекции принимает после проверки. Возможный результат — приоритет территории для ручного анализа. Экономия бюджета или эффект для бизнеса не измерены и не заявляются.

Демонстрация: `presentation/research_demo.html`, без сервера и внешних библиотек. Три реальных кейса показывают факт, прогноз до наблюдения и тревоги; отдельная таблица показывает замороженный ретроспективный прогноз 2025, фактов МО за этот год нет.

## Воспроизводимость и границы готовности

Полный снимок среды в `requirements.lock.txt`; `environment.yml` теперь использует его. HiGHS и BLAS ограничены одним потоком; команды Make используют пониженный приоритет. Исходный запуск воспроизведён по сохранённым прогнозам. `make verify` проверяет версии пакетов, SHA256 входов, доступность недельных данных, зрелость/разделение меток, сохранность файлов `prospective/` и совпадение исходных метрик.

```bash
conda env create -f environment.yml
conda activate sber
make test
make research-prophet   # 1 worker; совместимый кэш используется повторно
make research
make verify
make report
```

Перед research нужны исходные данные и прогнозы source_run. Полное воспроизведение с нуля: `make all`, затем команды выше; оно существенно дороже лёгкого повторного анализа. Установка зависимостей в новой среде с нуля и независимая внешняя временная отметка пока не подтверждены. Просмотренный тест не используется для заявления нового независимого выигрыша.

Критерии и веса подтверждены пользователем 07.10.2026. Точный baseline и правила валидации ещё не предоставлены; подробной шкалы качества внутри критериев нет. Итоговое сопоставление — `docs/CRITERIA_AUDIT_RU.md`; баллы жюри и численный шанс на приз не оцениваются.
"""
    regional=ROOT/"docs/REGIONAL_NEWS_IMPACT_RU.md"
    if regional.exists():
        text += "\n\n---\n\n" + regional.read_text()
    return text,lb,interval,market_table,action_summary


def demo(run):
    data=pd.read_csv(run/"case_series.csv")
    source=ROOT/json.loads((run/"manifest.json").read_text())["source_run"]
    forecasts=pd.read_csv(source/"forecast_2025.csv")
    items=[]
    for case,part in data.groupby("case"):
        first=part.iloc[0]
        f=forecasts[(forecasts.territory_id==first.territory_id)&(forecasts.category==first.category)]
        items.append({"name":f'{first.case_kind} · МО {first.territory_id} · {first.category}',
                      "event":first.event_date,"method":first.method,
                      "points":json.loads(part[["month","actual","forecast_h1","z","alarm"]].to_json(orient="records")),
                      "future":json.loads(f[["target_month","y_pred_ensemble","lo90","hi90"]].to_json(orient="records"))})
    page='''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Муниципальные расходы — исследовательская демонстрация</title>
<style>body{font:17px/1.5 system-ui;margin:0;background:#f4f7f9;color:#172b3a}main{max-width:1050px;margin:auto;padding:28px}select{font:inherit;max-width:100%;padding:10px}svg{width:100%;background:white;border-radius:12px}table{border-collapse:collapse;width:100%;background:white}td,th{padding:7px;border-bottom:1px solid #dae1e6;text-align:right}td:first-child,th:first-child{text-align:left}.note{color:#50616c}h1{font-size:29px}</style>
<main><h1>Расходы муниципалитета: прогноз и проверка тревог</h1><p>Сценарий: региональный аналитик выбирает территорию для ручной проверки спроса.</p>
<label for="case">Пример</label> <select id="case"></select><p id="note" class="note"></p>
<svg id="chart" viewBox="0 0 1000 370" role="img" aria-label="Факт и прогноз расходов"></svg>
<p><span style="color:#1474a4">● Факт</span> · <span style="color:#e28919">● Прогноз h=1 до наблюдения</span> · <span style="color:#c33845">● Тревога</span> · пунктир: дата внешнего события</p>
<h2>Прогноз 2025, замороженный в 2026 году</h2><p class="note">Ретроспективный прогноз по истории до декабря 2024. Фактов 2025 по МО нет. Диапазоны исходного запуска калибровались совместно с подбором весов и не имеют стандартной гарантии conformal.</p>
<table><thead><tr><th>Месяц</th><th>Прогноз, руб.</th><th>Нижняя граница</th><th>Верхняя граница</th></tr></thead><tbody id="forecast"></tbody></table>
<h2>Что делать при сигнале</h2><p>Проверить данные и официальные сообщения; затем решить, требуется ли ручная корректировка прогноза. Тревога без события в коротком реестре не является доказанной ложной тревогой.</p>
<p class="note">Все новые эксперименты выполнены после просмотра теста. Демонстрация показывает исследовательские результаты, а не подтверждённую эффективность сервиса.</p></main>
<script>const cases=__DATA__;const select=document.getElementById('case');cases.forEach((c,i)=>{const o=document.createElement('option');o.value=i;o.textContent=c.name;select.append(o)});
function render(){const c=cases[Number(select.value)],p=c.points,W=1000,H=370,L=75,R=30,T=25,B=45;
const vals=p.flatMap(x=>[x.actual,x.forecast_h1]).filter(x=>x!==null),max=Math.max(...vals)*1.08,min=Math.min(...vals)*.9;
const x=i=>L+i*(W-L-R)/(p.length-1),y=v=>T+(max-v)*(H-T-B)/(max-min);let svg='';
for(let i=0;i<5;i++){const v=min+i*(max-min)/4;svg+=`<line x1="${L}" y1="${y(v)}" x2="${W-R}" y2="${y(v)}" stroke="#e4eaef"/><text x="${L-8}" y="${y(v)+5}" text-anchor="end" font-size="14">${Math.round(v)}</text>`}
for(const [key,color] of [['actual','#1474a4'],['forecast_h1','#e28919']]){let path='',pen=false;p.forEach((d,i)=>{if(d[key]===null){pen=false;return}path+=(pen?'L':'M')+x(i)+','+y(d[key]);pen=true});svg+=`<path d="${path}" fill="none" stroke="${color}" stroke-width="2.5"/>`}
p.forEach((d,i)=>{if(d.alarm)svg+=`<circle cx="${x(i)}" cy="${y(d.actual)}" r="5" fill="#c33845"/>`;if(i%4===0)svg+=`<text x="${x(i)}" y="${H-14}" text-anchor="middle" font-size="13">${d.month.slice(0,7)}</text>`});
const eventIndex=p.findIndex(d=>d.month.slice(0,7)===c.event.slice(0,7));if(eventIndex>=0)svg+=`<line x1="${x(eventIndex)}" x2="${x(eventIndex)}" y1="${T}" y2="${H-B}" stroke="#4e555b" stroke-dasharray="6 6"/>`;
document.getElementById('chart').innerHTML=svg;document.getElementById('note').textContent='Детектор: '+c.method+'. Внешнее событие: '+c.event.slice(0,10)+'. Месячный сигнал обнаруживает наблюдённое отклонение; раннее предупреждение не подтверждено.';
const tbody=document.getElementById('forecast');tbody.replaceChildren();c.future.forEach(d=>{const tr=document.createElement('tr');[d.target_month.slice(0,7),...['y_pred_ensemble','lo90','hi90'].map(k=>Math.round(d[k]).toLocaleString('ru-RU'))].forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.append(td)});tbody.append(tr)})}
select.addEventListener('change',render);render();</script></html>'''
    payload=json.dumps(items,ensure_ascii=False).replace("<","\\u003c")
    (ROOT/"presentation/research_demo.html").write_text(page.replace("__DATA__",payload))


def main():
    run=ROOT/json.loads((ROOT/"artifacts/research_runs/latest.json").read_text())["directory"]
    text,lb,interval,market,action=build_text(run)
    crit=criteria(run)
    (ROOT/"docs/RESEARCH_AUDIT_RESULTS_RU.md").write_text(text)
    (ROOT/"report/research_audit.md").write_text(text+"\n\n"+crit)
    body=inline_md(text+"\n\n"+crit)
    for path in sorted((run/"figures").glob("case_*.png")):
        body+='<figure><img src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'" alt="Реальный событийный кейс"></figure>'
    (ROOT/"report/research_audit.html").write_text(html_page("Дополнение к исследованию",body))
    pieces=[
        slide('<h2>Исследовательская гипотеза</h2><p>Длинная национальная история переносит сезонность и рост на короткие муниципальные ряды.</p><p>Проверяем вклад локальной коррекции, доступность параметров и реакцию после тревог.</p><p class="warn">Доработки после просмотра теста: новые результаты исследовательские.</p>',15),
        slide('<h2>Сильные базы и вклад HGB</h2>'+html_table(lb.head(6))+'<p>Без HGB средняя ошибка почти та же. На h=3/12 превосходство над сильной сезонной базой по bootstrap месяцев не подтверждено.</p>',16),
        slide('<h2>Последовательные веса и интервалы</h2><p>Подбор: target ≤ origin−2. Калибровка: последние два зрелых месяца.</p>'+html_table(interval)+'<p>На h=12 нет истории для подбора и калибровки. Фиксированные веса; интервалы недоступны.</p>',17),
        slide('<h2>Данные по маркетплейсам</h2>'+html_table(market)+'<p>Недельный prior помогает относительно прежнего сезонного правила на h=1/3/6. На h=12 улучшения нет.</p>',18),
        slide('<h2>Детектор → действие</h2><p>Разность MAE после коррекции к NatPath_K3; отрицательная лучше.</p>'+html_table(action)+'<p>Эффект невелик. Два оценимых официальных события относятся к калибровке. Раннее предупреждение не подтверждено.</p>',19),
        slide('<h2>Готовность к конкурсной проверке</h2><ul><li>Критерии и веса подтверждены пользователем</li><li>Метрики, временной аудит, категории и регионы</li><li>Фиксированные версии и реальные кейсы</li><li>Новостной пилот: обе задачи; эффект неустойчив</li><li>Нужны настройки baseline и новые факты</li></ul>',20)]
    if (ROOT/"artifacts/regional_news_runs/latest.json").exists():
        nr=ROOT/json.loads((ROOT/"artifacts/regional_news_runs/latest.json").read_text())["directory"]
        nmanifest=json.loads((nr/"manifest.json").read_text())
        sources=', '.join(f'{name}: {count}' for name,count in nmanifest.get('news_sources',{}).items())
        nm=pd.read_csv(nr/"forecast_metrics.csv").query("stage=='test' and model in ['without_news','with_news']")
        nt=nm.pivot(index="h",columns="model",values="mae").reset_index()
        pieces.append(slide('<h2>Региональные новости: прогноз</h2><p>Оренбургская область; '+sources+'.</p>'+html_table(nt)+'<p>Одинаковая Ridge-модель и сетка прогноза; MAE, руб./жителя.</p><p class="warn">Тест уже просмотрен; пилот исследовательский. Темы сравниваются с контролем объёма публикаций.</p>',21))
        nd=pd.read_csv(nr/"detector_metrics.csv").query("proxy_penalty==4 and variant in ['without_news','with_news']")
        nt=nd.pivot(index="method",columns="variant",values="proxy_f1").reset_index()
        pieces.append(slide('<h2>Региональные новости: детекторы</h2>'+html_table(nt,floatfmt="{:.3f}")+'<p>Одинаковый калибровочный бюджет тревог. Ориентиры построены только по финансовым данным, без новостей.</p><p>Сравнение чувствительно к штрафу proxy-разметки; полный отчёт включает 7 вариантов.</p><p class="warn">Реальная экспертная разметка отсутствует; основной детектор сохраняется.</p>',22))
        allnews=pd.read_csv(nr/"forecast_metrics.csv").query("stage=='test'").pivot(index="model",columns="h",values="mae").reset_index()
        pieces.append(slide('<h2>ЧС и экономика: отдельные проверки</h2>'+html_table(allnews)+'<p>crisis_only и economic_only имеют одинаковые признаки покрытия. mchs_only использует прежнюю поисковую выборку.</p>',23))
        pieces.append(slide('<h2>Как получать дополнительные признаки</h2><ul><li>requests → JSON-архив; BeautifulSoup очищает заголовки</li><li>Даты GMT, кеш страниц, проверка полноты пагинации</li><li>10 тем: ЧС, цены, доходы, выплаты, торговля, предприятия, транспорт/жильё</li><li>Счётчики за 1/3 месяца; направления цен, доходов, открытия/закрытия бизнеса</li><li>Явные география и прошлый cutoff; федеральные перепечатки без привязки дают только покрытие</li></ul><p>Заголовки и правила, без браузера и LLM. Новостные метки не служат эталоном шоков.</p>',24))
    fragment="".join(pieces)
    (ROOT/"presentation/research_slides.fragment.html").write_text(fragment)
    (ROOT/"presentation/research_slides.html").write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><style>'+SLIDE_CSS+'</style><body>'+fragment+'</body></html>')
    demo(run)
    status={"run":str(run.relative_to(ROOT)),"completed":["strong_baselines","past_only_weights","separate_calibration", "weekly_marketplace_prior",
             "inflation_interpretation","official_event_cases","shock_action","economic_diagnostics","criteria_evidence","locked_environment","interactive_demo"],
            "pending_external":["official_baseline","unseen_municipal_facts","publication_destination"],
            "not_verified":["fresh_dependency_installation","external_timestamp","independent_real_shock_labels"]}
    completion_path=ROOT/"tracking/research_completion.json"
    if completion_path.exists():
        previous=json.loads(completion_path.read_text())
        previous.update(status)
        status=previous
    status["official_rubric_confirmed"]=json.loads((ROOT/"configs/contest_criteria.json").read_text())["verified"]
    if (ROOT/"docs/REGIONAL_NEWS_IMPACT_RU.md").exists():
        status["completed"] += ["regional_news_collection", "regional_news_forecast_and_detector_comparison"]
        status["regional_news_run"]=json.loads((ROOT/"artifacts/regional_news_runs/latest.json").read_text())["directory"]
    completion_path.write_text(json.dumps(status,ensure_ascii=False,indent=2))
    print("Отчёт, критерии, слайды и демонстрация сформированы:",run)


if __name__=="__main__":
    main()
