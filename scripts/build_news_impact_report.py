"""Отчёт о влиянии региональных новостей из сохранённых результатов эксперимента."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/sber-research-mpl')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import sys
import json
import hashlib
import base64
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.build_report import html_page, inline_md


def main():
    run=ROOT/json.loads((ROOT/'artifacts/regional_news_runs/latest.json').read_text())['directory']
    manifest=json.loads((run/'manifest.json').read_text())
    if manifest['status']!='complete':
        raise ValueError('Последний запуск не завершён')
    cfg=json.loads((run/'config.json').read_text())
    metrics=pd.read_csv(run/'forecast_metrics.csv')
    bench=pd.read_csv(run/'cached_benchmark_metrics.csv')
    test=metrics[metrics.stage=='test'].pivot(index='model',columns='h',values='mae')
    dev=metrics[metrics.stage=='validation'].pivot(index='model',columns='h',values='mae')
    comparison=pd.DataFrame({'Горизонт, мес':test.columns,'MAE без новостей':test.loc['without_news'].values,
                              'MAE с новостями':test.loc['with_news'].values,
                              'Изменение MAE, %':100*(test.loc['with_news'].values/test.loc['without_news'].values-1)})
    boot=pd.read_csv(run/'forecast_bootstrap.csv')
    controls=pd.read_csv(run/'news_vs_coverage_bootstrap.csv')
    d=pd.read_csv(run/'detector_metrics.csv')
    sel=pd.read_csv(run/'detector_selection.csv')
    sensitivity=d[d.variant.isin(['without_news','with_news'])].pivot(index=['method','proxy_penalty','n_proxy_events'],columns='variant',values='proxy_f1').reset_index()
    principal=d[(d.proxy_penalty==4)&d.variant.isin(['without_news','with_news'])][['method','variant','n_proxy_events','n_alarms','matched_proxy_events','proxy_precision','proxy_recall','proxy_f1','median_delay_months']]
    selection=[]
    for h in [1,3,6,12]:
        name='with_news' if dev.loc['with_news',h]<dev.loc['without_news',h]-1e-9 else 'without_news'
        selection.append({'h':h,'selected_on_validation':name,'validation_mae_without_news':dev.loc['without_news',h],
                          'validation_mae_with_news':dev.loc['with_news',h],'test_mae_selected':test.loc[name,h]})
    selection=pd.DataFrame(selection)
    selection.to_csv(run/'forecast_validation_selection.csv',index=False)
    audit=pd.read_csv(run/'news_availability_audit.csv')
    latest=pd.to_datetime(audit.latest_available_at,utc=True)
    end=pd.to_datetime(audit.cutoff_exclusive,utc=True)
    training=pd.read_csv(run/'forecast_training_audit.csv')
    checks={'training_targets_not_after_origin':bool((training.maximum_training_target<=training.origin).all()),
            'news_available_before_cutoff':bool((latest[latest.notna()]<end[latest.notna()]).all()),
            'all_calibration_alarm_rates_within_budget':bool((sel.calibration_alarm_rate_per_100<=cfg['detectors']['alarm_budget_per_100_series_months']).all()),
            'all_forecast_metric_rows_full_coverage':bool((metrics.coverage==1).all()),
            'input_hashes_match':all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value for name,value in manifest['inputs_sha256'].items()),
            'cached_benchmark_hashes_match':all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value for name,value in manifest['cached_benchmark_inputs'].items()),
            'code_hashes_match':all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value for name,value in manifest['code_sha256'].items())}
    if not all(checks.values()):
        raise AssertionError(checks)
    (run/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    figures=run/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10})
    fig,ax=plt.subplots(figsize=(8,4))
    positions=np.arange(4);width=.24
    for shift,name,label in [(-1,'without_news','Без новостей'),(0,'with_news','С новостями'),(1,'coverage_only','Только объём публикаций')]:
        ax.bar(positions+shift*width,test.loc[name,[1,3,6,12]],width,label=label)
    ax.set_xticks(positions,['1','3','6','12']);ax.set_xlabel('Горизонт, месяцев');ax.set_ylabel('MAE, руб./жителя');ax.legend(fontsize=8)
    ax.set_title('Оренбургская область: одинаковая Ridge-модель, тест 07–12.2024')
    fig.tight_layout();fig.savefig(figures/'forecast_news_mae.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(10,3.5),sharey=True)
    for ax,method in zip(axes,cfg['detectors']['methods']):
        for name,label in [('without_news','Без новостей'),('with_news','С новостями')]:
            sub=sensitivity[sensitivity.method==method]
            ax.plot(sub.proxy_penalty,sub[name],marker='o',label=label)
        ax.set_title(method);ax.set_xlabel('Штраф proxy-разметки');ax.set_xticks([2,4,8]);ax.set_ylim(0,.65)
    axes[0].set_ylabel('F1 относительно финансовых proxy-меток');axes[-1].legend(fontsize=8)
    fig.suptitle('Эффект зависит от разметки; реальная точность не установлена');fig.tight_layout();fig.savefig(figures/'detector_news_sensitivity.png',dpi=150);plt.close(fig)
    changes=pd.read_csv(run/'detector_changed_alarms.csv')
    cases=changes[(changes.month_index>=18)&(changes.news_signal>0)].sort_values(['method','territory_id','category']).head(2)
    alarms=pd.read_csv(run/'detector_alarms.csv')
    if len(cases):
        fig,axes=plt.subplots(len(cases),1,figsize=(9,3*len(cases)),squeeze=False)
        for ax,row in zip(axes[:,0],cases.itertuples()):
            hits=alarms[(alarms.method==row.method)&(alarms.territory_id==row.territory_id)&(alarms.category==row.category)&(alarms.month_index>=18)]
            for v,y,label in [('without_news',1,'Без новостей'),('with_news',2,'С новостями')]:
                sub=hits[hits.variant==v]
                ax.scatter(pd.to_datetime(sub.month),np.full(len(sub),y),label=label,s=60)
            ax.axvline(pd.Timestamp(row.month),color='gray',ls='--',label='Месяц новостного сигнала')
            ax.set_xlim(pd.Timestamp('2024-06-15'),pd.Timestamp('2025-01-15'));ax.set_ylim(.5,2.5);ax.set_yticks([1,2],['Без новостей','С новостями'])
            ax.set_title(f'{row.method}: {row.mo_name}, {row.category}');ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(figures/'changed_alarm_cases.png',dpi=150);plt.close(fig)
    table=lambda x:x.to_markdown(index=False,floatfmt='.3f',missingval='—')
    fulltest=pd.concat([metrics[metrics.stage=='test'],bench[bench.stage=='test']]).pivot(index='model',columns='h',values='mae').reset_index()
    news=pd.read_csv(ROOT/cfg['news_file'],dtype={'territory_ids':'string'})
    source_counts=news.groupby('source').size().reset_index(name='Публикаций')
    nmonths=pd.to_datetime(news.published_at,utc=True).dt.tz_convert('Europe/Moscow').dt.strftime('%Y-%m').nunique()
    improved=[str(int(h)) for h in test.columns if test.loc['with_news',h]<test.loc['without_news',h]]
    worse=[str(int(h)) for h in test.columns if test.loc['with_news',h]>test.loc['without_news',h]]
    conclusion=f"Новостной вариант снижает MAE относительно without_news на горизонтах {', '.join(improved) or 'ни одном'}; повышает на {', '.join(worse) or 'ни одном'}."
    ensemble_better=bool((test.loc['with_news']>bench.query("stage=='test' and model=='CachedEnsemble'").set_index('h').mae).all())
    benchmark_note='Сохранённый ансамбль точнее with_news на всех горизонтах.' if ensemble_better else 'Сравнение с сохранённым ансамблем приведено по каждому горизонту ниже.'
    selected_h=[str(int(r.h)) for r in selection.itertuples() if r.selected_on_validation=='with_news']
    control_h=[str(int(r.h)) for r in controls.itertuples() if r.month_ci_high<0]
    control_note='Месячный интервал разности news−coverage целиком ниже нуля на h='+','.join(control_h)+'.' if control_h else 'Ни на одном горизонте месячный интервал news−coverage не лежит целиком ниже нуля; отдельный вклад тем сверх покрытия не подтверждён.'
    coefficients=pd.read_csv(run/'news_coefficients.csv') if (run/'news_coefficients.csv').exists() else pd.DataFrame()
    coefficient_table=''
    detector_signal_note=''
    if (run/'detector_news_signal_audit.csv').exists():
        signal_audit=pd.read_csv(run/'detector_news_signal_audit.csv')
        signal_audit=signal_audit[pd.to_datetime(signal_audit.month)>='2024-07-01']
        signal_summary=signal_audit.groupby('variant').agg(mean_fraction_series_with_signal=('fraction_series_with_signal','mean'),months_same_as_coverage=('same_as_coverage','sum')).reset_index()
        detector_signal_note='\n### Насыщение новостного входа\n\n'+table(signal_summary)+'\n\nПри большом корпусе правило «есть хотя бы одна тематическая новость» может включаться во всех ряд-месяцах. Тогда новостной вариант лишь постоянно масштабирует финансовый остаток; изменение тревог объясняется масштабом и дискретной сеткой порогов. Совпадение с coverage_only означает, что дополнительная польза содержания новостей для детектора не показана. Следующий содержательный вариант должен проверять локальную интенсивность или новизну тем на новых данных; правило после просмотра теста здесь не подбиралось.\n'
    if len(coefficients):
        schema=json.loads((run/'news_feature_schema.json').read_text())
        cs=coefficients.query("variant=='with_news' and origin<=17").copy()
        cs['feature']=cs.feature_index.map(lambda i:schema['with_news'][i])
        cs['abs_coefficient']=cs.standardized_coefficient.abs()
        top=cs.groupby('feature').agg(mean_abs_coefficient=('abs_coefficient','mean'),mean_signed_coefficient=('standardized_coefficient','mean')).sort_values('mean_abs_coefficient',ascending=False).head(12).reset_index()
        top.to_csv(run/'news_top_coefficients_development.csv',index=False)
        coefficient_table='\n## Какие признаки использовала модель\n\n'+table(top)+'\n\nСредние коэффициенты Ridge после стандартизации на origin≤17, до тестового периода. Это диагностика модели, не причинный эффект: темы коррелируют между собой и со временем.\n'
    text=f'''# Влияние региональных новостей на прогнозы и детекторы

Пилот: Оренбургская область, {manifest['n_municipalities']} МО, {manifest['n_series']} ряда, 24 месяца (2023–2024). Корпус — {len(news)} заголовков, {len(source_counts)} источника; публикации найдены в {nmonths} из 24 месяцев. Запуск: `{run.relative_to(ROOT)}`. Расчёт занял {manifest['elapsed_seconds']:.1f} секунды, один вычислительный поток; обучение фундаментальных моделей не повторялось.

**Вывод:** {conclusion} {benchmark_note} Результат исследовательский; детекторные сравнения чувствительны к финансовой proxy-разметке.

{table(source_counts)}

## Протокол

Конфигурация — configs/news_impact.json. Обучение прогнозов: расширяющееся окно, метки только до origin включительно. Валидация — цели 01–06.2024, тест — 07–12.2024; горизонты 1/3/6/12. Ridge alpha=1000, shrink=0.25 и ограничение коррекции ±0.3 log заданы одинаково для вариантов до запуска. Категории кодируются индикаторами; заполнение пропусков и масштабирование обучаются только на прошлых строках. Будущий горизонт не усиливает дрейф дальше максимального обученного горизонта.

Прогнозная база — национальный сезонный путь NatPath_K3; Ridge оценивает муниципальное отклонение. Сравниваются одинаковые модели: without_news, coverage_only (объём по источникам), with_news (все признаки), delayed_news_3m (задержка на 3 месяца), mchs_only (исходный корпус), crisis_only (ЧС из расширенного корпуса), economic_only (экономические темы). Последние два варианта сохраняют одинаковые признаки покрытия, изолируя темы. Задержанный контроль не использует будущие публикации раньше оригинала; он не является доказательством причинности.

Новостные признаки: log(1+количество) за последние 1 и 3 календарных месяца, по источникам, десяти темам и известному МО. Четыре темы ЧС дополнены ценами, доходами/работой, выплатами, торговлей/услугами, предприятиями, транспортом/жильём. Пять признаков направления: рост/снижение цен, рост доходов, открытие/закрытие бизнеса. Всего {manifest.get('feature_count','см. схему')} признаков; имена — news_feature_schema.json. Правила фиксированы без подбора по тесту. Материалы с неподтверждённой географией не создают тематический сигнал. При известном МО событие не переносится на все МО области. Заголовок может иметь несколько тем; упоминание суммы не превращается в финансовый показатель.

Для детекторов используется сохранённый финансовый стандартизованный остаток z, рассчитанный по прошлой истории. Новостной вариант усиливает z в 1.5 раза только в месяц доступной тематической публикации соответствующей географии; нулевой финансовый остаток не превращается в тревогу. Пороги каждого варианта выбираются только на калибровке 11.2023–06.2024 при одном максимальном бюджете 1 тревога на 100 ряд-месяцев. Это бюджет общего количества тревог, а не доказанных ложных тревог. Реализованный бюджет может различаться из-за дискретной сетки. Тестовые частоты не обязаны соблюдать калибровочный бюджет.

{detector_signal_note}

## Прогнозы: основной контроль

Меньшая MAE лучше. Изменение положительное означает ухудшение.

{table(comparison)}

Валидационные MAE:

{table(dev.reset_index())}

Выбор между without_news и with_news по валидации допускает новости на h={','.join(selected_h) or 'ни одном'}; при равенстве выбирает without_news. Это исследовательский выбор внутри пилота, а не новая глобальная модель проекта.

{table(selection)}

Bootstrap по шести целевым месяцам, delta=MAE варианта − MAE без новостей:

{table(boot[boot.variant=='with_news'])}

{control_note} Сравнения множественные и исследовательские. Узкие интервалы по МО не устраняют зависимость от общих региональных месяцев; шестимесячный тест не подтверждает устойчивость эффекта.

{table(controls)}

Все новые варианты и сохранённые ориентиры на тех же рядах:

{table(fulltest)}

CachedEnsemble/CachedProphet — ранее сохранённые прогнозы, они не переобучались и не являются парной абляцией архитектуры Ridge. Их ретроспективные ограничения описаны в основном аудите. {benchmark_note}

{coefficient_table}

## Детекторы

Независимая от новостей ориентировочная разметка: DP-сегментация финансового z, минимальная длина сегмента 4, минимальный сдвиг 1.5σ. Проверены штрафы 2/4/8·log(T). Сами эти метки используют весь ряд и не являются доказанными реальными шоками. Новости не используются для формирования эталона. Совпадения считаются один-к-одному, задержка до 3 месяцев, только события и тревоги тестового периода.

Основной срез штрафа 4:

{table(principal)}

Уменьшение числа тревог само по себе не доказывает рост качества. Оценивать нужно precision, recall и чувствительность к варианту разметки совместно.

Чувствительность F1 к штрафу разметки:

{table(sensitivity)}

Полная таблица всех тематических контролей и методов сохранена в detector_metrics.csv. Универсального победителя по одному выбранному штрафу не устанавливаем.

Выбранные пороги и фактические частоты:

{table(sel)}

Изменившиеся тревоги могут появляться и позже месяца новости из-за состояния детектора, cooldown и нового порога. Их нельзя все объяснять конкретным событием. Ниже два локальных примера изменившихся тестовых тревог, не доказательство правильного обнаружения:

{table(cases)}

## Ограничения и решение

- Регион и тест уже рассматривались ранее; результат исследовательский, не независимая финальная оценка. Региональная выборка не обосновывает улучшение по всей России.
- МЧС — неполная поисковая выборка; Оренбург Медиа — датированный архив JSON, охватывающий также экономику и общество. Полнота страниц проверяется по X-WP-TotalPages в news_orenburg_archive_audit.csv; это не полнота всех новостей региона. Ноль найденных публикаций не означает отсутствие событий.
- API возвращает текущие версии заголовков; дата публикации берётся из date_gmt, история исправлений неизвестна. Региональные СМИ могут перепечатывать федеральные новости: тематические признаки требуют явной географии.
- Историческая доступность принимается по дате публикации. Фактический first_seen_at — 2026 год; в строгом онлайн-режиме эти сообщения не были получены системой в 2023–2024.
- Годовые версии географии не позволяют установить точный месяц административного изменения. Финансовые proxy-метки требуют экспертной проверки.
- Для h=12 валидация содержит один целевой месяц; доступные новости относятся к ранним origin, а не к будущему паводку. Будущие сообщения не подставлялись.

**Практическое решение:** основной ансамбль и выбранный основной детектор сохраняются. Новостной корпус расширен экономическим источником; следующий шаг — новые неиспользованные целевые месяцы, независимый источник экономики и экспертная проверка географии/тем.

## Воспроизведение и проверка

```bash
python scripts/collect_news_archive.py --offline # повторная обработка кеша без сети
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 15 python scripts/evaluate_news_impact.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 15 python scripts/build_news_impact_report.py
```

Нужны локальный CSV новостей, муниципальные входы и сохранённые z/прогнозы из исходного запуска. Хеши всех входов и кода сохранены в manifest.json, конфигурация — в config.json. forecast_predictions.csv содержит прогнозы теста, detector_alarms.csv — тревоги, detector_proxy_labels.csv — независимые от новостей ориентиры, detector_changed_alarms.csv — изменившиеся тревоги. Предыдущие запуски не перезаписываются.

Проверки текущего запуска: {sum(checks.values())}/{len(checks)} выполнено; метки обучения не позже origin, новости до cutoff, калибровочные частоты в бюджете, полное покрытие прогнозов и совпадение хешей. Первый технический запуск помечен superseded_invalid_geography_numeric_csv_cast: исправлена интерпретация числовых territory_id из CSV; его результаты не используются.
'''
    (ROOT/'docs/REGIONAL_NEWS_IMPACT_RU.md').write_text(text)
    (ROOT/'report/regional_news_impact.md').write_text(text)
    body=inline_md(text)
    for path in sorted(figures.glob('*.png')):
        body+='<figure><img style="max-width:100%" src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'" alt="Сравнение новостного пилота"></figure>'
    (ROOT/'report/regional_news_impact.html').write_text(html_page('Влияние региональных новостей',body))
    print('Отчёт готов:',run.relative_to(ROOT),'проверок:',len(checks))


if __name__=='__main__':
    main()
