"""Regional validation tuning, compared with frozen Orenburg transfer settings."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_news_decay import (REGIONS, feature_variants, apply_choices,
    add_families, hash_file, choose_representations)
from scripts.evaluate_llm_news import fit_corrections, select_validation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.mun_data import load_municipal, MunicipalPanel
from src.mun_eval import HORIZONS, make_grid, score_model, month_bootstrap_diff, per_series_mae
from src.llm_news import FIELDS


def choose_local_policies(chosen):
    """Same economic family as transfer, plus a separate filter-policy comparison."""
    economic = choose_representations(chosen)
    economic = economic[economic.family.eq('selected_news')].copy()
    economic['family'] = 'local_news'
    variants = ['all_windows','economic_windows','economic_shocks_windows'] + [
        name for name in chosen.variant.unique() if name.startswith('economic_decay_')]
    broad = chosen[chosen.variant.isin(variants)].copy()
    broad['complexity'] = broad.variant.map({name:i for i,name in enumerate(variants)})
    broad = broad.sort_values(['validation_mae','shrink','complexity','alpha']).groupby('h',sort=False).head(1)
    broad['family'] = 'local_policy'
    return pd.concat([economic,broad.drop(columns='complexity')],ignore_index=True)


def matched_coverage(variant):
    if variant.startswith('economic_decay_'):
        return variant.replace('economic_decay_', 'economic_coverage_decay_')
    return {'economic_windows':'economic_coverage_windows',
            'economic_shocks_windows':'economic_shocks_coverage', 'all_windows':'all_coverage'}[variant]


def assemble_coverage(selected, policies, family):
    result = next(iter(selected.values())).copy()
    for row in policies[policies.family.eq(family)].itertuples():
        result[:,row.h] = selected[matched_coverage(row.variant)][:,row.h]
    return result


def prepare_region(run, source_run, region, full, cfg):
    llm_cfg = json.loads((ROOT/REGIONS[region]).read_text())
    annotation = ROOT/llm_cfg['output_directory']/'pilot/annotated_news.csv'
    manifest = json.loads((annotation.parent/'manifest.json').read_text())
    previous = json.loads((source_run/region/'manifest.json').read_text())
    if hash_file(annotation) != previous['annotation_sha256'] or hash_file(annotation) != manifest['annotated_sha256']:
        raise ValueError('Use the same annotation as the transfer experiment')
    if hash_file(ROOT/llm_cfg['news_file']) != manifest['news_sha256']:
        raise ValueError('Corpus changed since annotation')
    records = pd.read_csv(annotation,dtype={'territory_ids':'string'})
    if len(records)!=manifest['selected_articles'] or records[FIELDS].isna().any().any() or records.region_code.ne(llm_cfg['region_code']).any():
        raise ValueError('Incomplete/mixed annotation')
    ids = np.flatnonzero(full.meta.region_code.to_numpy()==llm_cfg['region_code'])
    panel = MunicipalPanel(full.values[ids],full.meta.iloc[ids],full.months)
    old_index = pd.read_csv(source_run/region/'series_index.csv')
    np.testing.assert_array_equal(panel.meta[['territory_id','category']].to_numpy(),
                                  old_index[['territory_id','category']].to_numpy())
    extras, availability, schema = feature_variants(panel,records,cfg)
    for new,old in [('all_coverage','all_windows'),('economic_shocks_coverage','economic_shocks_windows')]:
        extras[new] = extras[old][:,:,[0,2,18,20]]
        schema[new] = [schema[old][i] for i in [0,2,18,20]]
    folder = run/region;folder.mkdir()
    panel.meta.to_csv(folder/'series_index.csv',index=False)
    availability.to_csv(folder/'availability_audit.csv',index=False)
    (folder/'feature_schema.json').write_text(json.dumps(schema,ensure_ascii=False,indent=2)+'\n')
    transfer_choices = pd.read_csv(source_run/'frozen_selection.csv')
    transfer_policies = pd.read_csv(source_run/'frozen_representations.csv')
    selected_bases, choices, policies, candidates_tables, training = {},[],[],[],[]
    with threadpool_limits(limits=1):
        for name in cfg['base_models']:
            paths = [p for p in previous['cached_prediction_sha256'] if Path(p).name==f'pred_{name}.npy']
            if len(paths)!=1 or hash_file(ROOT/paths[0]) != previous['cached_prediction_sha256'][paths[0]]:
                raise ValueError('Cached base forecast changed: '+name)
            base = np.load(ROOT/paths[0],mmap_mode='r')[:,:,ids].astype(float)
            variants,audits = fit_corrections(panel,base,extras,cfg)
            selected,choice,development = select_validation(panel,variants,make_grid())
            policy = choose_local_policies(choice)
            add_families(selected,policy)
            selected['local_news_coverage'] = assemble_coverage(selected,policy,'local_news')
            selected['local_policy_coverage'] = assemble_coverage(selected,policy,'local_policy')
            old_choice = transfer_choices[transfer_choices.base_model.eq(name)]
            transferred = apply_choices(variants,old_choice)
            add_families(transferred,transfer_policies[transfer_policies.base_model.eq(name)])
            selected['transferred_news'] = transferred['selected_news']
            selected_bases[name] = (base,selected)
            choices.append(choice.assign(base_model=name));policies.append(policy.assign(base_model=name))
            candidates_tables.append(development.assign(base_model=name))
            training.extend({'base_model':name,**a} for a in audits)
            print(region+': local validation fitted '+name,flush=True)
    pd.concat(choices).to_csv(folder/'frozen_selection.csv',index=False)
    policy_table = pd.concat(policies);policy_table.to_csv(folder/'frozen_policies.csv',index=False)
    pd.concat(candidates_tables).to_csv(folder/'validation_candidates.csv',index=False)
    pd.DataFrame(training).to_csv(folder/'training_audit.csv',index=False)
    freeze = {'selection_region':llm_cfg['region_code'],'selection_targets':'2024-01..2024-06',
        'test_targets':'2024-07..2024-12','frozen_at_utc':datetime.now(timezone.utc).isoformat(),
        'frozen_selection_sha256':hash_file(folder/'frozen_selection.csv'),
        'frozen_policies_sha256':hash_file(folder/'frozen_policies.csv'),
        'annotation_sha256':hash_file(annotation),'source_transfer_run':str(source_run.relative_to(ROOT)),
        'cached_prediction_sha256':previous['cached_prediction_sha256'],
        'region_name':previous['region_name'],'series':panel.n_series,'municipalities':panel.meta.territory_id.nunique(),
        'articles':len(records),'selection_before_current_test_scoring':True,
        'status':'exploratory; regional test previously inspected'}
    (folder/'selection_manifest.json').write_text(json.dumps(freeze,ensure_ascii=False,indent=2)+'\n')
    return panel,selected_bases,policy_table


def evaluate_region(run,source_run,region,prepared,cfg):
    panel,bases,policies = prepared
    folder = run/region;grid = make_grid()
    freeze = json.loads((folder/'selection_manifest.json').read_text())
    metrics,bootstrap,categories,wins = [],[],[],[]
    for name,(base,selected) in bases.items():
        for variant,pred in {'original':base,**selected}.items():
            metrics.append(score_model(panel,pred,grid,'test',variant).assign(base_model=name,variant=variant))
            if variant not in ['original','transferred_news','financial_only','local_news','local_policy',
                                'local_news_coverage','local_policy_coverage']:
                continue
            for category in panel.meta.category.unique():
                idx = np.flatnonzero(panel.meta.category.to_numpy()==category)
                sub = MunicipalPanel(panel.values[idx],panel.meta.iloc[idx],panel.months)
                categories.append(score_model(sub,pred[:,:,idx],grid,'test',variant).assign(base_model=name,variant=variant,category=category))
            if variant not in ['local_news','local_policy']:
                continue
            for h in HORIZONS:
                for comparator in ['original','transferred_news','financial_only',variant+'_coverage','coverage_all']:
                    control = base if comparator=='original' else selected[comparator]
                    d,lo,hi,n = month_bootstrap_diff(panel,pred,control,grid,'test',h,cfg['bootstrap_replicates'],cfg['seed'])
                    bootstrap.append({'base_model':name,'variant':variant,'h':h,'comparator':comparator,
                                      'mae_difference':d,'ci_low':lo,'ci_high':hi,'target_months':n})
                errors = pd.DataFrame({'territory_id':panel.meta.territory_id,
                    'original':per_series_mae(panel,base,grid,'test',h),
                    'news':per_series_mae(panel,pred,grid,'test',h)}).groupby('territory_id').mean()
                wins.append({'base_model':name,'variant':variant,'h':h,'municipalities':len(errors),
                             'improved_share':float((errors.news < errors.original).mean()),
                             'unchanged_share':float(np.isclose(errors.news,errors.original).mean())})
        np.save(folder/f'pred_{name}.npy',np.stack([base,selected['transferred_news'],selected['financial_only'],
                   selected['local_news'],selected['local_policy'],selected['local_news_coverage'],selected['local_policy_coverage']]))
    m = pd.concat(metrics);m.to_csv(folder/'metrics.csv',index=False)
    pd.DataFrame(bootstrap).to_csv(folder/'paired_bootstrap.csv',index=False)
    pd.concat(categories).to_csv(folder/'category_metrics.csv',index=False)
    pd.DataFrame(wins).to_csv(folder/'municipal_win_shares.csv',index=False)
    old = pd.read_csv(source_run/region/'metrics.csv').query('variant == "selected_news"')
    reproduced = m[m.variant.eq('transferred_news')].merge(old,on=['base_model','h'],suffixes=('_new','_old'))
    training = pd.read_csv(folder/'training_audit.csv')
    available = pd.read_csv(folder/'availability_audit.csv')
    dates = pd.to_datetime(available.latest_available_at,utc=True)
    cutoff = pd.to_datetime(available.cutoff_exclusive,utc=True)
    checks = {'full_prediction_coverage':bool(m.coverage.eq(1).all()),
        'past_only_news':bool((dates.isna() | dates.lt(cutoff)).all()),
        'mature_training_targets':bool(training.last_training_target.le(training.origin).all()),
        'local_selection_unchanged':hash_file(folder/'frozen_selection.csv')==freeze['frozen_selection_sha256'] and hash_file(folder/'frozen_policies.csv')==freeze['frozen_policies_sha256'],
        'transfer_metrics_reproduced':len(reproduced)==len(cfg['base_models'])*len(HORIZONS) and bool(np.allclose(reproduced.mae_new,reproduced.mae_old,rtol=1e-12))}
    (folder/'verification.json').write_text(json.dumps({'passed':all(checks.values()),'checks':checks},indent=2)+'\n')
    if not all(checks.values()):raise RuntimeError(checks)
    print('Completed local comparison: '+region,flush=True)


def build_report(run):
    chunks,summary_tables = [],[]
    for folder in [run/'nizhny',run/'kostroma']:
        manifest = json.loads((folder/'selection_manifest.json').read_text())
        metrics = pd.read_csv(folder/'metrics.csv');choices = pd.read_csv(folder/'frozen_policies.csv')
        bootstrap = pd.read_csv(folder/'paired_bootstrap.csv')
        show = ['original','transferred_news','financial_only','local_news_coverage','local_news','local_policy']
        primary = metrics[metrics.h.isin([1,3])].groupby(['base_model','variant']).mae.mean().unstack()[show]
        primary['local_news_change_pct'] = 100*(primary.local_news/primary.original-1)
        primary['local_policy_change_pct'] = 100*(primary.local_policy/primary.original-1)
        primary.reset_index().to_csv(folder/'primary_summary.csv',index=False)
        summary_tables.append(primary.reset_index().assign(region=manifest['region_name']))
        horizons = metrics[metrics.base_model.eq('Ensemble') & metrics.variant.isin(show)].pivot(index='variant',columns='h',values='mae')
        ensemble_choices = choices[choices.base_model.eq('Ensemble')]
        b = bootstrap[bootstrap.base_model.eq('Ensemble')]
        full_horizons = metrics[metrics.variant.isin(show)].pivot(index=['base_model','variant'],columns='h',values='mae')
        control = bootstrap[bootstrap.variant.eq('local_news') & bootstrap.h.isin([1,3]) & bootstrap.base_model.ne('Ensemble')]
        e = primary.loc['Ensemble']
        conclusion = f"Ensemble, средняя MAE по h=1/3: {e.original:.2f} → {e.local_news:.2f} ({e.local_news_change_pct:+.2f}%) при том же наборе экономических представлений, что в переносе; с выбором фильтра — {e.local_policy:.2f} ({e.local_policy_change_pct:+.2f}%)."
        if ensemble_choices[ensemble_choices.family.eq('local_news') & ensemble_choices.h.isin([1,3])].shrink.eq(0).all():
            conclusion += " Теперь нулевая поправка выбрана собственной региональной validation: сохранение прогноза уже не следствие только Оренбургских настроек."
        if e.financial_only < e.local_news:
            conclusion += (f" Финансовая поправка без новостей точнее экономического новостного варианта: "
                           f"{e.financial_only:.2f} против {e.local_news:.2f}.")
        for h in [1,3,6]:
            original = horizons.loc['original',h]
            policy_mae = horizons.loc['local_policy',h]
            if policy_mae < original-1e-9:
                improvement = 100*(policy_mae/original-1)
                conclusion += (f" При широком выборе фильтра на h={h} MAE снизилась "
                               f"{original:.2f} → {policy_mae:.2f} ({improvement:+.2f}%).")
                comparison = b[(b.variant=='local_policy') & (b.h==h) & (b.comparator=='local_policy_coverage')].iloc[0]
                if comparison.ci_low <= 0 <= comparison.ci_high:
                    conclusion += " Интервал сравнения с сопоставимым контролем объёма включает ноль; самостоятельный вклад сентимента и направлений не доказан."
        chunks.append(f"## {manifest['region_name']}\n\n{manifest['municipalities']} МО, {manifest['series']} рядов, {manifest['articles']} размеченных публикаций.\n\n{conclusion}\n\nОсновные горизонты h=1/3, средняя MAE:\n\n{primary.reset_index().to_markdown(index=False,floatfmt='.2f')}\n\nEnsemble по всем горизонтам:\n\n{horizons.reset_index().to_markdown(index=False,floatfmt='.2f')}\n\nВыбранные параметры Ensemble:\n\n{ensemble_choices.to_markdown(index=False,floatfmt='.3f')}\n\nBootstrap Ensemble, разность MAE относительно контроля (отрицательная лучше), 95% интервал шести целевых месяцев:\n\n{b.to_markdown(index=False,floatfmt='.3f')}\n\nКонтрольные сравнения остальных моделей, основной экономический вариант h=1/3:\n\n{control.to_markdown(index=False,floatfmt='.3f')}\n\nВсе модели по горизонтам:\n\n{full_horizons.reset_index().to_markdown(index=False,floatfmt='.2f')}\n")
    summary = pd.concat(summary_tables,ignore_index=True);summary.to_csv(run/'primary_summary.csv',index=False)
    ensemble_summary = summary[summary.base_model.eq('Ensemble')][[
        'region','original','transferred_news','financial_only','local_news_coverage','local_news',
        'local_policy','local_news_change_pct','local_policy_change_pct']]
    if not ensemble_summary.local_news_change_pct.lt(-1e-9).any():
        overall = ('**Местный подбор не улучшил среднюю MAE сильнейшего исходного Ensemble на h=1/3 '
                   'ни в одной из двух областей.** Расширение выбора фильтра также не даёт '
                   'среднего улучшения на этих горизонтах в данном запуске. '
                   'Основную модель менять по этому эксперименту оснований нет.')
    else:
        overall = ('Местный подбор дал отдельные точечные улучшения Ensemble на основных горизонтах. '
                   'Для оценки их устойчивости ниже приведены контрольные сравнения и интервалы; '
                   'само снижение MAE не доказывает вклад содержания новостей.')
    figures=run/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for ax,(region,table) in zip(axes,summary.groupby('region',sort=False)):
        x=np.arange(len(table))
        for i,(key,label) in enumerate([('original','Original'),('transferred_news','Orenburg settings'),('local_news','Local economic selection'),('local_policy','Local filter selection')]):
            ax.bar(x+(i-1.5)*.2,table[key],.2,label=label)
        ax.set_xticks(x,table.base_model,rotation=55,ha='right',fontsize=8);ax.set_title(region+'\nMean MAE, h=1/3');ax.set_ylabel('RUB per resident')
        ax.set_ylim(0,table[['original','transferred_news','local_news','local_policy']].max().max()*1.35);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(figures/'local_comparison.png',dpi=150);plt.close(fig)
    text=f'''# Местный подбор новостных признаков: Нижегородская и Костромская области

Запуск `{run.name}`. Восемь моделей, те же 1 200 размеченных публикаций на область и те же исходные прогнозы, что в [эксперименте переноса](NEWS_DECAY_TRANSFER_RU.md). Новая разметка и скачивание не выполнялись.

## Выводы

{overall}

Средняя MAE Ensemble по h=1/3, руб./жителя:

{ensemble_summary.to_markdown(index=False,floatfmt='.2f')}

Отдельное улучшение на одном горизонте не означает улучшения модели в целом. Ненулевые поправки, принятые validation, местами ухудшают второе полугодие — в особенности на h=6. Местный подбор требует проверки на новом периоде; он не гарантирует преимущества перед перенесёнными настройками. Для слабых исходных моделей есть отдельные выигрыши (полные таблицы ниже), но исходный Ensemble остаётся точнее новостных версий остальных моделей на основных горизонтах этого теста.

## Протокол

В каждой области alpha=100/1000, shrink=0/0.1/0.25/0.5 и представление выбираются только по validation январь–июнь 2024, отдельно для каждой модели и горизонта. После выбора в ОБЕИХ областях сохраняются frozen_selection.csv, frozen_policies.csv и SHA256; только затем считаются метрики test июль–декабрь. Обучение Ridge на каждом origin использует лишь созревшие ошибки и публикации не позже origin.

`local_news` выбирает экономические окна 1/3 месяца или затухание 14/30/90 дней — ровно тот же набор представлений, что прежний `selected_news`. Сравнение с `transferred_news` проверяет влияние местного подбора. `local_policy` дополнительно позволяет прежний all_windows и экономический отбор с ЧС: это отдельная более широкая проверка выбора фильтра. Все её варианты и гиперпараметры также выбираются по validation, не по test.

`financial_only` подбирается на местной validation без новостей. `local_news_coverage` использует объём/географию/категории и точно то же выбранное окно либо затухание, но исключает сентимент и направления. Его alpha/shrink подбираются по местной validation для соответствующего контроля. Аналогичный контроль для local_policy сохранён в метриках. Все нулевые поправки допустимы. На h=12 нет созревшего обучения в точке validation — исходный прогноз сохраняется.

![Местный подбор и перенос](../{run.relative_to(ROOT)}/figures/local_comparison.png)

'''+ '\n'.join(chunks)+f'''
## Ограничения и воспроизведение

Тест обеих областей уже просмотрен в предыдущем эксперименте. Это дополнительная исследовательская проверка на прежних данных, а не независимая будущая оценка; использование test для подбора в этом запуске исключено, но предыдущие результаты могли повлиять на сам выбор гипотезы. Шесть целевых месяцев, много сравнений, современная LLM без независимого ручного аудита и ретроспективные ограничения исходного Ensemble не позволяют объявить устойчивое улучшение. Региональный коэффициент Ridge не доказывает причинного влияния новости.

В активированном окружении sber: `python scripts/evaluate_local_news.py`. Настройки — configs/local_news.json; source_run указывает сохранённый перенос. При наличии Make: `make local-news`. Сырые ответы и разметка остаются в data/inputs/, бинарные прогнозы игнорируются Git. Метрики, график, выбор и манифесты сохраняются в `{run.relative_to(ROOT)}`.
'''
    (ROOT/'docs/LOCAL_NEWS_RESULTS_RU.md').write_text(text)
    (run/'results_report.md').write_text(text.replace('(NEWS_DECAY_TRANSFER_RU.md)','(../../../docs/NEWS_DECAY_TRANSFER_RU.md)').replace(f'../{run.relative_to(ROOT)}/figures/','figures/'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='configs/local_news.json');args=parser.parse_args()
    cfg_path=ROOT/args.config;cfg=json.loads(cfg_path.read_text());source_run=ROOT/cfg['source_run']
    source_files=[source_run/'frozen_selection.csv',source_run/'frozen_representations.csv',source_run/'selection_manifest.json']
    source_hashes={str(p.relative_to(ROOT)):hash_file(p) for p in source_files}
    run=ROOT/'artifacts/local_news_runs'/datetime.now(timezone.utc).strftime('local_%Y%m%dT%H%M%S_%fZ');run.mkdir(parents=True)
    (run/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
    full,_=load_municipal(ROOT)
    prepared={region:prepare_region(run,source_run,region,full,cfg) for region in cfg['regions']}
    (run/'all_selections_frozen.json').write_text(json.dumps({'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'regions':cfg['regions'],'before_any_current_test_scoring':True},indent=2)+'\n')
    for region,data in prepared.items():evaluate_region(run,source_run,region,data,cfg)
    assert source_hashes=={str(p.relative_to(ROOT)):hash_file(p) for p in source_files}
    (run/'manifest.json').write_text(json.dumps({'source_run':cfg['source_run'],'source_files_sha256':source_hashes,'config_sha256':hash_file(cfg_path),
        'code_sha256':{p:hash_file(ROOT/p) for p in ['scripts/evaluate_local_news.py','scripts/evaluate_news_decay.py','scripts/evaluate_llm_news.py','src/llm_news.py']},
        'all_region_choices_saved_before_test':True,'original_transfer_files_unchanged':True},indent=2)+'\n')
    build_report(run);(run.parent/'latest.json').write_text(json.dumps({'directory':str(run.relative_to(ROOT))})+'\n')
    print('Results:',run,flush=True)


if __name__=='__main__':main()
