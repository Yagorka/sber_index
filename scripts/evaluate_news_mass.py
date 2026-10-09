"""Absolute decaying news signals; choices frozen before scoring all three regions."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.report_language import explain_text
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
from scripts.evaluate_news_decay import REGIONS, hash_file, add_families
from scripts.evaluate_llm_news import fit_corrections, select_validation
from src.llm_news import aggregate_features, FIELDS, BASE_FEATURES
from src.mun_data import load_municipal, MunicipalPanel
from src.mun_eval import make_grid, score_model, month_bootstrap_diff, HORIZONS

MASS_NAMES = ['log_articles', 'log_geo_economic_articles', 'log_aligned_economic_articles',
              'log_negative_mass', 'log_positive_mass', 'log_mixed_mass', 'log_unknown_mass',
              'signed_log_sentiment_mass', 'log_price_up_mass', 'log_price_down_mass',
              'log_price_known_mass', 'log_income_up_mass', 'log_income_down_mass',
              'log_income_known_mass', 'log_business_up_mass', 'log_business_down_mass',
              'log_business_known_mass', 'log_shock_mass']


def features(panel, records, cfg):
    extras = {'financial_only':None};schema = {};audits = []
    for d in cfg['half_life_days']:
        for kind,mass in [('mass',True),('fractions',False)]:
            name = f'{kind}_{d}d'
            a,audit = aggregate_features(panel,records,economic_only=True,half_life_days=d,content_mass=mass)
            extras[name] = a;schema[name] = MASS_NAMES if mass else [f'{f}_{d}d' for f in BASE_FEATURES]
            audit['variant'] = name;audits.append(audit)
            if mass:
                extras[f'coverage_{d}d'] = a[:,:,[0,1,2]]
                schema[f'coverage_{d}d'] = MASS_NAMES[:3]
    return extras,pd.concat(audits),schema


def choose_families(choices):
    rows=[]
    for family,prefix in [('selected_mass','mass_'),('selected_fractions','fractions_')]:
        s=choices[choices.variant.str.startswith(prefix)].copy()
        s=s.sort_values(['validation_mae','shrink','variant','alpha']).groupby('h',sort=False).head(1)
        s['family']=family;rows.append(s)
    return pd.concat(rows,ignore_index=True)


def prepare(run,region,full,cfg):
    source=ROOT/cfg['source_run'];prior=json.loads((source/region/'manifest.json').read_text())
    lc=json.loads((ROOT/REGIONS[region]).read_text());labels=ROOT/lc['output_directory']/'pilot/annotated_news.csv'
    lm=json.loads((labels.parent/'manifest.json').read_text())
    assert hash_file(labels)==lm['annotated_sha256']==prior['annotation_sha256']
    assert hash_file(ROOT/lc['news_file'])==lm['news_sha256']
    records=pd.read_csv(labels,dtype={'territory_ids':'string'})
    assert len(records)==lm['selected_articles'] and not records[FIELDS].isna().any().any()
    assert records.region_code.eq(lc['region_code']).all()
    ids=np.flatnonzero(full.meta.region_code.to_numpy()==lc['region_code'])
    panel=MunicipalPanel(full.values[ids],full.meta.iloc[ids],full.months)
    np.testing.assert_array_equal(panel.meta[['territory_id','category']].to_numpy(),pd.read_csv(source/region/'series_index.csv')[['territory_id','category']].to_numpy())
    folder=run/region;folder.mkdir();panel.meta.to_csv(folder/'series_index.csv',index=False)
    extras,audit,schema=features(panel,records,cfg);audit.to_csv(folder/'availability_audit.csv',index=False)
    (folder/'feature_schema.json').write_text(json.dumps(schema,indent=2)+'\n')
    bases={};choices=[];policies=[];validation=[];training=[]
    with threadpool_limits(limits=1):
        for name in cfg['base_models']:
            path=next(p for p in prior['cached_prediction_sha256'] if Path(p).name==f'pred_{name}.npy')
            assert hash_file(ROOT/path)==prior['cached_prediction_sha256'][path]
            base=np.load(ROOT/path,mmap_mode='r')[:,:,ids].astype(float)
            candidates,ta=fit_corrections(panel,base,extras,cfg)
            selected,choice,dev=select_validation(panel,candidates,make_grid())
            policy=choose_families(choice);add_families(selected,policy)
            coverage=base.copy()
            for row in policy[policy.family.eq('selected_mass')].itertuples():
                coverage[:,row.h]=selected[row.variant.replace('mass_','coverage_')][:,row.h]
            selected['matched_coverage']=coverage
            bases[name]=(base,selected)
            choices.append(choice.assign(base_model=name));policies.append(policy.assign(base_model=name))
            validation.append(dev.assign(base_model=name));training.extend({'base_model':name,**a} for a in ta)
            print(region+': fitted '+name,flush=True)
    pd.concat(choices).to_csv(folder/'frozen_selection.csv',index=False)
    pd.concat(policies).to_csv(folder/'frozen_families.csv',index=False)
    pd.concat(validation).to_csv(folder/'validation_candidates.csv',index=False)
    pd.DataFrame(training).to_csv(folder/'training_audit.csv',index=False)
    manifest={'region_name':prior['region_name'],'annotation_sha256':hash_file(labels),
        'cached_prediction_sha256':prior['cached_prediction_sha256'],'series':panel.n_series,
        'municipalities':int(panel.meta.territory_id.nunique()),'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
        'selection_sha256':hash_file(folder/'frozen_selection.csv'),'families_sha256':hash_file(folder/'frozen_families.csv'),
        'selection_targets':'2024-01..2024-06','test_targets':'2024-07..2024-12',
        'status':'exploratory; test previously inspected'}
    (folder/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return panel,bases


def score(run,region,prepared,cfg):
    folder=run/region;panel,bases=prepared;metrics=[];boot=[];cats=[]
    variants=['original','financial_only','selected_fractions','selected_mass','matched_coverage']
    for name,(base,selected) in bases.items():
        for variant in variants:
            pred=base if variant=='original' else selected[variant]
            metrics.append(score_model(panel,pred,make_grid(),'test',variant).assign(base_model=name,variant=variant))
            if variant in ['original','selected_mass']:
                for cat in panel.meta.category.unique():
                    idx=np.flatnonzero(panel.meta.category.to_numpy()==cat)
                    sub=MunicipalPanel(panel.values[idx],panel.meta.iloc[idx],panel.months)
                    cats.append(score_model(sub,pred[:,:,idx],make_grid(),'test',variant).assign(base_model=name,variant=variant,category=cat))
        for h in HORIZONS:
            for comparator in ['original','financial_only','matched_coverage','selected_fractions']:
                control=base if comparator=='original' else selected[comparator]
                d,lo,hi,n=month_bootstrap_diff(panel,selected['selected_mass'],control,make_grid(),'test',h,cfg['bootstrap_replicates'],cfg['seed'])
                boot.append(dict(base_model=name,h=h,comparator=comparator,mae_difference=d,ci_low=lo,ci_high=hi,target_months=n))
        np.save(folder/f'pred_{name}.npy',np.stack([base]+[selected[v] for v in variants[1:]]))
    m=pd.concat(metrics);m.to_csv(folder/'metrics.csv',index=False)
    pd.DataFrame(boot).to_csv(folder/'paired_bootstrap.csv',index=False)
    pd.concat(cats).to_csv(folder/'category_metrics.csv',index=False)
    freeze=json.loads((folder/'manifest.json').read_text());audit=pd.read_csv(folder/'availability_audit.csv');ta=pd.read_csv(folder/'training_audit.csv')
    dates=pd.to_datetime(audit.latest_available_at,utc=True);ends=pd.to_datetime(audit.cutoff_exclusive,utc=True)
    checks={'coverage':bool(m.coverage.eq(1).all()),'past_only_news':bool((dates.isna()|dates.lt(ends)).all()),
        'mature_targets':bool(ta.last_training_target.le(ta.origin).all()),
        'selection_unchanged':hash_file(folder/'frozen_selection.csv')==freeze['selection_sha256'] and hash_file(folder/'frozen_families.csv')==freeze['families_sha256']}
    (folder/'verification.json').write_text(json.dumps({'passed':all(checks.values()),'checks':checks},indent=2)+'\n')
    assert all(checks.values()),checks


def report(run,cfg):
    tables=[];details=[]
    for region in cfg['regions']:
        folder=run/region;mf=json.loads((folder/'manifest.json').read_text());m=pd.read_csv(folder/'metrics.csv')
        t=m[m.h.isin([1,3])].groupby(['base_model','variant']).mae.mean().unstack().reset_index()
        oldrun=ROOT/(cfg['source_run'] if region=='orenburg' else cfg['previous_local_run'])
        oldvariant='selected_news' if region=='orenburg' else 'local_news'
        old=pd.read_csv(oldrun/region/'metrics.csv');old=old[old.variant.eq(oldvariant)&old.h.isin([1,3])].groupby('base_model').mae.mean()
        t['previous_news']=t.base_model.map(old)
        t['change_pct']=100*(t.selected_mass/t.original-1);t['region']=mf['region_name'];tables.append(t)
        b=pd.read_csv(folder/'paired_bootstrap.csv');choice=pd.read_csv(folder/'frozen_families.csv')
        horizons=m[m.base_model.eq('Ensemble')].pivot(index='variant',columns='h',values='mae')
        details.append(f"## {mf['region_name']}\n\n{t.drop(columns='region').to_markdown(index=False,floatfmt='.2f')}\n\nEnsemble по горизонтам:\n\n{horizons.to_markdown(floatfmt='.2f')}\n\nВыбор Ensemble:\n\n{choice[choice.base_model.eq('Ensemble')].to_markdown(index=False,floatfmt='.3f')}\n\nBootstrap Ensemble (отрицательная разность MAE лучше):\n\n{b[b.base_model.eq('Ensemble')].to_markdown(index=False,floatfmt='.3f')}\n")
    summary=pd.concat(tables);summary.to_csv(run/'primary_summary.csv',index=False)
    e=summary[summary.base_model.eq('Ensemble')]
    conclusion=('Средняя MAE лучшего исходного Ensemble на h=1/3 не улучшилась ни в одной области.' if not e.change_pct.lt(-1e-9).any() else 'Есть точечные улучшения Ensemble на h=1/3; их величину и сравнение с контролями показывает таблица ниже.')
    figures=run/'figures';figures.mkdir(exist_ok=True);fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,(_,row) in zip(axes,e.iterrows()):
        keys=['original','previous_news','selected_fractions','selected_mass','financial_only','matched_coverage']
        ax.bar(range(len(keys)),[row[k] for k in keys]);ax.set_xticks(range(len(keys)),['Исходная','Прежние\nновости','Доли\nновостей','С учётом\nдавности','Без\nновостей','Количество\nновостей'],rotation=55,ha='right',fontsize=8)
        ax.set_title(row.region,fontsize=10);ax.set_ylabel('Средняя MAE, 1 и 3 месяца, руб.')
    fig.tight_layout();fig.savefig(figures/'ensemble.png',dpi=150);plt.close(fig)
    text=f'''# Затухание самого новостного сигнала

{conclusion} Основная модель по этому исследовательскому запуску автоматически не заменяется.

{e.to_markdown(index=False,floatfmt='.2f')}

## Протокол

Используются экономические новости, относящиеся к нужному месту и категории расходов. Оценки новостей складываются, а старые публикации имеют меньший вес. Проверены сроки уменьшения веса вдвое 14, 30 и 90 дней. Например, при сроке 30 дней вес новости равен половине начального через месяц и четверти через два месяца. Если новых публикаций нет, её влияние продолжает уменьшаться.

Отдельно считаются позитивные, негативные, смешанные и неизвестные оценки, направления цен, доходов и работы бизнеса. Отсутствие сведений не считается ни ростом, ни снижением. Большие суммы оценок дополнительно уменьшаются логарифмом; в коде это `log1p`. Вес выборки учитывает, что размечена часть архива. Это оценка новостного потока, а не величина изменения экономики.

Для каждого региона срок, коэффициент регуляризации Ridge (100 или 1000) и доля применяемой поправки (0, 0,1, 0,25 или 0,5) выбираются на январе–июне 2024. Все настройки восьми моделей в трёх областях сохранены до расчёта текущих результатов на июле–декабре. Обучение использует только прошлые прогнозы, для которых уже получены фактические расходы. Доля поправки 0 сохраняет исходный прогноз. Для горизонта 12 месяцев на периоде подбора ещё нет нужных прошлых примеров, поэтому поправка не применяется.

Варианты сравнения:

- **Новости с уменьшением влияния со временем** — суммы оценок, описанные выше.
- **Доли новостей с учётом давности** — прежний способ: доли позитивных/негативных новостей с меньшим весом старых публикаций. Единственная старая негативная новость может оставаться «100% негативных».
- **Прежний вариант с новостями** — ранее выбранные окна 1/3 месяца или учёт давности. Для двух новых областей использованы местные настройки, для Оренбургской — прежний исходный эксперимент.
- **Поправка без новостей** — только прошлые расходы и известные ошибки модели.
- **Только количество новостей** — три счётчика публикаций: всего, с подходящей географией и с подходящей категорией. Срок уменьшения веса тот же, что у выбранного новостного варианта; параметры Ridge подобраны отдельно.

Объяснения MAE, горизонта, bootstrap и других обозначений — в [словаре терминов](TERMS_RU.md).

![Ensemble](../{run.relative_to(ROOT)}/figures/ensemble.png)

'''+ '\n'.join(details)+f'''
## Ограничения и воспроизведение

Повторная исследовательская оценка на уже просмотренном test июль–декабрь 2024, шесть целевых месяцев и много сравнений. Bootstrap по целевым месяцам не исправляет множественность сравнений и не доказывает причинность. Современная LLM, заголовки, пилотные 1200 публикаций на регион и ретроспективные ограничения исходной модели остаются. Независимой ручной точности разметки пока нет; см. [аудит разметки](NEWS_LABEL_AUDIT_RU.md). Новый период данных здесь не добавлялся.

Запуск в существующей среде: `python scripts/evaluate_news_mass.py`. Конфигурация configs/news_mass.json; результаты `{run.relative_to(ROOT)}`. Сырые разметки и бинарные прогнозы не включаются в Git.
'''
    text=explain_text(text)
    (ROOT/'docs/NEWS_MASS_RESULTS_RU.md').write_text(text)
    (run/'results_report.md').write_text(text.replace(f'../{run.relative_to(ROOT)}/figures/','figures/').replace('(NEWS_LABEL_AUDIT_RU.md)','(../../../docs/NEWS_LABEL_AUDIT_RU.md)').replace('(TERMS_RU.md)','(../../../docs/TERMS_RU.md)'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/news_mass.json');args=p.parse_args()
    cfg=json.loads((ROOT/args.config).read_text());run=ROOT/'artifacts/news_mass_runs'/datetime.now(timezone.utc).strftime('mass_%Y%m%dT%H%M%S_%fZ');run.mkdir(parents=True)
    (run/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
    full,_=load_municipal(ROOT);prepared={r:prepare(run,r,full,cfg) for r in cfg['regions']}
    (run/'all_selections_frozen.json').write_text(json.dumps({'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'regions':cfg['regions'],'before_any_current_test_scoring':True},indent=2)+'\n')
    for region,data in prepared.items():score(run,region,data,cfg)
    report(run,cfg)
    (run/'code_manifest.json').write_text(json.dumps({str(p.relative_to(ROOT)):hash_file(p) for p in [ROOT/args.config,Path(__file__),ROOT/'src/llm_news.py',ROOT/'scripts/evaluate_llm_news.py']},indent=2)+'\n')
    (run.parent/'latest.json').write_text(json.dumps({'directory':str(run.relative_to(ROOT))})+'\n');print('Results:',run,flush=True)

if __name__=='__main__':main()
