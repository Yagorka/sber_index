"""Economic filtering and decay, with settings frozen on Orenburg before transfer."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_llm_news import fit_corrections, select_validation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.llm_news import aggregate_features, economic_filter, BASE_FEATURES, FIELDS
from src.mun_data import load_municipal, MunicipalPanel
from src.mun_eval import HORIZONS, make_grid, score_model, month_bootstrap_diff, per_series_mae

REGIONS = {'orenburg': 'configs/llm_news.json', 'nizhny': 'configs/llm_news_nizhny.json',
           'kostroma': 'configs/llm_news_kostroma.json'}


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collection_summary(run, region, llm_cfg):
    news = Path(llm_cfg['news_file'])
    path = ROOT/news.with_name(news.stem+'_manifest.json')
    if not path.exists():
        raise FileNotFoundError('Missing collection manifest: '+str(path))
    manifest = json.loads(path.read_text())
    summary = {key:manifest[key] for key in ['articles','csv_sha256','config_sha256',
               'collected_at','all_regional_news_complete','code_sha256'] if key in manifest}
    summary.update(source=manifest.get('source',manifest.get('sources')),
                   complete_archive_periods=manifest.get('complete_weeks',manifest.get('complete_archive_months')),
                   period_kind='weeks' if 'complete_weeks' in manifest else 'months',
                   excluded_undated_objects=len(manifest.get('excluded_undated_archive_objects',[])),
                   period_start='2023-01-01',period_end='2024-12-31')
    (run/f'collection_{region}.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')


def feature_variants(panel, records, cfg):
    arrays, audits, schemas = {}, [], {}
    specifications = [('all_windows', {}), ('economic_windows', {'economic_only':True}),
                      ('economic_shocks_windows', {'economic_only':True,'retain_shocks':True})]
    specifications += [(f'economic_decay_{d}d', {'economic_only':True,'half_life_days':d})
                       for d in cfg['half_life_days']]
    for name, kwargs in specifications:
        array, audit = aggregate_features(panel, records, **kwargs)
        arrays[name] = array
        audit['variant'] = name
        audits.append(audit)
        labels = ([f'{f}_{kwargs["half_life_days"]}d' for f in BASE_FEATURES] if 'half_life_days' in kwargs else
                  [f'{f}_{w}m' for w in [1,3] for f in BASE_FEATURES])
        schemas[name] = labels
    arrays['financial_only'] = None
    arrays['coverage_all'] = arrays['all_windows'][:, :, [0,18]]
    schemas['coverage_all'] = [schemas['all_windows'][i] for i in [0,18]]
    arrays['economic_coverage_windows'] = arrays['economic_windows'][:, :, [0,2,18,20]]
    schemas['economic_coverage_windows'] = [schemas['economic_windows'][i] for i in [0,2,18,20]]
    for d in cfg['half_life_days']:
        name = f'economic_coverage_decay_{d}d'
        arrays[name] = arrays[f'economic_decay_{d}d'][:, :, [0,2]]
        schemas[name] = [schemas[f'economic_decay_{d}d'][i] for i in [0,2]]
    return arrays, pd.concat(audits), schemas


def choose_representations(chosen):
    """Freeze only validation choices, preferring simpler features and zero ties."""
    result = {}
    for family, variants in [('selected_decay', [f'economic_decay_{d}d' for d in [14,30,90]]),
                             ('selected_news', ['economic_windows']+[f'economic_decay_{d}d' for d in [14,30,90]])]:
        subset = chosen[chosen.variant.isin(variants)].copy()
        subset['complexity'] = subset.variant.ne('economic_windows').astype(int)
        selected = subset.sort_values(['validation_mae','shrink','complexity','variant','alpha']).groupby('h',sort=False).head(1)
        selected['family'] = family
        result[family] = selected.drop(columns='complexity')
    return pd.concat(result.values(), ignore_index=True)


def apply_choices(candidates, choices):
    selected = {}
    for name, subset in choices.groupby('variant'):
        result = candidates[next(k for k in candidates if k[0] == name)].copy()
        for row in subset.itertuples():
            result[:,row.h] = candidates[row.variant,row.alpha,row.shrink][:,row.h]
        selected[name] = result
    return selected


def add_families(selected, representations):
    for family, subset in representations.groupby('family'):
        result = next(iter(selected.values())).copy()
        for row in subset.itertuples():
            result[:,row.h] = selected[row.variant][:,row.h]
        selected[family] = result


def evaluate_region(root_run, region, full, cfg):
    folder = root_run/region
    if folder.exists():
        raise ValueError(f'Region already evaluated: {region}')
    llm_cfg = json.loads((ROOT/REGIONS[region]).read_text())
    labels_path = ROOT/llm_cfg['output_directory']/'pilot/annotated_news.csv'
    records = pd.read_csv(labels_path, dtype={'territory_ids':'string'})
    annotation_manifest = json.loads((labels_path.parent/'manifest.json').read_text())
    if len(records) != annotation_manifest['selected_articles'] or records[FIELDS].isna().any().any():
        raise ValueError('Incomplete annotation; do not score a partial sample')
    if records.region_code.ne(llm_cfg['region_code']).any():
        raise ValueError('Mixed regional corpus')
    if hash_file(labels_path) != annotation_manifest['annotated_sha256']:
        raise ValueError('Annotation changed after its manifest was written')
    if hash_file(ROOT/llm_cfg['news_file']) != annotation_manifest['news_sha256']:
        raise ValueError('News corpus changed after annotation')
    collection_summary(root_run, region, llm_cfg)
    ids = np.flatnonzero(full.meta.region_code.to_numpy() == llm_cfg['region_code'])
    if not len(ids):
        raise ValueError('Region absent from panel')
    panel = MunicipalPanel(full.values[ids], full.meta.iloc[ids], full.months)
    extras, availability, schemas = feature_variants(panel, records, cfg)
    folder.mkdir()
    availability.to_csv(folder/'availability_audit.csv',index=False)
    panel.meta.to_csv(folder/'series_index.csv',index=False)
    (folder/'feature_schema.json').write_text(json.dumps(schemas,ensure_ascii=False,indent=2)+'\n')
    source = ROOT/json.loads((ROOT/'artifacts/municipal_runs/latest.json').read_text())['directory']
    research = ROOT/json.loads((ROOT/'artifacts/research_runs/latest.json').read_text())['directory']
    frozen_path = root_run/'frozen_selection.csv'
    representations_path = root_run/'frozen_representations.csv'
    if region != 'orenburg':
        frozen_manifest = json.loads((root_run/'selection_manifest.json').read_text())
        if (hash_file(frozen_path) != frozen_manifest['frozen_selection_sha256'] or
                hash_file(representations_path) != frozen_manifest['frozen_representations_sha256']):
            raise ValueError('Frozen Orenburg settings changed before transfer')
        frozen = pd.read_csv(frozen_path)
        representations_all = pd.read_csv(representations_path)
        frozen_hash = hash_file(frozen_path)
        representation_hash = hash_file(representations_path)
    grid = make_grid()
    selections, representation_rows, developments, training, metrics, boot, category, wins, predictions = [],[],[],[],[],[],[],[],[]
    input_hashes = {}
    # For source selection freeze every base before computing ANY test metric.
    selected_bases = {}
    with threadpool_limits(limits=1):
        for base_name in cfg['base_models']:
            path = (research if base_name == 'PastOnlyBlend' else source)/f'pred_{base_name}.npy'
            base = np.load(path,mmap_mode='r')[:,:,ids].astype(float)
            input_hashes[str(path.relative_to(ROOT))] = hash_file(path)
            candidates, audit = fit_corrections(panel,base,extras,cfg)
            training.extend({'base_model':base_name,**r} for r in audit)
            if region == 'orenburg':
                selected, chosen, development = select_validation(panel,candidates,grid)
                representations = choose_representations(chosen)
                chosen['base_model'] = base_name
                representations['base_model'] = base_name
                development['base_model'] = base_name
                developments.append(development)
            else:
                chosen = frozen[frozen.base_model == base_name].copy()
                representations = representations_all[representations_all.base_model == base_name].copy()
                selected = apply_choices(candidates,chosen)
            selections.append(chosen)
            representation_rows.append(representations)
            add_families(selected,representations)
            selected_bases[base_name] = (base,selected)
            print(f'{region}: fitted {base_name}',flush=True)
        selection = pd.concat(selections,ignore_index=True)
        representation = pd.concat(representation_rows,ignore_index=True)
        selection.to_csv(folder/'applied_selection.csv',index=False)
        representation.to_csv(folder/'applied_representations.csv',index=False)
        if region == 'orenburg':
            selection.to_csv(frozen_path,index=False)
            representation.to_csv(representations_path,index=False)
            pd.concat(developments).to_csv(folder/'validation_candidates.csv',index=False)
            # This is written before entering the test loop and before loading other regions.
            (root_run/'selection_manifest.json').write_text(json.dumps({
                'selection_region':56,'selection_targets':'2024-01..2024-06',
                'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
                'frozen_selection_sha256':hash_file(frozen_path),
                'frozen_representations_sha256':hash_file(representations_path),
                'transfer_regions':[52,44], 'transfer_refits_mature_residual_coefficients':True,
                'transfer_retunes_hyperparameters':False},indent=2)+'\n')
        for base_name,(base,selected) in selected_bases.items():
            for name,pred in {'original':base,**selected}.items():
                metrics.append(score_model(panel,pred,grid,'test',name).assign(base_model=base_name,variant=name))
                if name in ['original','financial_only','coverage_all','economic_windows','selected_decay','selected_news']:
                    for cat in panel.meta.category.unique():
                        idx = np.flatnonzero(panel.meta.category.to_numpy() == cat)
                        sub = MunicipalPanel(panel.values[idx],panel.meta.iloc[idx],panel.months)
                        category.append(score_model(sub,pred[:,:,idx],grid,'test',name).assign(base_model=base_name,variant=name,category=cat))
                if name == 'original':
                    continue
                for h in HORIZONS:
                    d,lo,hi,n = month_bootstrap_diff(panel,pred,base,grid,'test',h,cfg['bootstrap_replicates'],cfg['seed'])
                    boot.append({'base_model':base_name,'variant':name,'h':h,'comparator':'original','mae_difference':d,'ci_low':lo,'ci_high':hi,'months':n})
                if name in ['economic_windows','selected_decay','selected_news']:
                    for h in HORIZONS:
                        if name == 'economic_windows':
                            coverage_name = 'economic_coverage_windows'
                        else:
                            choice = representation[(representation.base_model == base_name) & (representation.family == name) & (representation.h == h)].iloc[0].variant
                            coverage_name = choice.replace('economic_decay_', 'economic_coverage_decay_') if 'decay' in choice else 'economic_coverage_windows'
                        for comparator in ['financial_only','coverage_all',coverage_name]:
                            d,lo,hi,n = month_bootstrap_diff(panel,pred,selected[comparator],grid,'test',h,cfg['bootstrap_replicates'],cfg['seed'])
                            boot.append({'base_model':base_name,'variant':name,'h':h,'comparator':comparator,'mae_difference':d,'ci_low':lo,'ci_high':hi,'months':n})
                        local = pd.DataFrame({'territory_id':panel.meta.territory_id,'original':per_series_mae(panel,base,grid,'test',h),'news':per_series_mae(panel,pred,grid,'test',h)}).groupby('territory_id').mean()
                        wins.append({'base_model':base_name,'variant':name,'h':h,'municipalities':len(local),'improved_share':float((local.news<local.original).mean()),'unchanged_share':float(np.isclose(local.news,local.original).mean())})
                if name in ['original','selected_news']:
                    for row in grid[grid.stage.eq('test')].itertuples():
                        predictions.extend({'base_model':base_name,'variant':name,'origin':row.origin,'target':row.target,'h':row.h,'series_idx':i,'actual':panel.values[i,row.target],'prediction':pred[row.origin,row.h,i]} for i in range(panel.n_series))
    pd.concat(metrics).to_csv(folder/'metrics.csv',index=False)
    pd.concat(category).to_csv(folder/'category_metrics.csv',index=False)
    pd.DataFrame(boot).to_csv(folder/'paired_bootstrap.csv',index=False)
    pd.DataFrame(training).to_csv(folder/'training_audit.csv',index=False)
    pd.DataFrame(wins).to_csv(folder/'municipal_win_shares.csv',index=False)
    pd.DataFrame(predictions).to_csv(folder/'predictions.csv',index=False)
    dates = pd.to_datetime(availability.latest_available_at,utc=True)
    cutoff = pd.to_datetime(availability.cutoff_exclusive,utc=True)
    train = pd.DataFrame(training)
    checks = {'complete_annotation':len(records)==annotation_manifest['selected_articles'],
              'past_only_news':bool((dates.isna() | (dates<cutoff)).all()),
              'mature_training_targets':bool((train.last_training_target <= train.origin).all()),
              'finite_features':all(np.isfinite(a).all() for a in extras.values() if a is not None),
              'full_prediction_coverage':bool(pd.concat(metrics).coverage.eq(1).all()),
              'transfer_selection_unchanged':region=='orenburg' or (hash_file(frozen_path)==frozen_hash and hash_file(representations_path)==representation_hash),
              'selection_matches_source':region=='orenburg' or selection.reset_index(drop=True).equals(frozen.reset_index(drop=True))}
    (folder/'verification.json').write_text(json.dumps({'passed':all(checks.values()),'checks':checks},indent=2)+'\n')
    if not all(checks.values()):
        raise RuntimeError(checks)
    filtered = economic_filter(records)
    (folder/'manifest.json').write_text(json.dumps({'region_code':llm_cfg['region_code'],'region_name':str(panel.meta.region_name.iloc[0]),
        'series':panel.n_series,'municipalities':panel.meta.territory_id.nunique(),'articles':len(records),
        'unique_titles':records.title_sha256.nunique(),'economic_articles':int(filtered.sum()),
        'economic_and_shock_articles':int(economic_filter(records,True).sum()),
        'annotation_sha256':hash_file(labels_path),'annotation_manifest_sha256':hash_file(labels_path.parent/'manifest.json'),
        'code_sha256':{p:hash_file(ROOT/p) for p in ['scripts/evaluate_news_decay.py','scripts/evaluate_llm_news.py','src/llm_news.py']},
        'selected_before_region_scoring':True,'tuning_region':56,'cached_prediction_sha256':input_hashes,
        'source_hyperparameters_applied_unchanged':region!='orenburg'},ensure_ascii=False,indent=2)+'\n')
    print('Completed:',region,flush=True)


def report(run):
    chunks, all_tables, primary_tables, content_evidence = [], [], [], []
    for region in REGIONS:
        folder = run/region
        if not (folder/'metrics.csv').exists():
            continue
        manifest = json.loads((folder/'manifest.json').read_text())
        m = pd.read_csv(folder/'metrics.csv')
        means = m.groupby(['base_model','variant']).mae.mean().unstack()
        show = means[['original','financial_only','coverage_all','all_windows','economic_windows',
                      'economic_shocks_windows','selected_decay','selected_news']].copy()
        show['change_pct'] = 100*(show.selected_news/show.original-1)
        all_tables.append(show.reset_index().assign(region=manifest['region_name']))
        primary = m[m.h.isin([1,3])].groupby(['base_model','variant']).mae.mean().unstack()
        primary = primary[['original','financial_only','coverage_all','all_windows','economic_windows',
                           'economic_shocks_windows','selected_decay','selected_news']].copy()
        primary['change_pct'] = 100*(primary.selected_news/primary.original-1)
        primary.reset_index().to_csv(folder/'primary_summary.csv',index=False)
        primary_tables.append(primary.reset_index().assign(region=manifest['region_name']))
        horizons = m[m.variant.isin(['original','economic_windows','selected_decay','selected_news'])].pivot(index=['base_model','variant'],columns='h',values='mae')
        b = pd.read_csv(folder/'paired_bootstrap.csv')
        b = b[b.variant.eq('selected_news') & b.h.isin([1,3])]
        original_diff = b[b.comparator.eq('original')].set_index(['base_model','h']).mae_difference
        supported = b[b.comparator.str.startswith('economic_coverage') & b.ci_high.lt(0)].copy()
        if not supported.empty:
            supported = supported[[original_diff.loc[(row.base_model,row.h)] < -1e-9
                                   for row in supported.itertuples()]]
            if not supported.empty:
                content_evidence.append(supported.assign(region=manifest['region_name']))
        ensemble = primary.loc['Ensemble']
        conclusion = (f"Ensemble на основных h=1/3: {ensemble.original:.2f} → "
                      f"{ensemble.selected_news:.2f} ({ensemble.change_pct:+.2f}%).")
        chunks.append(f"## {manifest['region_name']}\n\n{manifest['municipalities']} МО, {manifest['series']} рядов; {manifest['articles']} публикаций ({manifest['unique_titles']} уникальных заголовков), экономический отбор оставил {manifest['economic_articles']}, с ЧС — {manifest['economic_and_shock_articles']}.\n\n{conclusion}\n\n**Основные горизонты h=1/3**, средняя MAE, руб./жителя. selected_news выбирает окна или затухание только по Оренбургской валидации, selected_decay — лучший из 14/30/90 дней по той же валидации.\n\n{primary.reset_index().to_markdown(index=False,floatfmt='.2f')}\n\nСредняя MAE по всем h=1/3/6/12:\n\n{show.reset_index().to_markdown(index=False,floatfmt='.2f')}\n\nПо горизонтам:\n\n{horizons.reset_index().to_markdown(index=False,floatfmt='.2f')}\n\nКонтрольные сравнения выбранного варианта на основных h=1/3: разность MAE (отрицательная лучше), 95% bootstrap целевых месяцев.\n\n{b.to_markdown(index=False,floatfmt='.2f')}\n")
    tables = pd.concat(all_tables,ignore_index=True)
    tables.to_csv(run/'summary.csv',index=False)
    primary_all = pd.concat(primary_tables,ignore_index=True)
    primary_all.to_csv(run/'primary_summary.csv',index=False)
    conclusions = ('На основных h=1/3 Ensemble остаётся исходным: на Оренбургской validation '
                   'выбран shrink=0, и это решение перенесено на другие регионы. '
                   'Преимущество над лучшим исходным Ensemble не получено. '
                   'У SeasonalNaive_NatGrowth экономические признаки с выбранным затуханием '
                   'снижают среднюю ошибку относительно исходной модели во всех оценённых регионах; '
                   'финансовый контроль и контроль объёма нужны для отделения вклада новостей.')
    seasonal = primary_all[primary_all.base_model.eq('SeasonalNaive_NatGrowth')][
        ['region','original','financial_only','coverage_all','selected_news','change_pct']]
    evidence_text = ''
    if content_evidence:
        evidence = pd.concat(content_evidence,ignore_index=True)
        evidence.to_csv(run/'content_evidence_primary.csv',index=False)
        evidence_text = ('Есть отдельные случаи уменьшения ошибки и относительно оригинала, '
                         'и относительно сопоставимого контроля экономического объёма; '
                         'в таблице показаны случаи с верхней границей bootstrap-интервала ниже нуля '
                         'для сравнения с объёмом. Это исследовательские сигналы, а не подтверждение '
                         'устойчивого эффекта: шесть месяцев, множественные сравнения; '
                         'интервал сравнения с оригиналом может включать ноль.\n\n'+
                         evidence.to_markdown(index=False,floatfmt='.3f'))
    fig, axes = plt.subplots(1,len(chunks),figsize=(6*len(chunks),4),squeeze=False)
    for ax,(region,table) in zip(axes[0],tables.groupby('region',sort=False)):
        x=np.arange(len(table)); ax.bar(x-.15,table.original,.3,label='Original'); ax.bar(x+.15,table.selected_news,.3,label='Frozen news selection')
        ax.set_xticks(x,table.base_model,rotation=55,ha='right',fontsize=8);ax.set_title(region+'\nMean MAE, h=1/3/6/12');ax.set_ylabel('MAE, RUB per resident');ax.legend(fontsize=8)
        ax.set_ylim(0,max(table.original.max(),table.selected_news.max())*1.25)
    fig.tight_layout();(run/'figures').mkdir(exist_ok=True);fig.savefig(run/'figures/regions.png',dpi=150);plt.close(fig)
    fig, axes = plt.subplots(1,len(chunks),figsize=(6*len(chunks),4),squeeze=False)
    for ax,(region,table) in zip(axes[0],primary_all.groupby('region',sort=False)):
        x=np.arange(len(table))
        ax.bar(x-.25,table.original,.25,label='Original')
        ax.bar(x,table.financial_only,.25,label='Financial correction, no news')
        ax.bar(x+.25,table.selected_news,.25,label='Frozen news selection')
        ax.set_xticks(x,table.base_model,rotation=55,ha='right',fontsize=8)
        ax.set_title(region+'\nMean MAE, h=1/3');ax.set_ylabel('MAE, RUB per resident');ax.legend(fontsize=8)
        ax.set_ylim(0,max(table.original.max(),table.financial_only.max(),table.selected_news.max())*1.35)
    fig.tight_layout();fig.savefig(run/'figures/regions_primary.png',dpi=150);plt.close(fig)
    text=f'''# Экономические новости, затухание и перенос между регионами

Запуск `{run.name}`. Период источников 2023–2024; validation январь–июнь 2024 только Оренбург; test июль–декабрь 2024. Основные горизонты 1 и 3 месяца, 6 — дополнительный; h=12 остаётся без поправки из-за отсутствия созревшего обучения при валидации.

## Результат

{conclusions}

SeasonalNaive_NatGrowth, средняя MAE по h=1/3:

{seasonal.to_markdown(index=False,floatfmt='.2f')}

{evidence_text}

## Протокол

Регионы зафиксированы до оценки: Оренбургская область для выбора параметров, Нижегородская область по запросу пользователя, Костромская область по доступности датированного архива. Тверская область рассматривалась, но её проверенный endpoint не предоставил пригодный JSON-архив; Кострома выбрана до расчёта метрик. Архивы НИА Нижний Новгород (публичная форма по неделям) и Кострома.Today (WordPress по месяцам) собираются полностью в рамках выбранных запросов. Это не все новости региона. Разметка — тот же Qwen и неизменный prompt; 30 прежних relevant и 20 other в месяц, детерминированный отбор без расходов. Вес — обратная вероятность отбора.

Экономический отбор требует economic_relevance=1 и экономической темы, явно указанного направления цен/доходов/бизнеса или связи с категорией расходов. Отдельный вариант economic_shocks_windows добавляет явно экономически релевантные ЧС; ЧС без категории действует только на совокупные расходы соответствующей территории. Обычный вариант all_windows уже фильтрует содержание по economic_relevance; новый отбор меняет и корпус, и признаки его объёма. Жёсткий отбор выполняется после LLM-разметки: общий архив сохранён для контроля покрытия.

Вес публикации на конец origin: sampling_weight × 2^(-возраст_в_днях / период_полураспада). Используются 14/30/90 дней; новости после origin исключены. Возраст считается по точному времени публикации; затухание использует всю уже доступную историю с января 2023, окна — 1/3 месяца. Ноль direction — явно без изменений, 9 — неизвестно; признаки доступности разделены. География и категории ограничивают применение новости.

Затухают взвешенные числа публикаций. Доли тональности/направлений и средний сентимент — относительные статистики с теми же весами: один оставшийся заголовок может сохранять знак среднего, пока его интенсивность уменьшается. Это не гарантирует экспоненциального затухания самой прогнозной поправки. Прогноз получает одновременно интенсивность и относительные статистики.

Сначала для всех восьми моделей на Оренбургской validation выбираются alpha, shrink (включая 0), лучший decay и лучший вариант окна/decay. Затем записываются frozen_selection.csv, frozen_representations.csv и их SHA256; только после этого считаются тестовые метрики. На двух других регионах эти настройки применяются без подбора. Коэффициенты Ridge обучаются локально только по уже созревшим прошлым ошибкам; это перенос настроек, не перенос коэффициентов. Контроли покрытия имеют тот же экономический отбор, географию и период затухания, но исключают сентимент и направления.

Для Ensemble на h=1/3 Оренбургская validation выбрала shrink=0. Поэтому на этих горизонтах прогноз совпадает с исходным и в двух других областях по зафиксированному протоколу. Такое совпадение не доказывает бесполезность всех возможных региональных новостных моделей. На h=6 поправка ненулевая; результат показан отдельно, включая ухудшение на Оренбургском тесте. Поисковый архив НИА содержит и недатированные карточки людей: они учитываются в полноте загрузки выдачи, но исключаются из новостей, даты им не приписываются; число исключений сохранено в collection_nizhny.json.

![Основные горизонты и финансовый контроль](../{run.relative_to(ROOT)}/figures/regions_primary.png)

![Все горизонты](../{run.relative_to(ROOT)}/figures/regions.png)

'''+ '\n'.join(chunks)+f'''
## Ограничения

Все сравнения исследовательские: Оренбургский тест ранее просмотрен; исходные прогнозы всех регионов получены в прежнем общем эксперименте, исходный Ensemble наследует ретроспективные ограничения весов. Проверка новых новостных признаков на другом регионе не является полностью независимым будущим тестом. Всего 24 месяца и шесть тестовых месяцев; bootstrap не устраняет временную зависимость и множественные сравнения. Современная LLM может знать последующие события; разметка по заголовкам не прошла независимую ручную оценку. Историческая доступность предполагается по публикации, сбор и разметка выполнены в 2026. Вес по времени — гипотеза; условный географический отбор теряет новости без явного топонима.

## Воспроизведение

В активированном окружении sber: `python scripts/collect_transfer_news.py` и `python scripts/collect_news_archive.py --config configs/news_archive_kostroma.json`; затем `python scripts/annotate_llm_news.py --config configs/llm_news_nizhny.json` и аналогично configs/llm_news_kostroma.json. `python scripts/evaluate_news_decay.py` создаёт Оренбургский запуск и фиксирует выбор. Затем `python scripts/evaluate_news_decay.py --run {run.relative_to(ROOT)} --regions nizhny kostroma` переносит настройки. Если Make установлен, сбор и разметку объединяют цели `news-transfer-collect` и `news-transfer-annotate`. В текущей среде Make отсутствует, эксперимент выполнен командами Python. Входы, ответы, кеш и подробные прогнозы игнорируются Git; итоговые метрики, график и зафиксированные параметры сохраняются. Исходные прогнозы не изменяются.
'''
    (ROOT/'docs/NEWS_DECAY_TRANSFER_RU.md').write_text(text)
    (run/'results_report.md').write_text(text.replace(f'../{run.relative_to(ROOT)}/figures/','figures/'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/news_decay.json')
    parser.add_argument('--run')
    parser.add_argument('--regions',nargs='+',choices=list(REGIONS),default=['orenburg'])
    args=parser.parse_args()
    if args.run:
        run=ROOT/args.run;cfg=json.loads((run/'config.json').read_text())
        if 'orenburg' in args.regions:
            raise ValueError('Do not retune an existing frozen run')
    else:
        if args.regions!=['orenburg']:
            raise ValueError('Start with Orenburg to freeze settings')
        cfg=json.loads((ROOT/args.config).read_text())
        run=ROOT/'artifacts/news_decay_runs'/datetime.now(timezone.utc).strftime('decay_%Y%m%dT%H%M%S_%fZ')
        run.mkdir(parents=True);(run/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
        (run/'code_manifest.json').write_text(json.dumps({p:hash_file(ROOT/p) for p in ['scripts/evaluate_news_decay.py','scripts/evaluate_llm_news.py','src/llm_news.py']},indent=2)+'\n')
    full,_=load_municipal(ROOT)
    for region in args.regions:
        evaluate_region(run,region,full,cfg)
        report(run)
    (run.parent/'latest.json').write_text(json.dumps({'directory':str(run.relative_to(ROOT))})+'\n')
    print('Results:',run,flush=True)


if __name__=='__main__':
    main()
