"""Исследовательская проверка LLM-признаков поверх сохранённых моделей.

Подбор поправки только на validation, включая нулевую поправку. Обучение
поправок на каждом origin использует только созревшие исторические ошибки.
Ограничения исходного Ensemble наследуются, PastOnlyBlend показан отдельно.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from src.llm_news import aggregate_features, feature_names, FIELDS
from src.mun_data import load_municipal, MunicipalPanel
from src.mun_eval import HORIZONS, make_grid, score_model, month_bootstrap_diff, per_series_mae
from src.news_impact import news_inputs


def financial_features(panel, base, origin, h):
    """Общие контрольные признаки, без статического среза 2024 и будущих фактов."""
    level = np.log(base[origin, h].clip(1e-6))
    current = np.log(panel.values[:, origin])
    columns = [np.eye(6)[panel.category_code], np.full((panel.n_series, 1), h/12),
               level[:, None], (current-level)[:, None]]
    for lag in (1, 3, 12):
        delta = current-np.log(panel.values[:, origin-lag]) if origin >= lag else np.zeros(panel.n_series)
        columns.append(delta[:, None])
    phase = 2*np.pi*(panel.months[origin].month+h-1)/12
    columns.append(np.tile([np.sin(phase), np.cos(phase)], (panel.n_series, 1)))
    # Ошибка ранее выданного h=1 прогноза, уже созревшая на текущем origin.
    previous = base[origin-1, 1] if origin > 5 else np.full(panel.n_series, np.nan)
    error = np.log(panel.values[:, origin]/previous.clip(1e-6))
    columns.append(np.nan_to_num(error, nan=0.)[:, None])
    return np.column_stack(columns)


def fit_corrections(panel, base, extras, cfg):
    """Одинаковая Ridge-поправка для каждого варианта; дрейф ограничен обученным h."""
    variants = {}
    audits = []
    for variant, extra in extras.items():
        for alpha in cfg['ridge_alphas']:
            drift = np.zeros_like(base)
            for origin in range(6, 23):
                xs, ys, weights, targets, horizons = [], [], [], [], []
                excluded_nonpositive = 0
                for a in range(5, origin):
                    for h in HORIZONS:
                        if a+h > origin:
                            continue
                        valid = np.isfinite(base[a, h]) & (base[a, h] > 0)
                        excluded_nonpositive += int((~valid).sum())
                        if not valid.any():
                            continue
                        X = financial_features(panel, base, a, h)
                        if extra is not None:
                            X = np.column_stack([X, extra[a]])
                        xs.append(X[valid])
                        ys.append(np.log(panel.values[valid, a+h]/base[a, h, valid])/h)
                        weights.append(np.full(valid.sum(), h))
                        targets.append(a+h)
                        horizons.append(h)
                if len(xs) < cfg['minimum_training_pairs']:
                    continue
                model = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                                      StandardScaler(), Ridge(alpha=alpha))
                model.fit(np.vstack(xs), np.concatenate(ys), ridge__sample_weight=np.concatenate(weights))
                for h in HORIZONS:
                    if not np.isfinite(base[origin, h]).all():
                        continue
                    X = financial_features(panel, base, origin, h)
                    if extra is not None:
                        X = np.column_stack([X, extra[origin]])
                    drift[origin, h] = model.predict(X)*min(h, max(horizons))
                audits.append({'variant': variant, 'alpha': alpha, 'origin': origin, 'training_pairs': len(xs),
                               'last_training_target': max(targets), 'maximum_trained_h': max(horizons),
                               'excluded_nonpositive_training_forecasts': excluded_nonpositive})
            for shrink in cfg['correction_shrinks']:
                key = (variant, alpha, shrink)
                variants[key] = base*np.exp(np.clip(shrink*drift, -cfg['maximum_log_correction'], cfg['maximum_log_correction']))
    return variants, audits


def select_validation(panel, variants, grid):
    rows = []
    for (variant, alpha, shrink), prediction in variants.items():
        metrics = score_model(panel, prediction, grid, 'validation', variant)
        for row in metrics.itertuples():
            rows.append({'variant': variant, 'alpha': alpha, 'shrink': shrink, 'h': row.h, 'validation_mae': row.mae})
    table = pd.DataFrame(rows)
    chosen = table.sort_values(['validation_mae', 'shrink', 'alpha']).groupby(['variant', 'h'], sort=False).head(1)
    selected = {}
    for variant in chosen.variant.unique():
        result = variants[next(key for key in variants if key[0] == variant)].copy()
        for row in chosen[chosen.variant == variant].itertuples():
            result[:, row.h] = variants[(variant, row.alpha, row.shrink)][:, row.h]
        selected[variant] = result
    return selected, chosen, table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/llm_news.json')
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    cfg_path = ROOT/args.config
    cfg = json.loads(cfg_path.read_text())
    folder = ROOT/cfg['output_directory']/('full' if args.full else 'pilot')
    news_path = folder/'annotated_news.csv'
    records = pd.read_csv(news_path, dtype={'territory_ids': 'string'})
    if records[FIELDS].isna().any().any():
        raise ValueError('Incomplete annotation')
    manifest = json.loads((folder/'manifest.json').read_text())
    full, _ = load_municipal(ROOT)
    ids = np.where(full.meta.region_code.to_numpy() == cfg['region_code'])[0]
    panel = MunicipalPanel(full.values[ids], full.meta.iloc[ids], full.months)
    array, availability = aggregate_features(panel, records)
    delayed, delayed_audit = aggregate_features(panel, records, delay_months=3)
    source_records = pd.read_csv(ROOT/cfg['news_file'], dtype={'territory_ids': 'string'})
    rules, _, _, _ = news_inputs(panel, source_records)
    # LLM content compared to coverage of exactly the same sampled corpus.
    coverage_indices = [i for i, name in enumerate(feature_names()) if name.startswith('log_known_articles')]
    coverage = array[:, :, coverage_indices]
    extras = {'financial_only': None, 'coverage_only': coverage, 'rules_full_corpus': rules,
              'llm': array, 'llm_delayed_3m': delayed}
    stamp = datetime.now(timezone.utc).strftime('llm_%Y%m%dT%H%M%S_%fZ')
    run = ROOT/'artifacts/llm_news_runs'/stamp
    run.mkdir(parents=True)
    (run/'config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2)+'\n')
    (run/'feature_schema.json').write_text(json.dumps(feature_names(), ensure_ascii=False, indent=2)+'\n')
    np.save(run/'features.npy', array)
    pd.concat([availability, delayed_audit]).to_csv(run/'availability_audit.csv', index=False)
    panel.meta.to_csv(run/'series_index.csv', index=False)
    feature_rows = []
    for t in range(panel.n_months):
        for k, name in enumerate(feature_names()):
            feature_rows.append({'month': panel.months[t], 'feature': name, 'mean': array[t, :, k].mean(),
                                 'nonzero_series': np.count_nonzero(array[t, :, k])})
    pd.DataFrame(feature_rows).to_csv(run/'feature_summary.csv', index=False)
    annotation_summary = records.groupby(['month', 'economic_relevance', 'sentiment']).size().reset_index(name='articles')
    annotation_summary.to_csv(run/'annotation_summary.csv', index=False)
    source = ROOT/json.loads((ROOT/'artifacts/municipal_runs/latest.json').read_text())['directory']
    research = ROOT/json.loads((ROOT/'artifacts/research_runs/latest.json').read_text())['directory']
    grid = make_grid()
    all_metrics, category_metrics, win_shares, selections, developments, audits, bootstrap, predictions = [], [], [], [], [], [], [], []
    cached_inputs = {}
    with threadpool_limits(limits=1):
        for base_name in cfg['base_models']:
            path = (research if base_name == 'PastOnlyBlend' else source)/f'pred_{base_name}.npy'
            base = np.load(path, mmap_mode='r')[:, :, ids].astype(float)
            cached_inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            candidates, audit = fit_corrections(panel, base, extras, cfg)
            selected, selection, development = select_validation(panel, candidates, grid)
            selection['base_model'] = base_name
            development['base_model'] = base_name
            selections.append(selection)
            developments.append(development)
            audits.extend({'base_model': base_name, **row} for row in audit)
            # Freeze selection before computing test metrics.
            pd.concat(selections).to_csv(run/'frozen_selection.csv', index=False)
            for name, prediction in {'original': base, **selected}.items():
                model_name = base_name if name == 'original' else base_name+'+'+name
                for stage in ('validation', 'test'):
                    all_metrics.append(score_model(panel, prediction, grid, stage, model_name))
                for category in panel.meta.category.unique():
                    group_ids = np.flatnonzero(panel.meta.category.to_numpy() == category)
                    group_panel = MunicipalPanel(panel.values[group_ids], panel.meta.iloc[group_ids], panel.months)
                    group_metrics = score_model(group_panel, prediction[:, :, group_ids], grid, 'test', model_name)
                    group_metrics['category'] = category
                    category_metrics.append(group_metrics)
                if name != 'original':
                    for h in HORIZONS:
                        d, lo, hi, n = month_bootstrap_diff(panel, prediction, base, grid, 'test', h,
                                                          cfg['bootstrap_replicates'], cfg['seed'])
                        bootstrap.append({'base_model': base_name, 'variant': name, 'h': h, 'comparator': 'original',
                                          'mae_difference': d, 'month_ci_low': lo, 'month_ci_high': hi, 'target_months': n})
                if name == 'llm':
                    for h in HORIZONS:
                        errors = pd.DataFrame({'territory_id': panel.meta.territory_id,
                            'original': per_series_mae(panel, base, grid, 'test', h),
                            'llm': per_series_mae(panel, prediction, grid, 'test', h)}).groupby('territory_id').mean()
                        win_shares.append({'base_model': base_name, 'h': h, 'municipalities': len(errors),
                            'improved_share': float((errors.llm < errors.original).mean()),
                            'unchanged_share': float(np.isclose(errors.llm, errors.original).mean())})
                    for comparator in ('financial_only', 'coverage_only', 'rules_full_corpus', 'llm_delayed_3m'):
                        for h in HORIZONS:
                            d, lo, hi, n = month_bootstrap_diff(panel, prediction, selected[comparator], grid, 'test', h,
                                                              cfg['bootstrap_replicates'], cfg['seed'])
                            bootstrap.append({'base_model': base_name, 'variant': name, 'h': h, 'comparator': comparator,
                                              'mae_difference': d, 'month_ci_low': lo, 'month_ci_high': hi, 'target_months': n})
                if name in ('original', 'llm', 'financial_only'):
                    for row in grid[grid.stage == 'test'].itertuples():
                        for i in range(panel.n_series):
                            predictions.append({'model': model_name, 'origin': row.origin, 'target': row.target, 'h': row.h,
                                                'series_idx': i, 'actual': panel.values[i, row.target], 'prediction': prediction[row.origin, row.h, i]})
            print('Evaluated:', base_name, flush=True)
    metrics = pd.concat(all_metrics)
    metrics.to_csv(run/'metrics.csv', index=False)
    pd.concat(category_metrics).to_csv(run/'category_metrics.csv', index=False)
    pd.DataFrame(win_shares).to_csv(run/'municipal_win_shares.csv', index=False)
    pd.concat(developments).to_csv(run/'development_metrics.csv', index=False)
    pd.DataFrame(audits).to_csv(run/'training_audit.csv', index=False)
    pd.DataFrame(bootstrap).to_csv(run/'paired_bootstrap.csv', index=False)
    pd.DataFrame(predictions).to_csv(run/'predictions.csv', index=False)
    (run/'manifest.json').write_text(json.dumps({'run': str(run.relative_to(ROOT)), 'annotation': manifest,
        'annotation_sha256': hashlib.sha256(news_path.read_bytes()).hexdigest(), 'cached_prediction_sha256': cached_inputs,
        'config_sha256': hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
        'code_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['src/llm_news.py', 'scripts/annotate_llm_news.py', 'scripts/evaluate_llm_news.py']},
        'n_series': panel.n_series, 'selection_saved_before_test_metrics': True,
        'limitations': ['Viewed test; exploratory comparisons', 'Modern LLM may know later events',
                        'Pilot inverse-probability estimates are noisy' if not args.full else 'Current titles, historical revisions unknown',
                        'Original cached Ensemble has retrospective weight-selection limitations',
                        'One region and six target months; no nationwide improvement claim', 'No independent human annotation audit']}, ensure_ascii=False, indent=2)+'\n')
    audit_table = pd.DataFrame(audits)
    dates = pd.to_datetime(availability.latest_available_at, utc=True)
    cutoff = pd.to_datetime(availability.cutoff_exclusive, utc=True)
    checks = {'complete_annotation': bool(len(records) == manifest['selected_articles']),
              'past_only_news': bool((dates.isna() | (dates < cutoff)).all()),
              'mature_training_targets': bool((audit_table.last_training_target <= audit_table.origin).all()),
              'finite_features': bool(np.isfinite(array).all()),
              'full_prediction_coverage': bool((metrics.coverage == 1).all())}
    (run/'verification.json').write_text(json.dumps({'passed': all(checks.values()), 'checks': checks}, indent=2)+'\n')
    if not all(checks.values()):
        raise RuntimeError(checks)
    (run.parent/'latest.json').write_text(json.dumps({'directory': str(run.relative_to(ROOT))})+'\n')
    build_report(run, metrics, records, cfg)
    print('Results:', run, flush=True)


def build_report(run, metrics, records, cfg):
    test = metrics[metrics.stage == 'test'].pivot(index='model', columns='h', values='mae')
    rows = []
    for name in cfg['base_models']:
        original, llm = test.loc[name], test.loc[name+'+llm']
        financial = test.loc[name+'+financial_only']
        rows.append({'model': name, 'baseline_mean_mae': original.mean(),
                     'financial_control_mean_mae': financial.mean(), 'llm_mean_mae': llm.mean(),
                     'change_pct': 100*(llm.mean()/original.mean()-1),
                     'delta_vs_financial_control': llm.mean()-financial.mean()})
    summary = pd.DataFrame(rows)
    summary.to_csv(run/'summary.csv', index=False)
    figures = run/'figures'
    figures.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(summary))
    ax.bar(x-.25, summary.baseline_mean_mae, .25, label='Original')
    ax.bar(x, summary.financial_control_mean_mae, .25, label='Financial correction without news')
    ax.bar(x+.25, summary.llm_mean_mae, .25, label='LLM correction, validation-selected')
    ax.set_xticks(x, summary.model, rotation=20, ha='right')
    ax.set_ylabel('MAE, RUB per resident')
    ax.set_title('Orenburg, 2024H2: exploratory LLM news features')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures/'model_comparison.png', dpi=150)
    plt.close(fig)
    chosen = pd.read_csv(run/'frozen_selection.csv')
    llm_choices = chosen[chosen.variant == 'llm']
    bootstrap = pd.read_csv(run/'paired_bootstrap.csv')
    content = bootstrap[(bootstrap.variant == 'llm') & (bootstrap.comparator.isin(['financial_only', 'coverage_only']))]
    category = pd.read_csv(run/'category_metrics.csv')
    category = category[category.model.isin(['Ensemble', 'Ensemble+llm', 'Ensemble+financial_only', 'Ensemble+coverage_only'])]
    category = category.pivot(index=['category', 'model'], columns='h', values='mae').reset_index()
    wins = pd.read_csv(run/'municipal_win_shares.csv')
    best = summary.loc[summary.baseline_mean_mae.idxmin()]
    gains = summary[summary.change_pct < -1e-9].sort_values('change_pct')
    conclusion = (f"Лучшая исходная модель — **{best['model']}**: средняя MAE "
                  f"{best.baseline_mean_mae:.2f} → {best.llm_mean_mae:.2f} руб./жителя "
                  f"({best.change_pct:+.2f}%).")
    if best['model'] == 'Ensemble' and best.change_pct == 0:
        conclusion += " Валидация выбрала shrink=0 на всех горизонтах: LLM-поправка не применяется."
    if not gains.empty:
        gain = gains.iloc[0]
        conclusion += (f" Максимальное среднее улучшение — {gain['model']}: "
                       f"{gain.baseline_mean_mae:.2f} → {gain.llm_mean_mae:.2f} "
                       f"({gain.change_pct:+.2f}%). Это не улучшение лучшего ансамбля.")
    supported = bootstrap[(bootstrap.variant == 'llm') &
                          (bootstrap.comparator == 'original') &
                          (bootstrap.month_ci_high < 0)]
    for _, result in supported.iterrows():
        controls = content[(content.base_model == result.base_model) & (content.h == result.h)]
        conclusion += (f" {result.base_model}, h={int(result.h)}: bootstrap поддерживает "
                       "снижение относительно исходной модели.")
        if len(controls) == 2 and (controls.month_ci_high >= 0).all() and (controls.month_ci_low <= 0).all():
            conclusion += (" Но интервалы сравнения с финансовым контролем и контролем "
                           "объёма включают ноль; добавочная польза содержания не доказана.")
    text = f'''# LLM-разметка новостей и прогнозы

Запуск `{run.name}`. Qwen через локальный llama.cpp; разметка только заголовков. Обработано **{len(records)} публикаций**, {records.title_sha256.nunique()} уникальных заголовков, за {records.month.nunique()} месяцев. Регион: Оренбургская область, 39 МО × 6 категорий = 234 ряда. Все сравнения исследовательские: тест уже просмотрен, современная LLM может знать последующие события.

## Сравнение с исходными моделями

{conclusion}

Средняя MAE по h=1/3/6/12; отрицательное изменение — улучшение.

{summary.to_markdown(index=False, floatfmt='.2f')}

![Сравнение моделей](../{run.relative_to(ROOT)}/figures/model_comparison.png)

Полная таблица по горизонтам:

{test.reset_index().to_markdown(index=False, floatfmt='.2f')}

## Региональные категории и муниципалитеты

Все числа относятся только к Оренбургской области. Разрез основной модели по категориям, MAE:

{category.to_markdown(index=False, floatfmt='.2f')}

Доля муниципалитетов с уменьшением MAE, усреднённой по шести категориям и целевым месяцам:

{wins.to_markdown(index=False, floatfmt='.3f')}

## Что именно проверено

Для каждой исходной модели проверена одинаковая Ridge-поправка по созревшим ошибкам: только финансовые признаки, контроль объёма выборки, существующие правила на полном корпусе, LLM-признаки, LLM с задержкой 3 месяца. Alpha и shrink выбираются отдельно по модели/горизонту на validation; shrink=0 разрешает оставить исходный прогноз. Поэтому улучшение против оригинала само по себе не доказывает вклад содержания: важны сравнения с financial_only и coverage_only. Нулевая выбранная поправка означает, что validation отвергла применение LLM на этом горизонте.

На h=12 в validation только один целевой месяц с origin=5: в этот момент нет созревших ошибок для обучения поправки. Все кандидаты совпадают с исходной моделью, при равенстве выбирается shrink=0. Поэтому h=12 остаётся без поправки; это отсутствие данных для выбора, а не доказательство бесполезности LLM на годовом горизонте.

Нулевые/неположительные исходные прогнозы исключаются только из обучения логарифмической поправки; число исключений записано в training_audit.csv. В оценке остаются все ряды и исходные нулевые прогнозы: мультипликативная поправка не заменяет их искусственным положительным значением.

Исходный Ensemble наследует ретроспективные ограничения весов. PastOnlyBlend показан отдельно; поздний подбор гиперпараметров по validation также не превращает опыт в строгий исторический онлайн-тест. Для такого теста понадобится новый период или последовательный подбор.

Выбранные параметры LLM-поправки:

{llm_choices.to_markdown(index=False, floatfmt='.3f')}

## Добавочная польза содержания

Разность MAE LLM минус контроль, 95% bootstrap шести целевых месяцев. Временная зависимость, малое число месяцев и множественные сравнения ограничивают интерпретацию.

{content.to_markdown(index=False, floatfmt='.3f')}

## Признаки и ограничения

36 признаков за 1 и 3 месяца: объём известной выборки, экономическая доля, число сообщений с совпадающей географией/категорией, доли негативной/позитивной/смешанной/неизвестной тональности, средний сентимент, направления цен/доходов/бизнеса с долями известности, доля чрезвычайных событий. Объяснения LLM не подаются в модель. География берётся из существующего сопоставления; неразрешённые топонимы не распространяются на область. Housing без категории не становится расходами на продовольствие/транспорт.

Выборка пилота стратифицирована по месяцу и прежнему признаку relevant, без финансовых значений. Обратные вероятности компенсируют разные доли отбора, но не устраняют шум и не гарантируют полного охвата новостей. Для полного корпуса используется вес 1. Разметка не прошла независимую ручную оценку; correctness JSON не означает correctness смысла. Историческая доступность принимается по публикации, фактический сбор и LLM-разметка выполнены в 2026 году.

## Воспроизведение

`python scripts/annotate_llm_news.py` — разметка с кешем и возобновлением; `--full` для всего корпуса, `--base-url` для другого адреса сервера. Затем `python scripts/evaluate_llm_news.py` (тот же `--full` при полном корпусе). Настройки: `configs/llm_news.json`. Сырые ответы и разметка: `data/inputs/llm_news/`, исключены из Git. Итоговые метрики, выбор и манифест: `{run.relative_to(ROOT)}`. Исходные прогнозы и frozen-файлы не менялись.
'''
    (ROOT/'docs/LLM_NEWS_RESULTS_RU.md').write_text(text)
    local_text = text.replace(f'../{run.relative_to(ROOT)}/figures/', 'figures/')
    (run/'results_report.md').write_text(local_text)


if __name__ == '__main__':
    main()
