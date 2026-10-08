"""Сравнение прогнозов и детекторов с региональными новостями, один CPU-поток."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits
from src.mun_data import load_municipal, MunicipalPanel, NationalPriors
from src.mun_models import Context, feature_matrix, training_set, base_forecast_log
from src.mun_eval import HORIZONS, make_grid, score_model, month_bootstrap_diff, cluster_bootstrap_diff, per_series_mae
from src.mun_shocks import run_detector_matrix
from src.transitions import offline_mean_changes, match_events
from src.news_impact import news_inputs, amplify_financial_input, feature_names
from src.news_collection import CRISIS_TOPICS, ECONOMIC_RULES


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def linear_features(X):
    # У категории нет линейного порядка: заменяем числовой код шестью индикаторами.
    return np.column_stack([np.delete(X, 1, axis=1), np.eye(6)[X[:, 1].astype(int)]])


def forecasts(panel, nat, cfg, arrays, run):
    grid = make_grid()
    out, metrics, audit, coefficients = {}, [], [], []
    base_ctx = Context(panel, nat)
    base = np.full((24, 13, panel.n_series), np.nan)
    for o in range(5, 24):
        for h in range(1, 13):
            base[o, h] = np.exp(base_forecast_log(base_ctx, o, h))
    out['NatPath_K3'] = base
    params = cfg['forecast']
    for name in params['variants']:
        ctx = Context(panel, nat)
        array = arrays.get(name)
        if array is not None:
            ctx.extra = lambda a, arr=array: arr[a]
        prediction = base.copy()
        for o in range(5, 23):
            train = training_set(ctx, o, weight_power=0.0, max_rows=params['maximum_training_rows'], seed=cfg['seed'])
            if train is None or len(train[1]) < params['minimum_training_rows']:
                continue
            X, y, weights, pairs = train
            model = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                                  StandardScaler(), Ridge(alpha=params['alpha'], solver='cholesky'))
            model.fit(linear_features(X), y, ridge__sample_weight=weights)
            if array is not None:
                # В linear_features удалён cat и в конец добавлены 6 one-hot.
                offset = X.shape[1] - array.shape[2] - 1
                for k, value in enumerate(model[-1].coef_[offset:offset+array.shape[2]]):
                    coefficients.append({'variant': name, 'origin': o, 'feature_index': k,
                                         'standardized_coefficient': float(value)})
            for h in HORIZONS:
                Xnew, _ = feature_matrix(ctx, o, h)
                drift = model.predict(linear_features(Xnew))
                correction = np.clip(params['shrink'] * drift * min(h, o-5),
                                     -params['maximum_abs_log_correction'], params['maximum_abs_log_correction'])
                prediction[o, h] *= np.exp(correction)
            audit.append({'variant': name, 'origin': o, 'rows': len(y),
                          'maximum_training_target': max(a+h for a,h in pairs),
                          'alpha': params['alpha'], 'shrink': params['shrink']})
        out[name] = prediction
        print('Прогноз:', name, flush=True)
    for name, pred in out.items():
        for stage in ['validation', 'test']:
            metrics.append(score_model(panel, pred, grid, stage, name))
    pd.concat(metrics).to_csv(run/'forecast_metrics.csv', index=False)
    pd.DataFrame(audit).to_csv(run/'forecast_training_audit.csv', index=False)
    pd.DataFrame(coefficients).to_csv(run/'news_coefficients.csv', index=False)
    rows = []
    for name in params['variants'][1:]:
        for h in HORIZONS:
            d, lo, hi, n = month_bootstrap_diff(panel, out[name], out['without_news'], grid, 'test', h,
                                              cfg['bootstrap_replicates'], cfg['seed'])
            a = per_series_mae(panel, out[name], grid, 'test', h)
            b = per_series_mae(panel, out['without_news'], grid, 'test', h)
            _, clo, chi = cluster_bootstrap_diff(a,b,panel.meta.territory_id.to_numpy(),cfg['bootstrap_replicates'],cfg['seed'])
            rows.append({'variant':name,'h':h,'delta_mae_vs_without_news':d,'month_ci_low':lo,'month_ci_high':hi,
                         'municipality_ci_low':clo,'municipality_ci_high':chi,'target_months':n})
    pd.DataFrame(rows).to_csv(run/'forecast_bootstrap.csv',index=False)
    controls=[]
    for h in HORIZONS:
        d,lo,hi,n=month_bootstrap_diff(panel,out['with_news'],out['coverage_only'],grid,'test',h,
                                     cfg['bootstrap_replicates'],cfg['seed'])
        controls.append({'h':h,'delta_mae_news_vs_coverage_only':d,'month_ci_low':lo,'month_ci_high':hi})
    pd.DataFrame(controls).to_csv(run/'news_vs_coverage_bootstrap.csv',index=False)
    details=[]
    for row in grid[grid.stage=='test'].itertuples():
        for variant in params['variants']:
            for i in range(panel.n_series):
                details.append({'variant':variant,'origin':row.origin,'target':row.target,'h':row.h,
                                'territory_id':panel.meta.territory_id[i],'category':panel.meta.category[i],
                                'actual':panel.values[i,row.target],'prediction':out[variant][row.origin,row.h,i]})
    pd.DataFrame(details).to_csv(run/'forecast_predictions.csv',index=False)
    return out


def detectors(panel, z, signals, cfg, run):
    params=cfg['detectors']
    chosen, tuning, alarms={},[],{}
    lo,hi=params['calibration_months']
    # Каждый вариант калибруется отдельно, один и тот же предельный бюджет.
    for variant in params['variants']:
        zv=amplify_financial_input(z,signals[variant],params['news_gain'])
        for method in params['methods']:
            threshold=np.inf
            for candidate in params['thresholds'][method]:
                acal=run_detector_matrix(zv[:,:hi+1],method,candidate,cooldown=params['cooldown_months'])
                rate=100*acal[:,lo:hi+1].mean()
                tuning.append({'variant':variant,'method':method,'threshold':candidate,
                               'calibration_alarm_rate_per_100':rate,'within_budget':bool(rate<=params['alarm_budget_per_100_series_months'])})
                if rate<=params['alarm_budget_per_100_series_months']:
                    threshold=candidate
                    break
            key=(variant,method)
            chosen[key]=threshold
            alarms[key]=run_detector_matrix(zv,method,threshold,cooldown=params['cooldown_months'])
        print('Детекторы:',variant,flush=True)
    pd.DataFrame(tuning).to_csv(run/'detector_threshold_tuning.csv',index=False)
    selected=[]
    for (v,m),threshold in chosen.items():
        a=alarms[v,m]
        selected.append({'variant':v,'method':m,'threshold':threshold,
                         'calibration_alarm_rate_per_100':100*a[:,lo:hi+1].mean(),
                         'test_alarm_rate_per_100':100*a[:,18:24].mean(),
                         'test_alarms':int(a[:,18:24].sum())})
    pd.DataFrame(selected).to_csv(run/'detector_selection.csv',index=False)
    valid=np.where(np.isfinite(z).all(0))[0]
    labels, metrics=[],[]
    for mult in params['proxy_penalties']:
        events={}
        for sid in range(panel.n_series):
            values=z[sid,valid]
            cuts=offline_mean_changes(values,mult*np.log(len(values)),params['proxy_min_segment'])
            keep=[int(valid[c]) for c in cuts if abs(values[:c].mean()-values[c:].mean())>=params['proxy_min_shift_sigma']]
            events[sid]=[t for t in keep if 18<=t<=23]
            for t in events[sid]:
                labels.append({'penalty':mult,'series':sid,'territory_id':panel.meta.territory_id[sid],
                               'category':panel.meta.category[sid],'month_index':t,'label_kind':'offline_financial_proxy_not_verified_shock'})
        for (variant,method),a in alarms.items():
            tp=total_alarms=total_events=0
            delays=[]
            for sid,evs in events.items():
                hits=match_events(evs,np.where(a[sid,18:24])[0]+18,params['max_delay_months'])
                tp+=len(hits); total_events+=len(evs); total_alarms+=int(a[sid,18:24].sum())
                delays.extend(b-e for e,b in hits)
            precision=tp/total_alarms if total_alarms else 0.
            recall=tp/total_events if total_events else np.nan
            f1=2*precision*recall/(precision+recall) if np.isfinite(recall) and precision+recall else (0. if total_events else np.nan)
            metrics.append({'variant':variant,'method':method,'proxy_penalty':mult,'n_proxy_events':total_events,
                            'n_alarms':total_alarms,'matched_proxy_events':tp,'proxy_precision':precision,'proxy_recall':recall,
                            'proxy_f1':f1,'median_delay_months':np.median(delays) if delays else np.nan,
                            'unmatched_alarms_per_100':100*(total_alarms-tp)/(panel.n_series*6),
                            'label_kind':'offline_financial_proxy_not_verified_shock'})
    pd.DataFrame(labels,columns=['penalty','series','territory_id','category','month_index','label_kind']).to_csv(run/'detector_proxy_labels.csv',index=False)
    pd.DataFrame(metrics).to_csv(run/'detector_metrics.csv',index=False)
    details=[]
    for (v,m),a in alarms.items():
        for sid,t in zip(*np.where(a)):
            details.append({'variant':v,'method':m,'territory_id':panel.meta.territory_id[sid],
                            'category':panel.meta.category[sid],'month_index':t,'month':panel.months[t],
                            'base_z':z[sid,t],'news_signal':signals[v][sid,t],
                            'input_z':z[sid,t]*(1+params['news_gain']*signals[v][sid,t])})
    pd.DataFrame(details).to_csv(run/'detector_alarms.csv',index=False)
    # Локальный смысл изменившихся тревог; новости не служат эталонными метками.
    changes=[]
    for method in params['methods']:
        before,after=alarms['without_news',method],alarms['with_news',method]
        for sid,t in zip(*np.where(before!=after)):
            changes.append({'method':method,'territory_id':panel.meta.territory_id[sid],
                            'mo_name':panel.meta.mo_name[sid],'category':panel.meta.category[sid],
                            'month_index':t,'month':panel.months[t], 'baseline_alarm':bool(before[sid,t]),
                            'news_alarm':bool(after[sid,t]),'news_signal':signals['with_news'][sid,t]})
    pd.DataFrame(changes,columns=['method','territory_id','mo_name','category','month_index','month','baseline_alarm','news_alarm','news_signal']).to_csv(run/'detector_changed_alarms.csv',index=False)


def main():
    started=time.time()
    cfg_path=ROOT/'configs/news_impact.json'
    cfg=json.loads(cfg_path.read_text())
    stamp=datetime.now(timezone.utc).strftime('news_%Y%m%dT%H%M%S_%fZ')
    run=ROOT/'artifacts/regional_news_runs'/stamp
    run.mkdir(parents=True)
    (run/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
    full,_=load_municipal(ROOT)
    ids=np.where(full.meta.region_code.to_numpy()==cfg['region_code'])[0]
    panel=MunicipalPanel(full.values[ids],full.meta.iloc[ids],full.months)
    nat_path=ROOT/json.loads((ROOT/'configs/national_forecast.json').read_text())['input']
    nat=NationalPriors.from_file(nat_path)
    news=pd.read_csv(ROOT/cfg['news_file'], dtype={'territory_ids': 'string'})
    real,cov,signal,audit=news_inputs(panel,news)
    delayed,_,delayed_signal,delayed_audit=news_inputs(panel,news,3)
    crisis,_,crisis_signal,_=news_inputs(panel,news,allowed_topics=CRISIS_TOPICS)
    economic,_,economic_signal,_=news_inputs(panel,news,allowed_topics=list(ECONOMIC_RULES))
    mchs=news[news.source.str.contains('МЧС',na=False)]
    mchs_features,_,mchs_signal,_=news_inputs(panel,mchs)
    pd.concat([audit,delayed_audit]).to_csv(run/'news_availability_audit.csv',index=False)
    panel.meta.to_csv(run/'series_index.csv',index=False)
    feature_rows=[]
    for o in range(panel.n_months):
        feature_rows.append({'origin':o,'month':panel.months[o],
                             'municipalities_with_news':panel.meta.loc[signal[:,o]>0,'territory_id'].nunique(),
                             'mean_news_features':float(real[o].mean())})
    pd.DataFrame(feature_rows).to_csv(run/'news_feature_summary.csv',index=False)
    arrays={'coverage_only':cov,'with_news':real,'delayed_news_3m':delayed,
            'crisis_only':crisis,'economic_only':economic,'mchs_only':mchs_features}
    names=feature_names(news)
    (run/'news_feature_schema.json').write_text(json.dumps({
        'with_news':names,'delayed_news_3m':names,'crisis_only':names,'economic_only':names,
        'coverage_only':names[:cov.shape[2]],'mchs_only':feature_names(mchs)},ensure_ascii=False,indent=2)+'\n')
    with threadpool_limits(limits=1):
        forecasts(panel,nat,cfg,arrays,run)
        z=np.load(ROOT/'artifacts/shocks/z.npy')[ids]
        signals={'without_news':np.zeros_like(signal),'coverage_only':np.repeat((cov[:,:,0]>0).T.astype(float),1,axis=0),
                 'with_news':signal,'delayed_news_3m':delayed_signal,
                 'crisis_only':crisis_signal,'economic_only':economic_signal,'mchs_only':mchs_signal}
        pd.DataFrame([{'variant':name,'month':panel.months[t],
                       'fraction_series_with_signal':float(values[:,t].mean()),
                       'same_as_coverage':bool(np.array_equal(values[:,t],signals['coverage_only'][:,t]))}
                      for name,values in signals.items() for t in range(panel.n_months)]).to_csv(run/'detector_news_signal_audit.csv',index=False)
        detectors(panel,z,signals,cfg,run)
    benchmark_source=ROOT/cfg['cached_benchmark_run']
    benchmark_metrics=[]
    benchmark_inputs={}
    for name in ['Ensemble','Prophet']:
        path=benchmark_source/f'pred_{name}.npy'
        if not path.exists():
            raise FileNotFoundError(f'Нет сохранённого прогноза для сравнения: {path}')
        prediction=np.load(path,mmap_mode='r')[:,:,ids]
        for stage in ['validation','test']:
            benchmark_metrics.append(score_model(panel,prediction,make_grid(),stage,'Cached'+name))
        benchmark_inputs[str(path.relative_to(ROOT))]=sha(path)
    pd.concat(benchmark_metrics).to_csv(run/'cached_benchmark_metrics.csv',index=False)
    manifest={'status':'complete','elapsed_seconds':time.time()-started,'created_at':datetime.now(timezone.utc).isoformat(),
              'n_series':panel.n_series,'n_municipalities':panel.meta.territory_id.nunique(),
              'news_articles':len(news),'news_sources':news.source.value_counts().to_dict(),
              'news_months':pd.to_datetime(news.published_at,utc=True).dt.tz_convert('Europe/Moscow').dt.strftime('%Y-%m').nunique(),
              'feature_count':real.shape[2],
              'evaluation_status':cfg['evaluation_status'],'availability':cfg['availability'],
              'cached_benchmark_inputs':benchmark_inputs,
              'inputs_sha256':{str(p.relative_to(ROOT)):sha(p) for p in
                               [cfg_path,ROOT/cfg['news_file'],ROOT/cfg['corpus_manifest'],nat_path,ROOT/'data/inputs/municipal_consumption.parquet',
                                ROOT/'data/inputs/municipal_dictionary.xlsx',ROOT/'data/inputs/municipal_market_access.parquet',ROOT/'artifacts/shocks/z.npy']},
              'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'scripts/evaluate_news_impact.py',ROOT/'src/news_impact.py',ROOT/'src/news_collection.py',ROOT/'src/mun_models.py',ROOT/'src/transitions.py']}}
    (run/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (run.parent/'latest.json').write_text(json.dumps({'directory':str(run.relative_to(ROOT))})+'\n')
    print('Готово:',run.relative_to(ROOT),'секунд:',round(time.time()-started,1),flush=True)


if __name__=='__main__':
    main()
