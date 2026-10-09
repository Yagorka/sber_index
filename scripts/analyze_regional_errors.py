"""Describe saved forecast errors without changing model selection or predictions."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.mun_data import load_municipal,MunicipalPanel
from src.mun_eval import make_grid,pair_errors
from scripts.evaluate_news_decay import hash_file

VARIANTS=['Исходная модель','Поправка без новостей','Прежние доли новостей','Новости с уменьшением влияния со временем','Только количество новостей']


def summarize(frame,keys):
    rows=[]
    for key,g in frame.groupby(keys,sort=False):
        if not isinstance(key,tuple):key=(key,)
        row=dict(zip(keys,key));row.update(n=len(g),mae=g.abs_error.mean(),median_error=g.abs_error.median(),
            bias=g.error.mean(),wape=g.abs_error.sum()/g.actual.sum(),underprediction_share=g.error.lt(0).mean(),
            improvement_vs_original=(g.original_abs_error-g.abs_error).mean(),total_abs_error=g.abs_error.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    cfg=json.loads((ROOT/'configs/news_mass.json').read_text())
    source=ROOT/json.loads((ROOT/'artifacts/news_mass_runs/latest.json').read_text())['directory']
    out=ROOT/'artifacts/regional_error_analysis';out.mkdir(exist_ok=True)
    full,_=load_municipal(ROOT);grid=make_grid();allmodels=[];ensemble=[];hashes={}
    for region in cfg['regions']:
        meta=pd.read_csv(source/region/'series_index.csv')
        ids=full.meta.set_index('series_id').index.get_indexer(meta.series_id);assert (ids>=0).all()
        panel=MunicipalPanel(full.values[ids],meta,full.months)
        for model in cfg['base_models']:
            path=source/region/f'pred_{model}.npy';hashes[str(path.relative_to(ROOT))]=hash_file(path)
            forecasts=np.load(path,mmap_mode='r');g,y,p,_=pair_errors(panel,forecasts[0],grid,'test')
            for h in [1,3,6,12]:
                mask=g.h.eq(h).to_numpy();err=p[mask]-y[mask]
                allmodels.append(dict(region=panel.meta.region_name.iloc[0],model=model,h=h,mae=float(np.abs(err).mean()),
                    bias=float(err.mean()),wape=float(np.abs(err).sum()/y[mask].sum())))
            if model!='Ensemble':continue
            for v,name in enumerate(VARIANTS):
                _,_,vp,_=pair_errors(panel,forecasts[v],grid,'test')
                for j,row in g.iterrows():
                    previous=panel.values[:,row.target-1]
                    rel=(y[j]/previous-1)
                    frame=meta[['territory_id','mo_name','category']].copy()
                    frame['region']=panel.meta.region_name.iloc[0];frame['variant']=name;frame['h']=row.h
                    frame['month']=panel.months[row.target].strftime('%Y-%m');frame['actual']=y[j]
                    frame['forecast']=vp[j];frame['error']=vp[j]-y[j];frame['abs_error']=np.abs(frame.error)
                    frame['original_abs_error']=np.abs(p[j]-y[j]);frame['monthly_change']=rel
                    frame['change_group']=np.where(np.abs(rel)>=.2,'Изменение за месяц ≥20%','Изменение за месяц <20%')
                    ensemble.append(frame)
    models=pd.DataFrame(allmodels);models.to_csv(out/'models_by_horizon.csv',index=False)
    frame=pd.concat(ensemble,ignore_index=True)
    tables={
        'region_horizon':summarize(frame,['region','variant','h']),
        'category':summarize(frame,['region','variant','h','category']),
        'month':summarize(frame,['region','variant','h','month']),
        'municipality':summarize(frame,['region','variant','h','territory_id','mo_name']),
        'large_changes':summarize(frame,['region','variant','h','change_group'])}
    checked=0
    for region in cfg['regions']:
        original=pd.read_csv(source/region/'metrics.csv').query('base_model=="Ensemble"')
        region_name=pd.read_csv(source/region/'series_index.csv').region_name.iloc[0]
        for i,variant in enumerate(['original','financial_only','selected_fractions','selected_mass','matched_coverage']):
            actual= tables['region_horizon'][(tables['region_horizon'].region==region_name)&(tables['region_horizon'].variant==VARIANTS[i])]
            comparison=actual.merge(original[original.variant.eq(variant)],on='h',suffixes=('_analysis','_saved'))
            assert len(comparison)==4
            np.testing.assert_allclose(comparison.mae_analysis,comparison.mae_saved,rtol=1e-12)
            checked+=len(comparison)
    (out/'verification.json').write_text(json.dumps({'passed':True,'saved_ensemble_metrics_reproduced':checked},indent=2)+'\n')
    for name,t in tables.items():t.to_csv(out/(name+'.csv'),index=False)
    # Detailed observations stay local, outside version control.
    local=ROOT/'data/inputs/regional_error_analysis';local.mkdir(exist_ok=True)
    frame.to_csv(local/'ensemble_errors.csv',index=False)
    chunks=[];conclusions=[]
    for region in frame.region.unique():
        base=frame[(frame.region==region)&frame.variant.eq(VARIANTS[0])&frame.h.isin([1,3])]
        cats=summarize(base,['category']).sort_values('mae',ascending=False)
        cats['error_share_pct']=100*cats.total_abs_error/cats.total_abs_error.sum()
        months=summarize(base,['month']).sort_values('mae',ascending=False)
        mos=summarize(base,['territory_id','mo_name']).sort_values('mae',ascending=False)
        topn=max(1,int(np.ceil(len(mos)*.1)));topshare=100*mos.head(topn).total_abs_error.sum()/mos.total_abs_error.sum()
        base_mae=base.abs_error.mean()
        # Reaggregate raw errors to weight both horizons equally and observations equally.
        changes=summarize(frame[(frame.region==region)&frame.h.isin([1,3])],['variant','change_group'])
        overall=summarize(frame[(frame.region==region)&frame.h.isin([1,3])],['variant'])
        worst=cats.iloc[0];worstmonth=months.iloc[0]
        conclusions.append(f"{region}: наибольшая ошибка в категории «{worst.category}» ({worst.mae:.2f} руб./жителя); её доля суммарной абсолютной ошибки — {worst.error_share_pct:.1f}%. Худший целевой месяц — {worstmonth.month} ({worstmonth.mae:.2f}). {topn} из {len(mos)} муниципалитетов дают {topshare:.1f}% ошибки.")
        def show(t,cols):
            names={'category':'Категория','mae':'MAE, руб.','wape':'Ошибка / сумма расходов','bias':'Среднее завышение (+) / занижение (−), руб.',
                'error_share_pct':'Доля всей ошибки, %','month':'Месяц','mo_name':'Муниципалитет','change_group':'Изменение фактических расходов',
                'n':'Число прогнозов','variant':'Вариант','improvement_vs_original':'Снижение ошибки против исходной, руб.'}
            return t[cols].rename(columns=names).to_markdown(index=False,floatfmt='.2f')
        chunks.append(f"## {region}\n\nОсновные горизонты — 1 и 3 месяца. Ошибка исходной модели {base_mae:.2f} руб./жителя.\n\nКатегории:\n\n{show(cats,['category','mae','wape','bias','error_share_pct'])}\n\nМесяцы:\n\n{show(months,['month','mae','bias'])}\n\nПять муниципалитетов с наибольшей ошибкой:\n\n{show(mos.head(5),['mo_name','mae','wape','bias'])}\n\nВсе варианты:\n\n{show(overall,['variant','mae','bias','improvement_vs_original'])}\n\nПри резких изменениях и в остальные месяцы:\n\n{show(changes,['variant','change_group','n','mae','improvement_vs_original'])}\n")
    base=frame[frame.variant.eq(VARIANTS[0])&frame.h.isin([1,3])]
    cat=summarize(base,['region','category']).pivot(index='category',columns='region',values='bias')
    monthly=summarize(base,['region','month']).pivot(index='month',columns='region',values='mae')
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    maxbias=float(np.abs(cat.to_numpy()).max())
    for ax,t,title,cmap,limits in [(axes[0],cat,'Завышение (+) и занижение (−) по категориям','RdBu_r',(-maxbias,maxbias)),
        (axes[1],monthly,'Средняя ошибка по целевым месяцам','YlOrRd',(0,float(monthly.to_numpy().max())))]:
        im=ax.imshow(t.to_numpy(),cmap=cmap,vmin=limits[0],vmax=limits[1],aspect='auto')
        ax.set_xticks(range(len(t.columns)),[x.replace(' область','') for x in t.columns],rotation=20,ha='right',fontsize=9)
        ax.set_yticks(range(len(t.index)),t.index,fontsize=9);ax.set_title(title,fontsize=11)
        for i in range(len(t)):
            for j in range(len(t.columns)):ax.text(j,i,f'{t.iloc[i,j]:.0f}',ha='center',va='center',fontsize=9)
        fig.colorbar(im,ax=ax,label='Руб. на жителя')
    fig.suptitle('Ensemble, прогноз на 1 и 3 месяца, июль–декабрь 2024',fontsize=12)
    fig.tight_layout();(out/'figures').mkdir(exist_ok=True);fig.savefig(out/'figures/errors.png',dpi=150);plt.close(fig)
    text='''# Где ошибается прогноз и помогают ли новости

Анализ сохранённых прогнозов восьми моделей в трёх областях. Модели, параметры, ответы LLM и исходные данные не изменялись. Ни одна часть этого разбора не использовалась для выбора модели.

MAE — средняя абсолютная ошибка: на сколько рублей в среднем прогноз отличается от факта. Средняя знаковая ошибка показывает систематическое завышение или занижение. WAPE — сумма абсолютных ошибок, делённая на сумму фактических расходов; 0,10 означает 10%. Высокая ошибка в рублях может быть следствием высокого уровня расходов, поэтому приведены оба показателя.

«Все категории» — отдельный ряд общего объёма расходов, а не ещё одна товарная категория. Как и в прежних экспериментах, каждый из шести рядов имеет одинаковый вес. «Доля всей ошибки» описывает эту схему оценки, а не долю категории в потреблении.

## Что видно по основным горизонтам

'''+ '\n\n'.join(conclusions)+'''

Расходы на продукты обычно завышены, а на маркетплейсах — занижены. Для маркетплейсов национальным ориентиром служат все непродовольственные товары, а не отдельный ряд маркетплейсов; это возможный источник расхождения, который требует отдельной проверки. По относительной ошибке общепит проблемен, хотя его ошибка в рублях мала.

При резких изменениях в Оренбургской области новости уменьшают ошибку на 5,20 руб., но вариант только с количеством новостей — на 5,70 руб. В остальные месяцы новостная поправка ухудшает результат. В Нижегородской на основных горизонтах поправка не применяется; в Костромской новости не помогают и в группе резких изменений. Это описательная проверка, без отдельного подтверждения статистической значимости.

![Где модель ошибается](../artifacts/regional_error_analysis/figures/errors.png)

'''+ '\n'.join(chunks)+'''
## Как читать проверку резких изменений

Порог 20% задан для описания фактических изменений расходов относительно предыдущего месяца, а не для настройки модели. Эти сведения становятся известны после получения факта. Проверка показывает, где прогноз не справился, но не доказывает, что изменение вызвано новостью. Высокая относительная ошибка на малых уровнях расходов тоже возможна. Во всех таблицах «снижение ошибки» больше нуля означает улучшение; меньше нуля — ухудшение.

Тестовый период — июль–декабрь 2024, всего шесть месяцев. Результаты уже просмотрены, а исходный Ensemble сохраняет прежние ограничения подбора весов. Разбор помогает выбрать следующие гипотезы, но не превращает повторную оценку в независимый тест.

Подробные ошибки остаются в data/inputs/regional_error_analysis/ensemble_errors.csv и исключены из Git; сводные таблицы и хеши — artifacts/regional_error_analysis. Воспроизведение: `python scripts/analyze_regional_errors.py`.
'''
    (ROOT/'docs/REGIONAL_ERROR_ANALYSIS_RU.md').write_text(text)
    (out/'manifest.json').write_text(json.dumps({'source_run':str(source.relative_to(ROOT)),'input_sha256':hashes,
        'code_sha256':hash_file(Path(__file__)),'change_threshold':.2,'selection_changed':False},ensure_ascii=False,indent=2)+'\n')
    print('\n'.join(conclusions))

if __name__=='__main__':main()
