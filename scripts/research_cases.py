"""Официальные событийные кейсы, диагностические графики и инфляция для интерпретации."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.mun_data import load_municipal


def main():
    run = ROOT / json.loads((ROOT/"artifacts/research_runs/latest.json").read_text())["directory"]
    source = ROOT / json.loads((run/"manifest.json").read_text())["source_run"]
    panel,_ = load_municipal(ROOT)
    registry = pd.read_csv(ROOT/"data/inputs/verified_event_cases.csv")
    z = np.load(ROOT/"artifacts/shocks/z.npy")
    methods = {m: np.load(ROOT/f"artifacts/shocks/alarms_{m}.npy") for m in ["CUSUM","PageHinkley","BOCPD"]}
    methods["AbsZ"] = np.load(run/"alarms_AbsZ.npy")
    detail,match = [],[]
    for ev in registry.itertuples():
        t = panel.months.get_indexer([pd.Timestamp(ev.event_date).to_period("M").to_timestamp()])[0]
        ids = np.where(panel.meta.mo_name.fillna("").str.contains(ev.mo_name_patterns,regex=False)
                       & (panel.meta.region_name == ev.regions))[0]
        evaluable = t >= 10 and len(ids)>0
        match.append({"event_id":ev.event_id,"title":ev.title,"source_url":ev.source_url,
                      "matched_territories":panel.meta.iloc[ids].territory_id.nunique(),"event_month_index":t,
                      "evaluable":evaluable,"reason":"available" if evaluable else "no_matched_rows_or_insufficient_history",
                      "period_role":"calibration" if 10<=t<=17 else "outside_calibration",
                      "label_kind":ev.label_kind})
        if not evaluable:
            continue
        for sid in ids:
            for method,a in methods.items():
                found = np.where(a[sid,t:min(t+4,24)])[0]
                detail.append({"event_id":ev.event_id,"territory_id":panel.meta.territory_id[sid],
                    "mo_name":panel.meta.mo_name[sid],"category":panel.meta.category[sid],"method":method,
                    "detected":bool(len(found)),"delay_months":int(found[0]) if len(found) else np.nan,
                    "max_abs_z":np.nanmax(np.abs(z[sid,t:min(t+4,24)])),
                    "background_alarm_share":a[:,t:min(t+4,24)].any(1).mean(),"label_kind":ev.label_kind})
    pd.DataFrame(match).to_csv(run/"verified_event_matching.csv",index=False)
    pd.DataFrame(detail).to_csv(run/"verified_event_detection.csv",index=False)
    # Вне полного списка событий тревоги нельзя достоверно назвать ложными.
    known = np.zeros_like(z,dtype=bool)
    for row in detail:
        ev = registry[registry.event_id==row["event_id"]].iloc[0]
        t = panel.months.get_indexer([pd.Timestamp(ev.event_date).to_period("M").to_timestamp()])[0]
        m = (panel.meta.territory_id==row["territory_id"]).to_numpy()
        known[m,t:min(t+4,24)] = True
    unmatched = []
    for method,a in methods.items():
        for sid,t in zip(*np.where(a & ~known)):
            if t >= 18:
                unmatched.append({"method":method,"territory_id":panel.meta.territory_id[sid],"mo_name":panel.meta.mo_name[sid],
                    "category":panel.meta.category[sid],"month":panel.months[t],"abs_z":abs(z[sid,t]),
                    "status":"alarm_without_verified_registry_event_not_proven_false_alarm"})
    unmatched = pd.DataFrame(unmatched).sort_values(["method","abs_z"],ascending=[True,False])
    unmatched.to_csv(run/"unmatched_alarms.csv",index=False)
    # Детерминированные примеры: событие с тревогой, событие без тревоги, тревога без события.
    examples = []
    d = pd.DataFrame(detail)
    for detected in (True,False):
        rows = d[d.detected==detected].sort_values(["event_id","territory_id","category","method"])
        if len(rows):
            r=rows.iloc[0]
            ev=registry[registry.event_id==r.event_id].iloc[0]
            examples.append((r.territory_id,r.category,r.method,ev.event_date,
                             "Событие с тревогой" if detected else "Событие без тревоги"))
    if len(unmatched):
        r=unmatched.sort_values(["territory_id","category","month","method"]).iloc[0]
        examples.append((r.territory_id,r.category,r.method,r.month,"Тревога без события в реестре"))
    figure_dir = run/"figures"
    figure_dir.mkdir()
    pred=np.load(source/"pred_NatPath_K3.npy")
    series_rows=[]
    for i,(tid,cat,method,date,title) in enumerate(examples):
        sid=np.where((panel.meta.territory_id==tid)&(panel.meta.category==cat))[0][0]
        issued=np.r_[np.full(6,np.nan),[pred[t-1,1,sid] for t in range(6,24)]]
        fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
        axes[0].plot(panel.months,panel.values[sid],label="Факт")
        axes[0].plot(panel.months,issued,label="Прогноз h=1 до наблюдения")
        axes[0].axvline(pd.Timestamp(date),color="black",ls="--")
        axes[0].set_ylabel("руб./жителя"); axes[0].legend()
        axes[0].set_title(f"{title}: {panel.meta.mo_name[sid]}, {cat}")
        axes[1].plot(panel.months,z[sid],label="Стандартизованный остаток")
        hits=methods[method][sid]
        axes[1].scatter(panel.months[hits],z[sid,hits],color="red",label=method)
        axes[1].axvline(pd.Timestamp(date),color="black",ls="--")
        axes[1].legend();fig.autofmt_xdate();fig.tight_layout()
        fig.savefig(figure_dir/f"case_{i+1}.png",dpi=140);plt.close(fig)
        for t in range(24):
            series_rows.append({"case":i+1,"case_kind":title,"territory_id":tid,"category":cat,"method":method,
                "month":panel.months[t],"actual":panel.values[sid,t],"forecast_h1":issued[t],"z":z[sid,t],
                "alarm":bool(hits[t]),"event_date":date})
    pd.DataFrame(series_rows).to_csv(run/"case_series.csv",index=False)
    # Дефляция для двух сопоставимых широких групп, только декабрь к декабрю.
    cpi=pd.read_csv(ROOT/"data/inputs/cpi_december_2024.csv")
    inflation=[]
    for cat,group in [("Все категории","Все товары и услуги"),("Продовольствие","Продовольственные товары")]:
        m=(panel.meta.category==cat).to_numpy()
        nominal=panel.values[m,23]/panel.values[m,11]
        price=float(cpi[cpi.price_group==group].index_yoy_pct.iloc[0])/100
        inflation.append({"category":cat,"price_group":group,"nominal_yoy_mean_pct":100*(nominal.mean()-1),
                          "deflated_proxy_yoy_mean_pct":100*((nominal/price).mean()-1),
                          "cpi_yoy_pct":100*(price-1),"note":"национальный ИПЦ; изменение структуры и охвата платежей не устранено"})
    pd.DataFrame(inflation).to_csv(run/"inflation_interpretation.csv",index=False)
    # Эффект изменения весов агрегирования для sanity-check общепита за 2025 год.
    f=pd.read_csv(source/"forecast_2025.csv",parse_dates=["target_month"])
    nat_cfg=json.loads((ROOT/"configs/national_forecast.json").read_text())
    from src.forecasting import load_panel
    nat=load_panel(ROOT/nat_cfg["input"])
    rows=[]
    for cat,ncol in [("Общественное питание","Общественное питание"),("Все категории","Всего"),("Продовольствие","Продовольственные товары")]:
        ids=np.where(panel.meta.category==cat)[0]
        terr=panel.meta.territory_id.iloc[ids].to_numpy()
        part=f[f.category==cat].pivot(index="territory_id",columns="target_month",values="y_pred_ensemble").reindex(terr)
        for m in range(12):
            date=pd.Timestamp(2025,m+1,1)
            base=panel.values[ids,12+m];forecast=part[date].to_numpy()
            rows.append({"category":cat,"month":date,"mean_mo_yoy_pct":100*(forecast/base-1).mean(),
                         "median_mo_yoy_pct":100*np.median(forecast/base-1),
                         "ratio_of_sums_yoy_pct":100*(forecast.sum()/base.sum()-1),
                         "national_actual_yoy_pct":100*(nat.loc[date,ncol]/nat.loc[date-pd.DateOffset(years=1),ncol]-1),
                         "note":"ratio_of_sums не является взвешиванием по населению; фактов МО нет"})
    pd.DataFrame(rows).to_csv(run/"2025_aggregation_sensitivity.csv",index=False)
    print("События и интерпретация:",run)


if __name__=="__main__":
    main()
