"""Проверка версий, целостности входов, временных границ и воспроизведения метрик."""
import hashlib
import importlib.metadata
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd


def main():
    checks=[]
    for line in (ROOT/"requirements.lock.txt").read_text().splitlines():
        m=re.fullmatch(r"([\w.-]+)==([^ ]+)",line.strip())
        if not m:
            continue
        name,expected=m.groups()
        try:
            actual=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual="missing"
        checks.append({"check":f"package:{name}","passed":actual==expected,"expected":expected,"actual":actual})
    run=ROOT/json.loads((ROOT/"artifacts/research_runs/latest.json").read_text())["directory"]
    manifest=json.loads((run/"manifest.json").read_text())
    migration=json.loads((ROOT/"configs/input_migration.json").read_text())
    for relative,expected in manifest["input_sha256"].items():
        path=ROOT/migration.get(relative,relative)
        actual=hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"
        checks.append({"check":f"sha256:{relative}","passed":actual==expected,"expected":expected,"actual":actual})
    selection=pd.read_csv(run/"past_only_selection_audit.csv")
    calibration=pd.read_csv(run/"calibration_audit.csv")
    merged=selection.merge(calibration,on=["origin","h"],suffixes=("_selection","_calibration"))
    lastcal=merged.last_calibration_target
    checks.append({"check":"past_only_selection","passed":bool((selection.selection_last_target <= selection.origin-2).all())})
    checks.append({"check":"mature_calibration","passed":bool((lastcal<=merged.origin).all())})
    disjoint=[]
    for r in merged.itertuples():
        targets=[] if pd.isna(r.calibration_targets) else [int(x) for x in str(r.calibration_targets).split(";")]
        disjoint.append(all(t>r.selection_last_target for t in targets))
    checks.append({"check":"selection_calibration_disjoint","passed":all(disjoint)})
    market=pd.read_csv(run/"marketplace_availability.csv")
    dates=pd.to_datetime(market.last_available_at)
    origin_end=pd.Timestamp("2023-01-01")+market.origin.map(lambda x:pd.DateOffset(months=int(x)))
    origin_end=pd.to_datetime(origin_end)+pd.offsets.MonthEnd(0)
    checks.append({"check":"weekly_published_before_origin","passed":bool((dates.isna()|(dates<=origin_end)).all())})
    reproduction=pd.read_csv(run/"reproduction.csv")
    checks.append({"check":"original_metric_reproduction","passed":bool(reproduction.mae_abs_difference.max()<1e-5),
                   "max_difference":float(reproduction.mae_abs_difference.max())})
    for name in ["forecast_2025_frozen","national_2026-09_frozen"]:
        path=ROOT/f"prospective/{name}.csv"
        expected=(ROOT/f"prospective/{name}.sha256").read_text().split()[0]
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        checks.append({"check":f"frozen:{name}","passed":actual==expected})
    passed=all(c["passed"] for c in checks)
    result={"checked_at_utc":datetime.now(timezone.utc).isoformat(),"python":sys.version,
            "executable":sys.executable,"run":str(run.relative_to(ROOT)),"passed":passed,"checks":checks,
            "limitations":"Проверка текущей среды; установка зависимостей с нуля и внешняя временная отметка не подтверждены."}
    (ROOT/"tracking/research_verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(f"Verification: {sum(c['passed'] for c in checks)}/{len(checks)} checks passed")
    for c in checks:
        if not c["passed"]:
            print(c)
    if not passed:
        sys.exit(1)


if __name__=="__main__":
    main()
