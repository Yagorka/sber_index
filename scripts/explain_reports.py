"""Apply reader-facing terminology to existing reports without recalculating metrics."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.report_language import explain_text


def main():
    paths=[ROOT/'README.md']+list((ROOT/'docs').glob('*.md'))
    paths += [ROOT/p for p in [
        'artifacts/llm_news_runs/llm_20261008T151443_476526Z/results_report.md',
        'artifacts/news_decay_runs/decay_20261008T153750_772322Z/results_report.md',
        'artifacts/local_news_runs/local_20261009T124348_521680Z/results_report.md',
        'artifacts/news_mass_runs/mass_20261009T155140_551755Z/results_report.md']]
    changed=0
    for path in paths:
        if path.name=='TERMS_RU.md':continue
        old=path.read_text();new=explain_text(old)
        if new!=old:path.write_text(new);changed+=1
    print('Reports clarified:',changed)

if __name__=='__main__':main()
