# Запуск из корня проекта в окружении `sber` (conda env create -f environment.yml).
PY ?= python

.PHONY: data test national municipal shocks weekly news nowcast check2025 figures report all research research-prophet research-report verify

data:
	$(PY) scripts/check_inputs.py --profile full

test:
	$(PY) -m pytest tests -q

# Национальный эксперимент N01 (8 семейств + ансамбль) — одна среда, без отдельного statsmodels-процесса
national:
	$(PY) scripts/train_forecasts.py

# Муниципальный блок: Prophet и фундаментальные модели считаются отдельно (долго на CPU)
foundation:
	$(PY) scripts/mun_prophet.py
	$(PY) scripts/mun_foundation.py --model bolt
	$(PY) scripts/mun_foundation.py --model chronos2
	$(PY) scripts/mun_foundation.py --model chronos2_nat
	$(PY) scripts/mun_foundation.py --model timesfm --sample-mo 200

municipal:
	$(PY) scripts/run_municipal.py
	$(PY) scripts/mun_subset_eval.py

shocks:
	$(PY) scripts/run_shocks.py
	$(PY) scripts/run_weekly_shocks.py

news:
	$(PY) scripts/run_news_ablation.py

nowcast:
	$(PY) scripts/nowcast_weekly.py

check2025:
	$(PY) scripts/check_2025_vs_national.py

figures:
	$(PY) scripts/make_figures.py

report: figures
	$(PY) scripts/build_report.py
	$(PY) scripts/build_models_metrics.py

all: data test national foundation municipal shocks news nowcast check2025 report

# Дополнительные эксперименты после просмотра test, исходный запуск сохраняется.
research-prophet:
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 15 $(PY) scripts/prophet_research.py --workers 1

research:
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 15 $(PY) scripts/run_research_audit.py
	$(PY) scripts/research_cases.py
	$(PY) scripts/build_research_report.py

research-report:
	$(PY) scripts/build_research_report.py

verify:
	$(PY) scripts/verify_research.py

.PHONY: regional-news regional-news-archive regional-news-impact
regional-news:
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 15 $(PY) scripts/collect_regional_news.py

regional-news-archive:
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 15 $(PY) scripts/collect_news_archive.py

regional-news-impact:
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 15 $(PY) scripts/evaluate_news_impact.py
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 15 $(PY) scripts/build_news_impact_report.py

.PHONY: llm-news llm-news-evaluate
llm-news:
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $(PY) scripts/annotate_llm_news.py

llm-news-evaluate:
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $(PY) scripts/evaluate_llm_news.py

.PHONY: news-transfer-collect news-transfer-annotate news-decay
news-transfer-collect:
	$(PY) scripts/collect_transfer_news.py --config configs/news_archive_nizhny.json
	$(PY) scripts/collect_news_archive.py --config configs/news_archive_kostroma.json

news-transfer-annotate:
	$(PY) scripts/annotate_llm_news.py --config configs/llm_news_nizhny.json
	$(PY) scripts/annotate_llm_news.py --config configs/llm_news_kostroma.json

news-decay:
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $(PY) scripts/evaluate_news_decay.py

.PHONY: local-news
local-news:
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $(PY) scripts/evaluate_local_news.py

.PHONY: news-mass news-label-audit
news-mass:
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $(PY) scripts/evaluate_news_mass.py

news-label-audit:
	$(PY) scripts/audit_news_labels.py
