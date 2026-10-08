# Запуск из корня проекта в окружении `sber` (conda env create -f environment.yml).
PY ?= python

.PHONY: data test national municipal shocks weekly news nowcast check2025 figures report all

data:
	bash scripts/get_data.sh

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
