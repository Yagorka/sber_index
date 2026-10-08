"""Муниципальная панель СберИндекса и национальные приоры, доступные на момент прогноза.

Таргет — средние безналичные расходы жителя МО за месяц, руб. (consumption.parquet).
Панель: 2016 МО x 6 категорий x 24 месяца (2023-01..2024-12), только полные ряды.

Национальные ряды (5 категорий, с 2018-12) используются лишь через значения не позже
якорного месяца a: сезонность и темп роста выводятся из истории, известной на a.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.forecasting import load_panel

CATEGORIES = ["Все категории", "Продовольствие", "Здоровье", "Маркетплейсы",
              "Общественное питание", "Транспорт"]
# Соответствие муниципальных категорий национальным рядам задано до просмотра ошибок.
NATIONAL_MAP = {"Все категории": "Всего",
                "Продовольствие": "Продовольственные товары",
                "Общественное питание": "Общественное питание",
                "Маркетплейсы": "Непродовольственные товары",
                "Здоровье": "Непродовольственные товары",
                "Транспорт": "Услуги"}
NATIONAL_COLUMNS = ["Всего", "Продовольственные товары", "Непродовольственные товары",
                    "Общественное питание", "Услуги"]


class MunicipalPanel:
    """values: (S, T) в рублях; meta: по строке на ряд; months: DatetimeIndex."""

    def __init__(self, values, meta, months):
        self.values = values
        self.meta = meta.reset_index(drop=True)
        self.months = months
        self.log = np.log(values)
        self.n_series, self.n_months = values.shape
        self.category_code = self.meta.category.map({c: i for i, c in enumerate(CATEGORIES)}).to_numpy()

    def month_index(self, month):
        return int(self.months.get_loc(pd.Timestamp(month)))


def load_municipal(root: Path):
    raw = pd.read_parquet(root / "data/inputs/municipal_consumption.parquet")
    raw["date"] = pd.to_datetime(raw.date + "-01")
    wide = raw.pivot_table(index=["territory_id", "category"], columns="date", values="value")
    months = wide.columns
    if not months.equals(pd.date_range(months.min(), months.max(), freq="MS")):
        raise ValueError("Календарь муниципальной панели не непрерывен")
    complete = wide.dropna()
    if (complete <= 0).any().any():
        raise ValueError("Ожидаются положительные расходы")
    meta = complete.index.to_frame(index=False)
    meta["series_id"] = meta.territory_id.astype(str) + "|" + meta.category
    attrs = load_attributes(root)
    meta = meta.merge(attrs, on="territory_id", how="left")
    meta["incomplete_dropped"] = False
    return MunicipalPanel(complete.to_numpy(dtype=float), meta, months), len(wide) - len(complete)


def load_attributes(root: Path):
    """Статические признаки МО: регион, тип, координаты центра, доступность рынков."""
    dictionary = pd.read_excel(root / "data/inputs/municipal_dictionary.xlsx")
    # Для 2024 берём актуальную версию территории (интервал [year_from, year_to)).
    active = dictionary[(dictionary.year_from <= 2024) & (dictionary.year_to > 2024)]
    active = active.sort_values("year_from").drop_duplicates("territory_id", keep="last")
    out = active[["territory_id", "region_code", "region_name", "municipal_district_type",
                  "municipal_district_name_short", "municipal_district_center_lat",
                  "municipal_district_center_lon"]].rename(
        columns={"municipal_district_type": "mo_type", "municipal_district_name_short": "mo_name",
                 "municipal_district_center_lat": "lat", "municipal_district_center_lon": "lon"})
    access_path = root / "data/inputs/municipal_market_access.parquet"
    if access_path.exists():
        out = out.merge(pd.read_parquet(access_path), on="territory_id", how="left")
    else:
        out["market_access"] = np.nan
    out["mo_type_code"] = out.mo_type.astype("category").cat.codes
    return out


class NationalPriors:
    """Национальные пути изменения, вычисленные только по данным не позже якоря."""

    def __init__(self, national: pd.DataFrame):
        self.panel = national[NATIONAL_COLUMNS]
        self.dates = self.panel.index
        self.log = np.log(self.panel.to_numpy())
        self.col = {c: i for i, c in enumerate(NATIONAL_COLUMNS)}

    @classmethod
    def from_file(cls, path):
        return cls(load_panel(path))

    def idx(self, date):
        return int(self.dates.get_loc(pd.Timestamp(date)))

    def growth(self, anchor_date):
        """log(среднее последних 12 / предыдущих 12 мес.) по 5 национальным рядам."""
        a = self.idx(anchor_date)
        return np.log(np.exp(self.log[a - 11:a + 1]).mean(0) / np.exp(self.log[a - 23:a - 11]).mean(0))

    def path(self, anchor_date, h, damping=1.0):
        """SeasonalGrowth: log(ожидаемое nat[a+h] / nat[a]); значение a+h-12 известно при h<=12."""
        a = self.idx(anchor_date)
        if not 1 <= h <= 12:
            raise ValueError("h должен быть 1..12")
        seasonal = self.log[a + h - 12] - self.log[a]
        return seasonal + damping * self.growth(anchor_date)

    def realized(self, anchor_date, k):
        """log(nat[a] / nat[a-k]) — уже наблюдённое национальное изменение."""
        a = self.idx(anchor_date)
        return self.log[a] - self.log[a - k]

    def mapped(self, vector, categories):
        """Выбирает из вектора по 5 нац. рядам компонент для каждой муниципальной категории."""
        pick = np.array([self.col[NATIONAL_MAP[c]] for c in categories])
        return np.asarray(vector)[pick]
