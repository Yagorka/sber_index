"""Признаки «новостей» как датированных официальных событий (не текстовый корпус).

Источники: решения Банка России по ключевой ставке (cbr.ru) и реестр региональных событий
с проверяемыми ссылками (data/external/event_registry.csv). Все признаки в якоре a
используют только события с available_at <= конец месяца a. Плацебо: те же события
переносятся на случайные МО и случайные месяцы.
"""

import numpy as np
import pandas as pd

EXTRA_NAMES = ["kr_level", "kr_d3", "kr_d6", "kr_months_since", "ev_mo_recent", "ev_region_recent"]


class NewsFeatures:
    def __init__(self, panel, key_rate_daily, registry, placebo_seed=None):
        self.panel = panel
        self.rate = key_rate_daily.set_index("date").rate_pct.sort_index()
        meta = panel.meta
        S = panel.n_series
        T = panel.n_months
        self.mo_event = np.zeros((S, T))       # severity в месяце события
        self.region_event = np.zeros((S, T))
        rng = np.random.default_rng(placebo_seed) if placebo_seed is not None else None
        regions = meta.region_name.fillna("").to_numpy()
        for ev in registry.itertuples():
            t = int(panel.months.get_indexer([pd.Timestamp(ev.event_date).to_period("M").to_timestamp()])[0])
            if t < 0:
                continue
            region_hit = np.array([any(rg in r for rg in ev.regions.split(";")) for r in regions])
            mo_hit = np.zeros(S, dtype=bool)
            for pat in [x for x in ev.mo_name_patterns.split(";") if x]:
                mo_hit |= (meta.mo_name.fillna("").str.contains(pat).to_numpy() & region_hit)
            if rng is not None:
                # плацебо: тот же размер охвата, случайные территории и месяц
                terr = meta.territory_id.unique()
                n_t = max(1, meta[mo_hit].territory_id.nunique())
                chosen = rng.choice(terr, n_t, replace=False)
                mo_hit = meta.territory_id.isin(chosen).to_numpy()
                n_r = max(1, int(region_hit.sum() / 6 // 1))
                region_hit = meta.region_code.isin(rng.choice(meta.region_code.unique(),
                                                              max(1, meta[region_hit].region_code.nunique()),
                                                              replace=False)).to_numpy()
                t = int(rng.integers(3, T))
            self.mo_event[mo_hit, t] = max(self.mo_event[mo_hit, t].max() if mo_hit.any() else 0, ev.severity)
            self.region_event[region_hit, t] = np.maximum(self.region_event[region_hit, t], ev.severity)

    def __call__(self, a):
        end = self.panel.months[a] + pd.offsets.MonthEnd(0)
        known = self.rate[self.rate.index <= end]
        level = float(known.iloc[-1])
        def at(offset):
            ref = end - pd.DateOffset(months=offset)
            prev = self.rate[self.rate.index <= ref]
            return float(prev.iloc[-1]) if len(prev) else level
        changes = known[known.diff().fillna(0) != 0]
        since = (end.year - changes.index[-1].year) * 12 + end.month - changes.index[-1].month if len(changes) else 24
        S = self.panel.n_series
        window = slice(max(0, a - 2), a + 1)
        cols = [np.full(S, level), np.full(S, level - at(3)), np.full(S, level - at(6)),
                np.full(S, float(min(since, 24))),
                self.mo_event[:, window].max(1), self.region_event[:, window].max(1)]
        return np.column_stack(cols)
