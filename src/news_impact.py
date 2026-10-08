"""Признаки и преобразование детекторного входа для регионального пилота."""
import numpy as np
import pandas as pd
from src.news_collection import TOPICS, FACT_RULES


def news_inputs(panel, records, delay_months=0, allowed_topics=None, operational=False):
    """Возвращает признаки (T,S,K), покрытие (T,S,4) и сигнал (S,T).

    Подтверждённое региональное событие без МО используется на региональном уровне.
    При известном МО событие не распространяется на остальные МО. Задержанное
    плацебо доступно только позже оригинала и не переносит будущие новости в прошлое.
    """
    rows = records.copy().drop_duplicates('duplicate_key')
    available_col = 'operational_available_at' if operational else 'historical_available_at'
    dates = pd.to_datetime(rows[available_col], utc=True).dt.tz_convert('Europe/Moscow')
    if delay_months:
        dates = dates.map(lambda d: d + pd.DateOffset(months=delay_months))
    rows['_available'] = dates
    T, S = panel.n_months, panel.n_series
    sources = sorted(rows.source.dropna().unique()) if 'source' in rows else []
    cov = np.zeros((T, S, 4 + 2 * len(sources)))
    thematic = np.zeros((T, S, 4 * len(TOPICS)))
    facts = np.zeros((T, S, 2 * len(FACT_RULES)))
    signal = np.zeros((S, T))
    audit = []
    ids = panel.meta.territory_id.to_numpy()
    for a in range(T):
        end = (panel.months[a] + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)).tz_localize('Europe/Moscow')
        for j, months in enumerate([1, 3]):
            start = panel.months[max(0, a - months + 1)].tz_localize('Europe/Moscow')
            known = rows[(rows._available >= start) & (rows._available < end)]
            cov[a, :, j] = np.log1p(len(known))
            cov[a, :, j+2] = bool(len(known))
            for si, source in enumerate(sources):
                cov[a, :, 4 + j * len(sources) + si] = np.log1p((known.source == source).sum())
            local = known[(known.relevant == True) & (known.event_region_confirmed_in_title == True)]
            for ev in local.itertuples():
                # str(NaN) не означает известную территорию.
                text = str(ev.territory_ids)
                territories = []
                for token in text.split(';'):
                    try:
                        value = float(token)
                    except ValueError:
                        continue
                    if np.isfinite(value) and value.is_integer():
                        territories.append(int(value))
                # Не распространяем неразрешённый топоним на всю область.
                if not territories and str(getattr(ev, 'geo_status', '')) == 'ambiguous_or_unresolved':
                    continue
                hit = np.isin(ids, territories) if territories else np.ones(S, dtype=bool)
                tags = str(getattr(ev, 'topics', ev.topic)).split(';')
                tags = [t for t in tags if t in TOPICS and (allowed_topics is None or t in allowed_topics)]
                for topic in tags:
                    k = TOPICS.index(topic)
                    thematic[a, hit, j*2*len(TOPICS)+k] += 1
                    if territories:
                        thematic[a, hit, j*2*len(TOPICS)+len(TOPICS)+k] += 1
                for fi, name in enumerate(FACT_RULES):
                    value = getattr(ev, name, False)
                    economic_allowed = allowed_topics is None or bool(set(allowed_topics) - set(TOPICS[:4]))
                    if tags and economic_allowed and (value is True or str(value).lower() == 'true'):
                        facts[a, hit, j*len(FACT_RULES)+fi] += 1
                if j == 0 and tags:
                    signal[hit, a] = 1
            audit.append({'origin': a, 'window_months': months, 'known_articles': len(known),
                          'relevant_geographically_confirmed_articles': len(local),
                          'latest_available_at': known._available.max(), 'cutoff_exclusive': end,
                          'delay_months': delay_months})
    return np.concatenate([cov, np.log1p(thematic), np.log1p(facts)], axis=2), cov, signal, pd.DataFrame(audit)


def feature_names(records):
    sources = sorted(records.source.dropna().unique()) if 'source' in records else []
    names = ['log_articles_1m', 'log_articles_3m', 'has_articles_1m', 'has_articles_3m']
    names += [f'log_source_{source}_{m}m' for m in [1, 3] for source in sources]
    names += [f'log_{scope}_{topic}_{m}m' for m in [1, 3] for scope in ['relevant', 'local'] for topic in TOPICS]
    names += [f'log_{fact}_{m}m' for m in [1, 3] for fact in FACT_RULES]
    return names


def amplify_financial_input(z, signal, gain):
    """Новости изменяют чувствительность к финансовому остатку, не создают тревогу сами."""
    return z * (1.0 + gain * signal)
