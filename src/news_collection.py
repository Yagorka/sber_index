"""Лёгкая обработка метаданных региональных новостей без моделей и HTTP."""
import hashlib
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

MONTHS_RU = ['января', 'февраля', 'марта', 'апреля', 'мая', 'июня',
             'июля', 'августа', 'сентября', 'октября', 'ноября', 'декабря']

CRISIS_TOPICS = ['flood_or_evacuation', 'infrastructure', 'fire', 'weather']
ECONOMIC_RULES = {
    'prices': r'\bцен(?:а|ы|у|е|ам|ах|ой)\b|ценов|подорож|подешев|инфляц|тариф',
    'income_jobs': r'зарплат|заработн|доход|безработ|ваканси|рынок труда|пенси',
    'payments_support': r'выплат|пособи|компенсац|субсиди|маткапитал|материнск.*капитал',
    'retail_services': r'магазин|торгов|ритейл|розниц|маркетплейс|wildberries|ozon|ресторан|кафе|общепит',
    'business_industry': r'предприяти|завод|производств|промышлен|инвестиц|бизнес|предпринимател',
    'transport_housing': r'\bдорог(?:а|и|у|е|ам|ах|ой)\b|дорожн|транспорт|автобус|маршрут|перевоз|жиль|ипотек|строитель|новостро',
}
TOPICS = CRISIS_TOPICS + list(ECONOMIC_RULES)
FACT_RULES = {
    'price_increase': r'(?:подорож|рост.*цен|цен.*(?:вырос|растут|повыс))',
    'price_decrease': r'(?:подешев|снижен.*цен|цен.*(?:сниз|упал))',
    'income_increase': r'(?:рост|увелич|повыш|повыс|вырос).*(?:зарплат|пенси|доход)|(?:зарплат|пенси|доход).*(?:рост|увелич|повыш|повыс|вырос)',
    'business_opening': r'откры(?:л|т|ва)|запуск|запуст|введ.*эксплуатац',
    'business_closure': r'закры(?:л|т|ва)|банкрот|сокращен|приостан',
}


def title_tags(title):
    """Несколько тем одного заголовка; без LLM, будущих текстов и обучения на тесте."""
    topic, relevant = classify_title(title)
    if topic == 'routine_or_advice':
        return [], {name: False for name in FACT_RULES}
    t = title.lower().replace('ё', 'е')
    topics = [topic] if relevant else []
    topics += [name for name, pattern in ECONOMIC_RULES.items() if re.search(pattern, t)]
    facts = {name: bool(re.search(pattern, t)) for name, pattern in FACT_RULES.items()}
    # «Открыли дорогу» не является открытием предприятия.
    for name in ['business_opening', 'business_closure']:
        facts[name] &= bool(set(topics) & {'business_industry', 'retail_services'})
    return topics, facts


def parse_publication(text):
    match = re.search(r'(\d{1,2}) (' + '|'.join(MONTHS_RU) + r') (\d{4}), (\d{2}):(\d{2})', text)
    if not match:
        raise ValueError(f'Не найдена точная дата публикации: {text}')
    day, month, year, hour, minute = match.groups()
    return datetime(int(year), MONTHS_RU.index(month) + 1, int(day),
                    int(hour), int(minute), tzinfo=ZoneInfo('Europe/Moscow'))


def classify_title(title):
    """Тема заголовка — диагностический признак, не разметка сдвига расходов."""
    t = title.lower()
    if re.search(r'погибли.*с начала|итоги работы', t):
        return 'summary', False
    if re.search(r'поздрав|торжеств|учени[яй]|трениров|аттест|урок|курсант|абитуриент|погоны|профилакти|инструктаж|как |важно знать|безопасност|соблюдайте|не забывай|будь внимател', t):
        return 'routine_or_advice', False
    if re.search(r'павод|подтоп|половод|размыв.*дамб|эваку', t):
        return 'flood_or_evacuation', True
    if re.search(r'отключен|электроэнерг|электроснабж|авари', t):
        return 'infrastructure', True
    if re.search(r'пожар|горени|огнем|огнём', t):
        return 'fire', True
    if re.search(r'шторм|снегопад|метел|ураган|сильн.*ветр', t):
        return 'weather', True
    return 'other', False


def match_territories(title, published, dictionary, config):
    """Привязка только по явным алиасам и версии справочника в год публикации."""
    active = dictionary[(dictionary.region_code == config['region_code']) &
                        (dictionary.year_from <= published.year) &
                        (dictionary.year_to > published.year)]
    matched, unresolved = [], []
    for name, pattern in config['territory_aliases'].items():
        if not re.search(pattern, title, re.IGNORECASE):
            continue
        ids = active.loc[active.municipal_district_name_short == name, 'territory_id'].unique()
        if len(ids) == 1:
            matched.append(int(ids[0]))
        else:
            unresolved.append(name)
    return sorted(set(matched)), unresolved


def metadata_record(raw, dictionary, config, fetched_at):
    if raw.get('published_at'):
        published = pd.Timestamp(raw['published_at'])
        if published.tzinfo is None:
            raise ValueError('Дата публикации должна содержать часовой пояс')
        published = published.to_pydatetime()
    else:
        published = parse_publication(raw['published_text'])
    ids, unresolved = match_territories(raw['title'], published, dictionary, config)
    topic, relevant = classify_title(raw['title'])
    topics, facts = title_tags(raw['title'])
    if topic == 'other' and topics:
        topic = topics[0]
    relevant = bool(topics)
    normalized = re.sub(r'\W+', ' ', raw['title'].lower().replace('ё', 'е')).strip()
    duplicate_key = hashlib.sha256((published.date().isoformat() + '|' + normalized).encode()).hexdigest()
    return {'article_id': raw['url'].rstrip('/').split('/')[-1], 'url': raw['url'],
            'source': config['source_name'], 'title': raw['title'],
            'published_at': published.isoformat(), 'event_date': '',
            'event_date_status': 'not_extracted', 'first_seen_at': fetched_at,
            'fetched_at': fetched_at, 'historical_available_at': published.isoformat(),
            'operational_available_at': max(pd.Timestamp(published), pd.Timestamp(fetched_at)).isoformat(),
            'availability_status': 'historical_publication_assumption_without_vintages',
            'region_code': config['region_code'], 'region_name': config['region_name'],
            'event_region_confirmed_in_title': bool(ids or re.search(
                config.get('region_title_pattern', r'Оренбур' if config['region_code'] == 56 else re.escape(config['region_name'])),
                raw['title'], re.IGNORECASE)),
            'geo_time_resolution': 'annual_dictionary_version',
            'territory_ids': ';'.join(map(str, ids)),
            'geo_status': 'ambiguous_or_unresolved' if unresolved else ('matched_title_alias' if ids else 'region_source_only'),
            'unresolved_places': ';'.join(unresolved), 'topic': topic, 'topics': ';'.join(topics),
            'relevant': relevant, **facts,
            'duplicate_key': duplicate_key, 'metadata_source': raw.get('metadata_source', 'web_search_source_excerpt'),
            'coverage': config['coverage'], 'label_kind': 'news_metadata_not_verified_spending_shift'}


def monthly_counts(records, start, end):
    """Количество найденных сообщений; пустой месяц неизвестен, а не ноль событий."""
    frame = records.copy()
    frame['month'] = pd.to_datetime(frame.published_at, utc=True).dt.tz_convert('Europe/Moscow').dt.strftime('%Y-%m')
    rows = []
    for month in pd.period_range(start, end, freq='M').astype(str):
        current = frame[frame.month == month]
        row = {'month': month, 'retrieved_articles': len(current),
               'coverage_complete': False, 'has_retrieved_articles': bool(len(current))}
        for topic in ['flood_or_evacuation', 'infrastructure', 'fire', 'weather']:
            row['retrieved_' + topic] = int((current.topic == topic).sum()) if len(current) else None
        rows.append(row)
    return pd.DataFrame(rows)


def features_at(records, cutoff, operational=False):
    """Региональные счётчики за 1 и 3 месяца, без будущих публикаций.

    operational=True требует фактического first_seen_at. Для пилота 2023–2024
    допустим лишь явно обозначенный ретроспективный режим по published_at.
    """
    end = pd.Timestamp(cutoff)
    if end.tzinfo is None:
        raise ValueError('cutoff должен содержать часовой пояс')
    available_col = 'operational_available_at' if operational else 'historical_available_at'
    available = pd.to_datetime(records[available_col], utc=True)
    published = pd.to_datetime(records.published_at, utc=True)
    result = {}
    for months in [1, 3]:
        mask = (available <= end) & (published > end - pd.DateOffset(months=months))
        current = records[mask].drop_duplicates('duplicate_key')
        result[f'retrieved_articles_{months}m'] = len(current)
        current = current[current.event_region_confirmed_in_title == True]
        for topic in ['flood_or_evacuation', 'infrastructure', 'fire', 'weather']:
            result[f'retrieved_{topic}_{months}m'] = int((current.topic == topic).sum())
    result['coverage_complete'] = False
    return result
