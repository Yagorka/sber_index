import pandas as pd
from src.news_collection import metadata_record, features_at, monthly_counts, match_territories, classify_title, parse_publication


def settings():
    return {'region_code': 56, 'region_name': 'Оренбургская область',
            'source_name': 'МЧС', 'coverage': 'incomplete',
            'territory_aliases': {'Орск': r'\bОрск(?:а|е|ом)?\b'}}


def dictionary():
    return pd.DataFrame({'region_code': [56, 56], 'year_from': [2018, 2024],
                         'year_to': [2024, 9999], 'municipal_district_name_short': ['Орск', 'Орск'],
                         'territory_id': [1, 2]})


def record(day, title='В Орске началась эвакуация'):
    return metadata_record({'title': title, 'published_text': day,
                            'url': 'https://56.mchs.gov.ru/novosti/1'},
                           dictionary(), settings(), '2026-10-07T10:00:00+00:00')


def test_future_news_and_collection_time():
    earlier = record('6 апреля 2024, 08:42')
    future = record('20 апреля 2024, 08:42', 'В Орске продолжается эвакуация')
    cutoff = '2024-04-10T23:59:59+03:00'
    assert features_at(pd.DataFrame([earlier]), cutoff) == features_at(pd.DataFrame([earlier, future]), cutoff)
    assert features_at(pd.DataFrame([earlier]), cutoff)['retrieved_flood_or_evacuation_1m'] == 1
    assert features_at(pd.DataFrame([earlier]), cutoff, operational=True)['retrieved_articles_1m'] == 0


def test_geography_uses_publication_year_and_not_oblast_name():
    ids, _ = match_territories('Эвакуация в Орске', parse_publication('6 апреля 2023, 08:42'), dictionary(), settings())
    assert ids == [1]
    assert record('6 апреля 2024, 08:42')['territory_ids'] == '2'
    assert record('6 апреля 2024, 08:42', 'Паводок в пострадавших регионах')['event_region_confirmed_in_title'] is False


def test_empty_month_is_unknown_and_advice_is_not_event():
    frame = pd.DataFrame([record('6 апреля 2024, 08:42')])
    counts = monthly_counts(frame, '2024-03-01', '2024-04-30')
    assert pd.isna(counts.iloc[0].retrieved_flood_or_evacuation)
    assert not counts.coverage_complete.any()
    assert classify_title('Как уберечься от молнии во время грозы')[1] is False
    assert classify_title('24 человека погибли на пожарах с начала 2024 года')[0] == 'summary'
