import numpy as np
import pandas as pd
import pytest
from scripts.collect_news_archive import wordpress_record, plain_title
from src.news_collection import title_tags, ECONOMIC_RULES, CRISIS_TOPICS
from src.news_impact import news_inputs, feature_names
from tests.test_news_impact import panel, records


def test_api_date_uses_gmt_and_html_entities_are_decoded():
    post = {'id': 10, 'date': '2024-03-01T01:00:00', 'date_gmt': '2024-02-29T20:00:00',
            'title': {'rendered': 'В Орске &laquo;новый&raquo; <b>магазин</b>'}, 'link': 'https://orenburg.media/?p=10'}
    row = wordpress_record(post)
    assert row['published_at'] == '2024-02-29T23:00:00+03:00'
    assert row['title'] == 'В Орске «новый» магазин'
    with pytest.raises(ValueError):
        wordpress_record({**post, 'date_gmt': ''})
    assert plain_title('A&nbsp; B') == 'A B'


def test_multi_topic_and_direction_flags_are_not_event_labels():
    tags, facts = title_tags('В Орске открыли новый магазин и повысили зарплаты')
    assert {'retail_services', 'income_jobs'} <= set(tags)
    assert facts['business_opening'] and facts['income_increase']
    assert not facts['business_closure']
    assert 'prices' in title_tags('В Оренбурге подорожал бензин')[0]
    assert 'prices' not in title_tags('Работа завода оценена экспертами')[0]
    assert not title_tags('Как получить пособие: инструкция')[0]


def test_economic_ablation_and_unknown_geography_and_operational_mode():
    frame = records().assign(topic='prices', topics='prices;retail_services', source='СМИ',
                             operational_available_at='2026-10-07T10:00:00+00:00',
                             geo_status='matched_title_alias', price_increase=True)
    full, cov, signal, _ = news_inputs(panel(), frame)
    assert full.shape[2] == len(feature_names(frame))
    assert signal[0, 1] and not signal[1].any()
    economic, _, esignal, _ = news_inputs(panel(), frame, allowed_topics=list(ECONOMIC_RULES))
    np.testing.assert_array_equal(economic, full)
    assert esignal[0, 1]
    crisis, ccov, csignal, _ = news_inputs(panel(), frame, allowed_topics=CRISIS_TOPICS)
    np.testing.assert_array_equal(cov, ccov)
    assert not crisis[:, :, cov.shape[2]:].any() and not csignal.any()
    unknown = frame.assign(territory_ids=pd.NA, geo_status='ambiguous_or_unresolved')
    _, _, unknown_signal, _ = news_inputs(panel(), unknown)
    assert not unknown_signal.any()
    online, _, osignal, _ = news_inputs(panel(), frame, operational=True)
    assert not online.any() and not osignal.any()


def test_multisource_future_publication_does_not_change_prior_features():
    frame = records().assign(source='СМИ', topics='prices', topic='prices')
    future = frame.assign(duplicate_key='b', historical_available_at='2024-04-06T08:42:00+03:00')
    before, _, _, _ = news_inputs(panel(), frame)
    after, _, _, _ = news_inputs(panel(), pd.concat([frame, future]))
    np.testing.assert_array_equal(before[:3], after[:3])
