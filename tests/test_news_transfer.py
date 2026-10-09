import numpy as np
import pandas as pd
import pytest

from scripts.collect_transfer_news import parse_niann
from scripts.evaluate_news_decay import apply_choices, choose_representations, add_families


def test_niann_archive_date_canonical_url_and_cap_detection():
    html='''<div id="block_118"><div class="search_full__found">Найдено: 1</div>
    <div class="cell_news_item"><div class="cell_news_alltext"><div class="date-wrap">
    <span class="date_standart">31 января 2023 18:56</span></div>
    <a href="/?id=588781&amp;query_id=42">В Нижегородской области повысили зарплаты</a></div></div>
    <a href="/?id=118&amp;page118=2&amp;query_id=42">2</a></div>'''
    rows,count,capped,pages=parse_niann(html)
    assert count==1 and not capped
    assert rows[0]['published_at']=='2023-01-31T18:56:00+03:00'
    assert rows[0]['url']=='https://www.niann.ru/?id=588781'
    assert pages[2].endswith('query_id=42')
    assert parse_niann(html.replace('Найдено: 1','Найдено: более 500'))[2]
    undated = html.replace('<span class="date_standart">31 января 2023 18:56</span>', '')
    assert parse_niann(undated)[0][0]['published_at'] is None
    with pytest.raises(ValueError):
        parse_niann('<html>Access unavailable</html>')


def test_transfer_applies_frozen_settings_without_reading_targets():
    candidates={('economic_windows',100,0.):np.ones((24,13,2)),
                ('economic_windows',100,.1):np.full((24,13,2),2.)}
    frozen=pd.DataFrame([{'variant':'economic_windows','alpha':100,'shrink':0.,'h':1},
                         {'variant':'economic_windows','alpha':100,'shrink':.1,'h':3}])
    snapshot=frozen.copy(deep=True)
    selected=apply_choices(candidates,frozen)
    assert (selected['economic_windows'][:,1]==1).all()
    assert (selected['economic_windows'][:,3]==2).all()
    pd.testing.assert_frame_equal(frozen,snapshot)
    representations=frozen.assign(family='selected_news')
    add_families(selected,representations)
    np.testing.assert_array_equal(selected['selected_news'][:,3],selected['economic_windows'][:,3])


def test_representation_ties_prefer_zero_and_windows_without_test_values():
    chosen=pd.DataFrame([{'variant':v,'alpha':100,'shrink':s,'h':h,'validation_mae':1.}
        for v in ['economic_windows','economic_decay_14d','economic_decay_30d','economic_decay_90d']
        for s in [0.,.1] for h in [1,3,6,12]])
    picked=choose_representations(chosen)
    assert picked.shrink.eq(0).all()
    assert picked[picked.family.eq('selected_news')].variant.eq('economic_windows').all()
