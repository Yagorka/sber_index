"""Значимые проверки временных границ и доступности новых экспериментов."""
import numpy as np
import pandas as pd
from src.research_audit import rolling_blend, past_calibration, shock_adjustment, select_past_weights
from src.mun_news import NewsFeatures
from tests.test_municipal import make_world
from src.frozen_artifacts import preserve_or_create


def world():
    values=np.tile(np.arange(24,dtype=float)+100,(3,1))
    a=np.full((24,13,3),101.,dtype=float)
    b=np.full_like(a,110.)
    return values,{"a":a,"b":b}


def test_future_targets_and_calibration_targets_cannot_change_selection():
    y,p=world()
    w,pairs,_=select_past_weights(y,p,16,1)
    changed=y.copy();changed[:,15:]*=100
    other,other_pairs,_=select_past_weights(changed,p,16,1)
    assert w==other and pairs==other_pairs
    assert max(t for _,t in pairs)<=14


def test_long_horizon_without_mature_labels_has_fixed_fallback_and_no_interval():
    y,p=world()
    blend,audit=rolling_blend(y,p)
    for o in range(6,12):
        record=next(r for r in audit if r["origin"]==o and r["h"]==12)
        assert record["selection_months"]==0
        assert record["weight_a"]==record["weight_b"]==.5
        q,pairs=past_calibration(y,blend,np.arange(3),o,12)
        assert np.isnan(q).all() and not pairs


def test_calibration_is_disjoint_and_uses_only_mature_prequential_errors():
    y,p=world();blend,_=rolling_blend(y,p)
    _,selection,_=select_past_weights(y,p,16,1)
    q,calibration=past_calibration(y,blend,np.zeros(3,dtype=int),16,1)
    changed=y.copy();changed[:,17:]*=100
    q2,_=past_calibration(changed,blend,np.zeros(3,dtype=int),16,1)
    np.testing.assert_array_equal(q,q2)
    assert not ({t for _,t in selection}&{t for _,t in calibration})
    assert max(t for _,t in calibration)<=16


def test_future_alarm_and_future_residual_do_not_change_issued_forecasts():
    _,p=world();r=np.zeros((3,24));a=np.zeros_like(r,dtype=bool)
    a[:,10]=True;r[:,10]=.2
    expected=shock_adjustment(p["a"],r,a)
    a[:,16:]=True;r[:,16:]=50
    other=shock_adjustment(p["a"],r,a)
    np.testing.assert_array_equal(expected[:16],other[:16])
    assert expected[9,1,0]==101 and expected[10,1,0]>101


def test_late_news_is_not_backdated_to_event_month():
    panel,_=make_world()
    rate=pd.DataFrame({"date":pd.date_range("2020-01-01","2025-01-01"),"rate_pct":10.})
    registry=pd.DataFrame([{"event_date":"2023-07-01","published_at":"2023-09-01",
                            "regions":"r1","mo_name_patterns":"mo1","severity":3}])
    features=NewsFeatures(panel,rate,registry)
    assert not features(7)[:,4:].any()
    assert features(8)[:,5].max()==3


def test_frozen_forecast_is_preserved_when_reproduction_changes(tmp_path):
    path=tmp_path/"frozen.csv";candidate=tmp_path/"candidate.csv"
    created,different,digest=preserve_or_create(path,"original\n",candidate)
    assert created and not different
    created,different,other=preserve_or_create(path,"changed\n",candidate)
    assert not created and different and digest==other
    assert path.read_text()=="original\n" and candidate.read_text()=="changed\n"
