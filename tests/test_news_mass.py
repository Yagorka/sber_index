import numpy as np
import pandas as pd
import pytest
from tests.test_llm_news import panel, record
from src.llm_news import aggregate_features
from scripts.evaluate_news_mass import choose_families


def test_content_mass_decays_without_events_but_fraction_does_not():
    p=panel();rows=pd.DataFrame([record(sampling_weight=4)])
    mass,_=aggregate_features(p,rows,half_life_days=30,content_mass=True)
    fractions,_=aggregate_features(p,rows,half_life_days=30)
    assert fractions[1,1,3]==fractions[2,1,3]==1
    ratio=2**(-31/30)
    assert np.expm1(mass[2,1,3])/np.expm1(mass[1,1,3])==pytest.approx(ratio)
    assert mass[2,1,7]<0 and abs(mass[2,1,7])<abs(mass[1,1,7])
    assert not mass[:,:,13].any()  # unknown income not a known-direction event
    assert not mass[:,2,2:].any()  # another municipality/category receives no event


def test_mass_future_invariance_cancellation_and_neutral_availability():
    rows=pd.DataFrame([record(),record(duplicate_key='b',sentiment=1,income_direction=0)])
    mass,_=aggregate_features(panel(),rows,half_life_days=14,content_mass=True)
    assert not mass[:,:,7].any()  # signed sentiments cancel; event counts remain
    assert mass[1,1,3]>0 and mass[1,1,4]>0 and mass[1,1,13]>0
    future=record(duplicate_key='future',historical_available_at='2024-04-15T00:00:00+03:00')
    changed,_=aggregate_features(panel(),pd.concat([rows,pd.DataFrame([future])]),half_life_days=14,content_mass=True)
    np.testing.assert_array_equal(mass[:3],changed[:3])
    with pytest.raises(ValueError):aggregate_features(panel(),rows,content_mass=True)


def test_family_ties_keep_zero_correction():
    choices=pd.DataFrame([dict(variant=f'{prefix}_{d}d',h=h,validation_mae=1,shrink=s,alpha=100)
        for prefix in ['mass','fractions'] for d in [14,30,90] for h in [1,3,6,12] for s in [.1,0]])
    chosen=choose_families(choices)
    assert len(chosen)==8 and chosen.shrink.eq(0).all()
