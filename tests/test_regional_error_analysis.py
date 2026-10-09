import pandas as pd
import pytest
from scripts.analyze_regional_errors import summarize


def test_error_summary_pools_observations_and_keeps_improvement_sign():
    frame=pd.DataFrame({'region':['A','A','A'],'actual':[100.,100.,1000.],
        'error':[10.,-20.,-100.],'abs_error':[10.,20.,100.],
        'original_abs_error':[20.,40.,150.]})
    row=summarize(frame,['region']).iloc[0]
    assert row.n==3 and row.mae==pytest.approx(130/3)
    assert row.wape==pytest.approx(130/1200)
    assert row.bias==pytest.approx(-110/3)
    assert row.improvement_vs_original==pytest.approx(80/3)
    assert row.underprediction_share==pytest.approx(2/3)


def test_multiple_group_keys_do_not_mix_months_or_regions():
    frame=pd.DataFrame({'region':['A','A','B'],'month':['2024-07','2024-08','2024-07'],
        'actual':[100.,100.,100.],'error':[1.,2.,-3.],'abs_error':[1.,2.,3.],
        'original_abs_error':[1.,2.,3.]})
    table=summarize(frame,['region','month'])
    assert len(table)==3
    assert table.query('region=="B"').iloc[0].bias==-3
