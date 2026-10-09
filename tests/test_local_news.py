from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.evaluate_llm_news import select_validation
from scripts.evaluate_local_news import choose_local_policies, assemble_coverage, matched_coverage
from src.mun_eval import make_grid


def test_local_policy_selection_ignores_test_targets_and_keeps_zero_correction():
    months = np.arange(24,dtype=float)
    panel = SimpleNamespace(values=np.vstack([100+months**2*.001,101+months**2*.002]))
    base = np.full((24,13,2),100.)
    names = ['all_windows','economic_windows','economic_shocks_windows',
             'economic_decay_14d','economic_decay_30d','economic_decay_90d']
    candidates = {(v,100,s):base*(1+s) for v in names for s in [0.,.1]}
    _,chosen,_ = select_validation(panel,candidates,make_grid())
    original = choose_local_policies(chosen)
    panel.values[:,18:] *= 100
    _,changed,_ = select_validation(panel,candidates,make_grid())
    pd.testing.assert_frame_equal(original,choose_local_policies(changed))
    assert original.shrink.eq(0).all()
    assert original[original.family.eq('local_news')].variant.eq('economic_windows').all()


def test_local_and_broad_policy_can_choose_different_filters_on_validation():
    rows = [{'variant':v,'alpha':100,'shrink':.1,'h':h,'validation_mae':error}
            for v,error in [('all_windows',1.),('economic_windows',2.),('economic_shocks_windows',3.),
                            ('economic_decay_14d',4.),('economic_decay_30d',5.),('economic_decay_90d',6.)]
            for h in [1,3,6,12]]
    chosen = choose_local_policies(pd.DataFrame(rows))
    assert chosen[chosen.family.eq('local_news')].variant.eq('economic_windows').all()
    assert chosen[chosen.family.eq('local_policy')].variant.eq('all_windows').all()


def test_coverage_tracks_selected_filter_and_decay_on_each_horizon():
    selected = {'economic_coverage_windows':np.ones((24,13,2)),
                'economic_coverage_decay_30d':np.full((24,13,2),2.),
                'economic_shocks_coverage':np.full((24,13,2),3.)}
    policy = pd.DataFrame([{'family':'local_news','variant':'economic_windows','h':1},
                          {'family':'local_news','variant':'economic_decay_30d','h':3},
                          {'family':'local_news','variant':'economic_shocks_windows','h':6}])
    coverage = assemble_coverage(selected,policy,'local_news')
    assert (coverage[:,1]==1).all() and (coverage[:,3]==2).all() and (coverage[:,6]==3).all()
    assert matched_coverage('all_windows')=='all_coverage'
    assert selected['economic_coverage_windows'].min()==selected['economic_coverage_windows'].max()==1
