import pytest

from sklearn.model_selection import RepeatedKFold

from main.pram_gridsearch_cv import ParamGridSearch

CV_N_SPLITS = 5
CV_N_REPEATS = 1


@pytest.fixture
def param_search():
  cv = RepeatedKFold(n_splits=CV_N_SPLITS, n_repeats=CV_N_REPEATS, random_state=0)
  return ParamGridSearch(cv)


def test_conduct_ann_cv(param_search):
  grid_params = {
      'num_layers': [2],
      'num_units': [50, 100]
  }
  valid_size_in_whole_datasets = 0.1
  valid_size_in_trainval = valid_size_in_whole_datasets / (1 - (1 / CV_N_SPLITS))
  param_search.conduct_ann_cv(grid_params=grid_params, valid_size_in_trainval=valid_size_in_trainval)
