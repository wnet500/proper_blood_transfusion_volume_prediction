import pytest

from sklearn.model_selection import RepeatedKFold

from main.pram_gridsearch_cv import ParamGridSearch

CV_N_SPLITS = 5
CV_N_REPEATS = 1

VALID_SIZE_IN_WHOLE_DATA = 0.1


@pytest.fixture
def param_search():
  cv = RepeatedKFold(n_splits=CV_N_SPLITS, n_repeats=CV_N_REPEATS, random_state=0)
  return ParamGridSearch(cv)


@pytest.fixture
def valid_size_in_trainval():
  valid_size_in_trainval = VALID_SIZE_IN_WHOLE_DATA / (1 - (1 / CV_N_SPLITS))
  return valid_size_in_trainval


def test_conduct_msbos_cv(param_search):
  param_search.conduct_current_practice_cv()


def test_conduct_ann_cv(param_search, valid_size_in_trainval):
  grid_params = {
      'num_layers': [2],
      'num_units': [50, 100]
  }
  param_search.conduct_ann_cv(
      grid_params=grid_params,
      valid_size_in_trainval=valid_size_in_trainval
  )


def test_conduct_xgb_cv(param_search, valid_size_in_trainval):
  grid_params = {
      'colsample_bytree': [1],
      'gamma': [0],
      'learning_rate': [0.01],
      'max_depth': [3, 5],
      'reg_lambda': [1],
      'subsample': [1]
  }
  tree_method = "gpu_hist"
  param_search.conduct_xgb_cv(
      grid_params=grid_params,
      tree_method=tree_method,
      valid_size_in_trainval=valid_size_in_trainval
  )
