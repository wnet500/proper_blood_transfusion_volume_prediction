import pytest

from sklearn.model_selection import RepeatedKFold

from main.pram_gridsearch_cv import ParamGridSearch

CV_N_SPLITS = 5
CV_N_REPEATS = 3

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
      "num_layers": [2],
      "num_units": [50, 100]
  }
  param_search.conduct_ann_cv(
      grid_params=grid_params,
      valid_size_in_trainval=valid_size_in_trainval
  )


def test_conduct_xgb_cv(param_search, valid_size_in_trainval):
  grid_params = {
      "colsample_bytree": [1],
      "gamma": [0],
      "learning_rate": [0.01],
      "max_depth": [3, 5],
      "reg_lambda": [1],
      "subsample": [1]
  }
  tree_method = "gpu_hist"
  param_search.conduct_xgb_cv(
      grid_params=grid_params,
      tree_method=tree_method,
      valid_size_in_trainval=valid_size_in_trainval
  )


def test_conduct_lr_cv(param_search):
  param_search.conduct_lr_cv()


def test_conduct_rf_cv(param_search):
  grid_params = {
      'bootstrap': [False],
      'max_depth': [50, None],
      'max_features': ["auto", "sqrt"],
      'min_samples_leaf': [2],
      'min_samples_split': [2],
      'n_estimators': [1000]
  }
  param_search.conduct_rf_cv(grid_params)


def test_conduct_mlp_cv(param_search):
  grid_params = {
      "hidden_layer_sizes": [
          [60, 30, 15, 7],
          [120, 60, 30, 15],
          [180, 120, 60, 30],
          [60, 30, 15],
          [120, 60, 30],
          [180, 120, 60],
          [60, 30],
          [120, 60],
          [180, 60]
      ],
      "max_iter": [200, 500, 1000]
  }
  param_search.conduct_mlp_cv(grid_params)
