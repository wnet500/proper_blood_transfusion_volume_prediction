from sklearn.model_selection import RepeatedKFold

from main.pram_gridsearch_cv import ParamGridSearch

CV_N_SPLITS = 5
CV_N_REPEATS = 10
VALID_SIZE_IN_WHOLE_DATA = 0.1

valid_size_in_trainval = VALID_SIZE_IN_WHOLE_DATA / (1 - (1 / CV_N_SPLITS))
cv = RepeatedKFold(
    n_splits=CV_N_SPLITS,
    n_repeats=CV_N_REPEATS,
    random_state=0
)
param_search_conductor = ParamGridSearch(cv)

# =======================================================================
# Current practice (msbos) cv
# =======================================================================
param_search_conductor.conduct_current_practice_cv()

# =======================================================================
# ANN parameter gridsearch
# =======================================================================
ann_grid_params = {
    "num_layers": [1, 2, 3],
    "num_units": [50, 100, 150]
}  # 9
param_search_conductor.conduct_ann_cv(
    grid_params=ann_grid_params,
    valid_size_in_trainval=valid_size_in_trainval
)

# =======================================================================
# XGBoost parameter gridsearch
# =======================================================================
xgb_grid_params = {
    "colsample_bytree": [0.8, 1],
    "gamma": [0, 0.1],
    "learning_rate": [0.01, 0.1],
    "max_depth": [3, 5, 7],
    "reg_lambda": [1, 2, 3],
    "subsample": [0.8, 1]
}  # 144
tree_method = "gpu_hist"
param_search_conductor.conduct_xgb_cv(
    grid_params=xgb_grid_params,
    tree_method=tree_method,
    valid_size_in_trainval=valid_size_in_trainval
)
