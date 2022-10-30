from sklearn.model_selection import RepeatedKFold

from main.pram_gridsearch_cv import ParamGridSearch

CV_N_SPLITS = 5
CV_N_REPEATS = 3
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
# Linear Regression cv
# =======================================================================
param_search_conductor.conduct_lr_cv()

# =======================================================================
# ANN (pytorch) parameter gridsearch
# =======================================================================
ann_grid_params = {
    "num_layers": [1, 2, 3],
    "num_units": [60, 120, 180]
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
    "max_depth": [3, 6],
    "reg_lambda": [1, 3],
    "subsample": [0.8, 1],
    "reg_alpha": [0, 1],
    "min_child_weight": [1, 5]
} # 256
tree_method = "gpu_hist"
param_search_conductor.conduct_xgb_cv(
    grid_params=xgb_grid_params,
    tree_method=tree_method,
    valid_size_in_trainval=valid_size_in_trainval
)

# =======================================================================
# Random Forest parameter gridsearch
# =======================================================================
rf_grid_params = {
    'bootstrap': [True, False],
    'max_depth': [10, 50, None],
    'max_features': ["sqrt"],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [100, 500, 1000]
}  # 162
param_search_conductor.conduct_rf_cv(rf_grid_params)

# =======================================================================
# ANN (sklearn) parameter gridsearch
# =======================================================================
mlp_grid_params = {
    "hidden_layer_sizes": [
        [100],
        [100, 100],
        [100, 50, 25],
        [100, 100, 50],
        [100, 50],
        [50],
        [50, 50],
        [50, 25, 13],
        [50, 50, 25],
        [50, 25]
    ],
    "max_iter": [200]
}
param_search_conductor.conduct_mlp_cv(mlp_grid_params)
