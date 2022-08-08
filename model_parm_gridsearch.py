from sklearn.model_selection import RepeatedKFold

from main.pram_gridsearch_cv import ParamGridSearch

CV_N_SPLITS = 5
CV_N_REPEATS = 10

cv = RepeatedKFold(
    n_splits=CV_N_SPLITS,
    n_repeats=CV_N_REPEATS,
    random_state=0
)
param_search_conductor = ParamGridSearch(cv)

# ANN parameter gridsearch
ann_grid_params = {
    'num_layers': [2, 3, 4, 5],
    'num_units': [50, 100, 150]
}
valid_size_in_whole_datasets = 0.1
valid_size_in_trainval = valid_size_in_whole_datasets / (1 - (1 / CV_N_SPLITS))
param_search_conductor.conduct_ann_cv(
    grid_params=ann_grid_params,
    valid_size_in_trainval=valid_size_in_trainval
)
