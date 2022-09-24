# %%
import pandas as pd

from main.data_processor import DataProcessor
from main.model_trainer import ModelTrainer
from main.utils import disable_logging_and_userwaring

# %%
X_train, _, y_train, _ = DataProcessor().make_modeling_X_y_datasets()
model_trainer = ModelTrainer(X_train, y_train)

# =======================================================================
# %% ANN (pytorch) model training
# =======================================================================
ann_param_search_results = pd.read_csv("output/gridsearch_results/ann_results.csv")

ann_best_param = \
    ann_param_search_results\
    .sort_values(by="ann_mse_mean")\
    .head(1)[["param", "best_epoch_mean"]]

disable_logging_and_userwaring()
ann_trainer, ann_model = model_trainer.train_ann(
    param=eval(ann_best_param["param"][0]),
    num_epochs=int(ann_best_param["best_epoch_mean"][0]),
    has_bar_callback=True,
    save_model_file="trained_ann_model"
)

# =======================================================================
# %% XGBoost model training
# =======================================================================
xgb_param_search_results = pd.read_csv("output/gridsearch_results/xgb_results.csv")

xgb_best_param = \
    xgb_param_search_results\
    .sort_values(by="xgb_mse_mean")\
    .head(1)[["param", "early_stopping_round_mean"]]

tree_method = "gpu_hist"

xgb_model = model_trainer.train_xgboost(
    param=eval(xgb_best_param["param"][0]),
    tree_method=tree_method,
    n_estimators=int(xgb_best_param["early_stopping_round_mean"][0]),
    save_model_file="trained_xgb_model"
)

# =======================================================================
# %% Linear Regression model training
# =======================================================================
lr_model = model_trainer.train_linear_regression(save_model_file="trained_lr_model")


# =======================================================================
# %% Random Forest model training
# =======================================================================
rf_param_search_results = pd.read_csv("output/gridsearch_results/rf_results.csv")
rf_best_param = \
    rf_param_search_results\
    .sort_values(by="rf_mse_mean")\
    .head(1)[["param"]]

rf_model = model_trainer.train_random_forest(
    param=eval(rf_best_param["param"][0]),
    save_model_file="trained_rf_model"
)

# =======================================================================
# %% ANN (sklearn) model training
# =======================================================================
mlp_param_search_results = pd.read_csv("output/gridsearch_results/mlp_results.csv")
mlp_best_param = \
    mlp_param_search_results\
    .sort_values(by="mlp_mse_mean")\
    .head(1)[["param"]]

mlp_model = model_trainer.train_mlp(
    param=eval(mlp_best_param["param"][0]),
    save_model_file="trained_mlp_model"
)
