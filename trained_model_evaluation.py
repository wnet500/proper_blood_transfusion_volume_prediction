# %%
import pandas as pd
import torch
import xgboost

from sklearn.metrics import mean_squared_error, r2_score

from main.data_processor import DataProcessor
from main.torch_base import ProperBloodVolPredModel
from main.utils import adjust_pred_value, get_adjusted_r2, save_blandaltman

# %%
_, X_test, _, y_test = DataProcessor().make_modeling_X_y_datasets()

# =======================================================================
# %% Current practice MOBOS evaluation with test dataset
# =======================================================================
_, msbos_test, _, true_val_test = DataProcessor().make_current_practice_X_y_datasets()

msbos_mse = mean_squared_error(true_val_test.values, msbos_test.values)
msbos_r2 = r2_score(true_val_test.values, msbos_test.values)

print()
print("=======================================================================")
print("Current practice MSBOS model evaluation")
print("=======================================================================")
print(f"MSE: {msbos_mse :.3f}")
print(f"Adjusted r square: {msbos_r2 :.3f}")

# =======================================================================
# %% ANN model evaluation with test dataset
# =======================================================================
ann_param_search_results = pd.read_csv("output/gridsearch_results/ann_results.csv")

ann_best_param = \
    ann_param_search_results\
    .sort_values(by="ann_mse_mean")\
    .head(1)[["param", "best_epoch_mean"]]

ann_model = ProperBloodVolPredModel(
    param=eval(ann_best_param["param"][0]),
    feature_num=X_test.shape[1],
    output_class=1
)
ann_model = ann_model.load_from_checkpoint("output/torch_models/trained_ann_model.ckpt")
ann_model.eval()

X_test_tensor = torch.as_tensor(X_test.values).float()
ann_y_pred = torch.flatten(
    ann_model.forward(X_test_tensor)
).detach().numpy()

ann_mse = mean_squared_error(
    y_test.values, adjust_pred_value(ann_y_pred)
)
ann_adj_r2 = get_adjusted_r2(
    y_test.values,
    adjust_pred_value(ann_y_pred),
    X_test.shape[1]
)
print()
print("=======================================================================")
print("ANN model evaluation")
print("=======================================================================")
print(f"MSE: {ann_mse :.3f}")
print(f"Adjusted r square: {ann_adj_r2 :.3f}")

# =======================================================================
# %% XGBoost model evaluation with test dataset
# =======================================================================
xgb_model = xgboost.XGBRegressor()
xgb_model.load_model("output/traditional_ml_models/trained_xgb_model.json")

xgb_y_pred = xgb_model.predict(X_test.values)

xgb_mse = mean_squared_error(
    y_test.values, adjust_pred_value(xgb_y_pred)
)
xgb_adj_r2 = get_adjusted_r2(
    y_test.values,
    adjust_pred_value(xgb_y_pred),
    X_test.shape[1]
)
print()
print("=======================================================================")
print("XGBoost model evaluation")
print("=======================================================================")
print(f"MSE: {xgb_mse :.3f}")
print(f"Adjusted r square: {xgb_adj_r2 :.3f}")
# %%
evaluation_result_df = pd.DataFrame(
    {
        "model": ["msbos", "ann", "xgboost"],
        "mse": [msbos_mse, ann_mse, xgb_mse],
        "r2": [msbos_r2, ann_adj_r2, xgb_adj_r2]
    }
)
evaluation_result_df.to_csv("output/model_evaluation_results.csv", index=False)

save_blandaltman(true_val_test.values, msbos_test.values, "msbos_bland_altman_plot")

save_blandaltman(y_test.values, ann_y_pred, "ann_bland_altman_plot_raw")
save_blandaltman(y_test.values, adjust_pred_value(ann_y_pred), "ann_bland_altman_plot_adj")

save_blandaltman(y_test.values, xgb_y_pred, "xgb_bland_altman_plot_raw")
save_blandaltman(y_test.values, adjust_pred_value(xgb_y_pred), "xgb_bland_altman_plot_adj")
