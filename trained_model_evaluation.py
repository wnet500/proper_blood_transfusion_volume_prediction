# %%
import pandas as pd
import torch

from sklearn.metrics import mean_squared_error

from main.data_processor import DataProcessor
from main.torch_base import ProperBloodVolPredModel
from main.utils import adjust_pred_value, get_adjusted_r2, save_blandaltman

# %%
_, X_test, _, y_test = DataProcessor().make_modeling_X_y_datasets()

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
  y_test, adjust_pred_value(ann_y_pred)
)
ann_adj_r2 = get_adjusted_r2(
  y_test,
  adjust_pred_value(ann_y_pred),
  X_test.shape[1]
)
print()
print("=======================================================================")
print("ANN model evaluation")
print("=======================================================================")
print(f"MSE: {ann_mse :.3f}")
print(f"Adjusted r square: {ann_adj_r2 :.3f}")

# %%
evaluation_result_df = pd.DataFrame(
  {
    "model": ["ann"],
    "mse": [ann_mse],
    "r2": [ann_adj_r2]
  }
)
evaluation_result_df.to_csv("output/model_evaluation_results.csv", index=False)

save_blandaltman(y_test.values, ann_y_pred, "ann_bland_altman_plot_raw")
save_blandaltman(y_test.values, adjust_pred_value(ann_y_pred), "ann_bland_altman_plot_adj")

# %%
