import numpy as np
import pandas as pd
import shutil
import torch
import re

from datetime import datetime
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, ParameterGrid, train_test_split
from torch.utils.data import DataLoader

from main.config import data_loader_num_workwers, evalset_loader_batch_size, testset_loader_batch_size
from main.data_processor import DataProcessor
from main.model_trainer import ModelTrainer
from main.torch_base import CustomDataset
from main.utils import (
    adjust_pred_value,
    get_adjusted_r2,
    get_95_conf_interval,
    disable_logging_and_userwaring
)


class ParamGridSearch:
  """
  각 모델에 대해 지정한 파라미터 후보들에 대해 파라미터 조합 그리드서치를 진행하고,
  결과를 output/gridsearch_results 폴더에 저장합니다.
  성능 평가 지표로 mse, adjusted r square, r square를 활용합니다.
  """

  def __init__(
      self,
      cv=RepeatedKFold(n_splits=5, n_repeats=10, random_state=0)
  ) -> None:
    self.cv = cv
    self.X_trainval, self.X_test, self.y_trainval, self.y_test = self._get_np_X_y_datasets()
    self.ouput_dir = Path(__file__).parent.parent.joinpath("output")

  def _get_np_X_y_datasets(self):
    X_trainval, X_test, y_trainval, y_test = DataProcessor().make_modeling_X_y_datasets()
    return (X_trainval.values, X_test.values, y_trainval.values, y_test.values)

  def conduct_current_practice_cv(self):
    """
    기존 프렉티스로 사용되는 msbos의 cv 수행 성능을 구합니다.
    """
    x_trainval, x_test, y_trainval, y_test = DataProcessor().make_current_practice_X_y_datasets()

    start_time = datetime.now()
    print("\n=======================================================================")
    print(f"[{start_time}] Start current practice (msbos) cv")
    print("=======================================================================")

    mse_evals = []
    r2_evals = []

    for trainval_index, test_index in self.cv.split(x_trainval.values, y_trainval.values):
      _, x_test_in = x_trainval.values[trainval_index], x_trainval.values[test_index]
      _, y_test_in = y_trainval.values[trainval_index], y_trainval.values[test_index]

      mse = mean_squared_error(y_test_in, x_test_in)
      r2 = r2_score(y_test_in, x_test_in)

      mse_evals.append(mse)
      r2_evals.append(r2)

    print(f"Current practice MSE (cv mean): {np.mean(mse_evals) :.3f}")
    print(f"ANN Adj r2 (cv mean): {np.mean(r2_evals) :.3f}")
    print(f"Cumulative time: {(datetime.now() - start_time).seconds / 60 :.3f} minutes\n")

    mse_mean, mse_95_ci_lower, mse_95_ci_upper = get_95_conf_interval(mse_evals)
    r2_mean, r2_95_ci_lower, r2_95_ci_upper = get_95_conf_interval(r2_evals)

    result_df = pd.DataFrame({
        "prev_practice_mse_cv_results": [mse_evals],
        "prev_practice_r2_results": [r2_evals],
        "prev_practice_mse_mean": [mse_mean],
        "prev_practice_mse_95_ci_lower": [mse_95_ci_lower],
        "prev_practice_mse_95_ci_upper": [mse_95_ci_upper],
        "prev_practice_r2_mean": [r2_mean],
        "prev_practice_r2_95_ci_lower": [r2_95_ci_lower],
        "prev_practice_r2_95_ci_upper": [r2_95_ci_upper]
    })

    result_df.to_csv(
        str(self.ouput_dir.joinpath(
            "gridsearch_results",
            "current_practice_msbos_results.csv"
        )),
        index=False
    )

  def conduct_ann_cv(
      self,
      grid_params: dict,
      valid_size_in_trainval: float = 1 / 8,
      is_disable_logging_and_userwaring: bool = True
  ):
    if is_disable_logging_and_userwaring:
      disable_logging_and_userwaring()

    tb_log_path = self.ouput_dir.joinpath("gridsearch_tb_logs")

    self.ouput_dir.joinpath("gridsearch_results").mkdir(parents=True, exist_ok=True)
    shutil.rmtree(str(tb_log_path), ignore_errors=True)

    start_time = datetime.now()
    gridsearch_results = []

    print("\n=======================================================================")
    print(f"[{start_time}] Start ann model parameter search cv")
    print("=======================================================================")

    for param_ind, param in enumerate(ParameterGrid(grid_params)):
      mse_evals = []
      adj_r2_evals = []
      best_epochs = []

      print()
      print(f"[{param_ind + 1}] param:\n{param}")

      for cv_num, (trainval_index, test_index) in enumerate(self.cv.split(self.X_trainval, self.y_trainval)):
        X_trainval_in, X_test_in = self.X_trainval[trainval_index], self.X_trainval[test_index]
        y_trainval_in, y_test_in = self.y_trainval[trainval_index], self.y_trainval[test_index]

        X_train_in, X_valid_in, y_train_in, y_valid_in = train_test_split(
            X_trainval_in, y_trainval_in,
            test_size=valid_size_in_trainval,
            random_state=0
        )

        eval_dataset = CustomDataset(X_valid_in, y_valid_in)
        test_dataset = CustomDataset(X_test_in, y_test_in)

        evalset_loader = DataLoader(
            eval_dataset,
            batch_size=evalset_loader_batch_size,
            shuffle=False,
            num_workers=data_loader_num_workwers
        )
        testset_loader = DataLoader(
            test_dataset,
            batch_size=testset_loader_batch_size,
            shuffle=False,
            num_workers=data_loader_num_workwers
        )

        logger = TensorBoardLogger(
            save_dir=str(tb_log_path),
            name=f"param_grid_{param_ind}",
            version=f"cross_validation_{cv_num}"
        )
        checkpoint_cb = ModelCheckpoint(
            monitor="val_loss",
            dirpath=str(tb_log_path.joinpath(f"param_grid_{param_ind}/cross_validation_{cv_num}")),
            filename="ann_{epoch:03d}_{val_loss:.3f}",
            mode="min"
        )

        model_trainer = ModelTrainer(X_train_in, y_train_in)
        trainer, ann_model = model_trainer.train_ann(
            param=param,
            logger=logger,
            checkpoint_cb=checkpoint_cb,
            evalset_loader=evalset_loader,
            has_bar_callback=False
        )

        results = trainer.predict(ann_model, dataloaders=testset_loader)
        y_pred = torch.vstack([results[i][0] for i in range(len(results))]).cpu().numpy()[:, 0]
        y_pred = adjust_pred_value(y_pred)

        mse = mean_squared_error(y_test_in, y_pred)
        mse_evals.append(mse)

        adj_r2 = get_adjusted_r2(y_test_in, y_pred, X_test_in.shape[1])
        adj_r2_evals.append(adj_r2)

        best_epoch = int(re.search(r"epoch=(?P<epoch>\d*)", checkpoint_cb.best_model_path).group("epoch"))
        best_epochs.append(best_epoch + 1)

      gridsearch_results.append((param, mse_evals, adj_r2_evals, best_epochs))

      print(f"ANN MSE (cv mean): {np.mean(mse_evals) :.3f}")
      print(f"ANN Adj r2 (cv mean): {np.mean(adj_r2_evals) :.3f}")
      print(f"Cumulative time: {(datetime.now() - start_time).seconds / 60 :.3f} minutes\n")

    gridsearch_result_df = pd.DataFrame(
        gridsearch_results,
        columns=["param", "ann_mse_cv_results", "ann_adj_r2_cv_results", "best_epochs"]
    )
    gridsearch_result_df["best_epoch_mean"] = round(gridsearch_result_df["best_epochs"].apply(lambda x: np.mean(x)))

    conf_info_mse = gridsearch_result_df["ann_mse_cv_results"].apply(get_95_conf_interval).apply(pd.Series)
    conf_info_mse.columns = ["ann_mse_mean", "ann_mse_95_ci_lower", "ann_mse_95_ci_upper"]
    conf_info_r2 = gridsearch_result_df["ann_adj_r2_cv_results"].apply(get_95_conf_interval).apply(pd.Series)
    conf_info_r2.columns = ["ann_adj_r2_mean", "ann_adj_r2_95_ci_lower", "ann_adj_r2_95_ci_upper"]

    gridsearch_result_df = pd.concat(
        [gridsearch_result_df, conf_info_mse, conf_info_r2],
        axis=1
    ).sort_values(by="ann_mse_mean")

    gridsearch_result_df.to_csv(str(self.ouput_dir.joinpath("gridsearch_results", "ann_results.csv")), index=False)

  def conduct_xgb_cv(
      self,
      grid_params: dict,
      tree_method: str = "gpu_hist",
      valid_size_in_trainval=1 / 8
  ):
    start_time = datetime.now()
    gridsearch_results = []
    print("\n=======================================================================")
    print(f"[{start_time}] Start XGBoost model parameter search cv")
    print("=======================================================================")

    for param_ind, param in enumerate(ParameterGrid(grid_params)):
      mse_evals = []
      adj_r2_evals = []
      best_ntree_limits = []

      print()
      print(f"[{param_ind + 1}] param:\n{param}")

      for trainval_index, test_index in self.cv.split(self.X_trainval, self.y_trainval):
        X_trainval_in, X_test_in = self.X_trainval[trainval_index], self.X_trainval[test_index]
        y_trainval_in, y_test_in = self.y_trainval[trainval_index], self.y_trainval[test_index]

        X_train_in, X_valid_in, y_train_in, y_valid_in = train_test_split(
            X_trainval_in, y_trainval_in,
            test_size=valid_size_in_trainval,
            random_state=0
        )

        xgb_model = ModelTrainer(X_train_in, y_train_in).train_xgboost(
            param=param,
            tree_method=tree_method,
            eval_set=[(X_valid_in, y_valid_in)],
            n_estimators=10000
        )
        y_pred = adjust_pred_value(xgb_model.predict(X_test_in))

        mse = mean_squared_error(y_test_in, y_pred)
        mse_evals.append(mse)

        adj_r2 = get_adjusted_r2(y_test_in, y_pred, X_test_in.shape[1])
        adj_r2_evals.append(adj_r2)

        best_ntree_limits.append(xgb_model.best_ntree_limit)

      gridsearch_results.append((param, mse_evals, adj_r2_evals, best_ntree_limits))

      print(f"XGBoost MSE (cv mean): {np.mean(mse_evals) :.3f}")
      print(f"XGBoost Adj r2 (cv mean): {np.mean(adj_r2_evals) :.3f}")
      print(f"Cumulative time: {(datetime.now() - start_time).seconds / 60 :.3f} minutes\n")

    gridsearch_result_df = pd.DataFrame(
        gridsearch_results,
        columns=["param", "xgb_mse_cv_results", "xgb_adj_r2_cv_results", "early_stopping_rounds"]
    )
    gridsearch_result_df["early_stopping_round_mean"] = \
        round(gridsearch_result_df["early_stopping_rounds"].apply(lambda x: np.mean(x)))

    conf_info_mse = gridsearch_result_df["xgb_mse_cv_results"].apply(get_95_conf_interval).apply(pd.Series)
    conf_info_mse.columns = ["xgb_mse_mean", "xgb_mse_95_ci_lower", "xgb_mse_95_ci_upper"]
    conf_info_r2 = gridsearch_result_df["xgb_adj_r2_cv_results"].apply(get_95_conf_interval).apply(pd.Series)
    conf_info_r2.columns = ["xgb_adj_r2_mean", "xgb_adj_r2_95_ci_lower", "xgb_adj_r2_95_ci_upper"]

    gridsearch_result_df = pd.concat(
        [gridsearch_result_df, conf_info_mse, conf_info_r2],
        axis=1
    ).sort_values(by="xgb_mse_mean")

    gridsearch_result_df.to_csv(str(self.ouput_dir.joinpath("gridsearch_results", "xgb_results.csv")), index=False)

  def conduct_lr_cv(self):
    start_time = datetime.now()
    print("\n=======================================================================")
    print(f"[{start_time}] Start Linear Regression model parameter search cv")
    print("=======================================================================")

    mse_evals = []
    adj_r2_evals = []

    for trainval_index, test_index in self.cv.split(self.X_trainval, self.y_trainval):
      X_train_in, X_test_in = self.X_trainval[trainval_index], self.X_trainval[test_index]
      y_train_in, y_test_in = self.y_trainval[trainval_index], self.y_trainval[test_index]

      lr_model = ModelTrainer(X_train_in, y_train_in).train_linear_regression()
      y_pred = adjust_pred_value(lr_model.predict(X_test_in))
      y_pred = np.where(y_pred > 15, 15, y_pred)

      mse = mean_squared_error(y_test_in, y_pred)
      mse_evals.append(mse)

      adj_r2 = get_adjusted_r2(y_test_in, y_pred, X_test_in.shape[1])
      adj_r2_evals.append(adj_r2)

    print(f"Linear Regression MSE (cv mean): {np.mean(mse_evals) :.3f}")
    print(f"Linear Regression Adj r2 (cv mean): {np.mean(adj_r2_evals) :.3f}")
    print(f"Cumulative time: {(datetime.now() - start_time).seconds / 60 :.3f} minutes\n")

    lr_mse_mean, lr_mse_95_ci_lower, lr_mse_95_ci_upper = get_95_conf_interval(mse_evals)
    lr_adj_r2_mean, lr_adj_r2_95_ci_lower, lr_adj_r2_95_ci_upper = get_95_conf_interval(adj_r2_evals)

    result_df = pd.DataFrame({
        "lr_mse_cv_results": [mse_evals],
        "lr_adj_r2_results": [adj_r2_evals],
        "lr_mse_mean": [lr_mse_mean],
        "lr_mse_95_ci_lower": [lr_mse_95_ci_lower],
        "lr_mse_95_ci_upper": [lr_mse_95_ci_upper],
        "lr_adj_r2_mean": [lr_adj_r2_mean],
        "lr_adj_r2_95_ci_lower": [lr_adj_r2_95_ci_lower],
        "lr_adj_r2_95_ci_upper": [lr_adj_r2_95_ci_upper]
    })

    result_df.to_csv(str(self.ouput_dir.joinpath("gridsearch_results", "lr_results.csv")), index=False)

  def conduct_rf_cv(self, grid_params: dict):
    start_time = datetime.now()
    gridsearch_results = []
    print("\n=======================================================================")
    print(f"[{start_time}] Start Random Forest model parameter search cv")
    print("=======================================================================")

    for param_ind, param in enumerate(ParameterGrid(grid_params)):
      mse_evals = []
      adj_r2_evals = []

      print()
      print(f"[{param_ind + 1}] param:\n{param}")

      for train_index, test_index in self.cv.split(self.X_trainval, self.y_trainval):
        X_train_in, X_test_in = self.X_trainval[train_index], self.X_trainval[test_index]
        y_train_in, y_test_in = self.y_trainval[train_index], self.y_trainval[test_index]

        rf_model = ModelTrainer(X_train_in, y_train_in).train_random_forest(param)
        y_pred = adjust_pred_value(rf_model.predict(X_test_in))

        mse = mean_squared_error(y_test_in, y_pred)
        mse_evals.append(mse)

        adj_r2 = get_adjusted_r2(y_test_in, y_pred, X_test_in.shape[1])
        adj_r2_evals.append(adj_r2)

      gridsearch_results.append((param, mse_evals, adj_r2_evals))

      print(f"Random Forest MSE (cv mean): {np.mean(mse_evals) :.3f}")
      print(f"Random Forest Adj r2 (cv mean): {np.mean(adj_r2_evals) :.3f}")
      print(f"Cumulative time: {(datetime.now() - start_time).seconds / 60 :.3f} minutes\n")

    gridsearch_result_df = pd.DataFrame(
        gridsearch_results,
        columns=["param", "rf_mse_cv_results", "rf_adj_r2_cv_results"]
    )
    conf_info_mse = gridsearch_result_df["rf_mse_cv_results"].apply(get_95_conf_interval).apply(pd.Series)
    conf_info_mse.columns = ["rf_mse_mean", "rf_mse_95_ci_lower", "rf_mse_95_ci_upper"]
    conf_info_r2 = gridsearch_result_df["rf_adj_r2_cv_results"].apply(get_95_conf_interval).apply(pd.Series)
    conf_info_r2.columns = ["rf_adj_r2_mean", "rf_adj_r2_95_ci_lower", "rf_adj_r2_95_ci_upper"]

    gridsearch_result_df = pd.concat(
        [gridsearch_result_df, conf_info_mse, conf_info_r2],
        axis=1
    ).sort_values(by="rf_mse_mean")

    gridsearch_result_df.to_csv(str(self.ouput_dir.joinpath("gridsearch_results", "rf_results.csv")), index=False)

  def conduct_mlp_cv(self, grid_params: dict):
    start_time = datetime.now()
    gridsearch_results = []
    print("\n=======================================================================")
    print(f"[{start_time}] Start MLP model parameter search cv")
    print("=======================================================================")

    for param_ind, param in enumerate(ParameterGrid(grid_params)):
      mse_evals = []
      adj_r2_evals = []

      print()
      print(f"[{param_ind + 1}] param:\n{param}")

      for train_index, test_index in self.cv.split(self.X_trainval, self.y_trainval):
        X_train_in, X_test_in = self.X_trainval[train_index], self.X_trainval[test_index]
        y_train_in, y_test_in = self.y_trainval[train_index], self.y_trainval[test_index]

        mlp_model = ModelTrainer(X_train_in, y_train_in).train_mlp(param)
        y_pred = adjust_pred_value(mlp_model.predict(X_test_in))

        mse = mean_squared_error(y_test_in, y_pred)
        mse_evals.append(mse)

        adj_r2 = get_adjusted_r2(y_test_in, y_pred, X_test_in.shape[1])
        adj_r2_evals.append(adj_r2)

      gridsearch_results.append((param, mse_evals, adj_r2_evals))

      print(f"MLP MSE (cv mean): {np.mean(mse_evals) :.3f}")
      print(f"MLP Adj r2 (cv mean): {np.mean(adj_r2_evals) :.3f}")
      print(f"Cumulative time: {(datetime.now() - start_time).seconds / 60 :.3f} minutes\n")

    gridsearch_result_df = pd.DataFrame(
        gridsearch_results,
        columns=["param", "mlp_mse_cv_results", "mlp_adj_r2_cv_results"]
    )
    conf_info_mse = gridsearch_result_df["mlp_mse_cv_results"].apply(get_95_conf_interval).apply(pd.Series)
    conf_info_mse.columns = ["mlp_mse_mean", "mlp_mse_95_ci_lower", "mlp_mse_95_ci_upper"]
    conf_info_r2 = gridsearch_result_df["mlp_adj_r2_cv_results"].apply(get_95_conf_interval).apply(pd.Series)
    conf_info_r2.columns = ["mlp_adj_r2_mean", "mlp_adj_r2_95_ci_lower", "mlp_adj_r2_95_ci_upper"]

    gridsearch_result_df = pd.concat(
        [gridsearch_result_df, conf_info_mse, conf_info_r2],
        axis=1
    ).sort_values(by="mlp_mse_mean")

    gridsearch_result_df.to_csv(str(self.ouput_dir.joinpath("gridsearch_results", "mlp_results.csv")), index=False)
