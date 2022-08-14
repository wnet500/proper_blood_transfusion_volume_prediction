import numpy as np
import pandas as pd
import shutil
import torch
import re

from datetime import datetime
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import mean_squared_error
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
  성능 평가 지표로 mse, adjusted r square를 활용합니다.
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

        best_epoch = int(re.search(r'epoch=(?P<epoch>\d*)', checkpoint_cb.best_model_path).group('epoch'))
        best_epochs.append(best_epoch)

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
