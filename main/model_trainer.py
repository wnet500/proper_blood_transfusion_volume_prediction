import joblib
import numpy as np
import pytorch_lightning as pl
import xgboost

from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader
from tqdm import tqdm

from main.config import pl_accelerator, pl_devices, pl_global_seed, data_loader_num_workwers
from main.torch_base import CustomDataset, ProperBloodVolPredModel


class ModelTrainer:
  def __init__(
      self,
      X_train: np.ndarray,
      y_train: np.ndarray
  ) -> None:
    self.X_train = X_train
    self.y_train = y_train
    self.ouput_dir = Path(__file__).parent.parent.joinpath("output")

  def train_ann(
      self,
      param: dict,
      logger: TensorBoardLogger = False,
      num_epochs: int = 1000,
      has_bar_callback: bool = False,
      checkpoint_cb: ModelCheckpoint = None,
      evalset_loader: DataLoader = None,
      save_model_file: str = "ann_model"
  ):
    pl.seed_everything(pl_global_seed, workers=True)  # Global seed setting

    train_dataset = CustomDataset(self.X_train, self.y_train)
    trainset_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=data_loader_num_workwers)

    callbacks = []
    if has_bar_callback:
      class LitProgressBar(TQDMProgressBar):
        def init_validation_tqdm(self):
          bar = super().init_validation_tqdm()
          return bar

        def init_predict_tqdm(self):
          bar = tqdm(disable=True)
          return bar

      bar = LitProgressBar()
      callbacks = [*callbacks, bar]
    if evalset_loader and checkpoint_cb:
      early_stop_callback = EarlyStopping(
          monitor="val_loss", patience=20, verbose=False, mode="min")
      callbacks = [*callbacks, checkpoint_cb, early_stop_callback]

    ann_model = ProperBloodVolPredModel(
        param=param,
        feature_num=self.X_train.shape[1],
        output_class=1
    )

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=num_epochs,
        accelerator=pl_accelerator,
        devices=pl_devices,
        gradient_clip_val=5,
        log_every_n_steps=1,
        accumulate_grad_batches=1,
        callbacks=callbacks if callbacks else None,
        deterministic=True,
        enable_progress_bar=has_bar_callback,
        enable_model_summary=False,
        enable_checkpointing=True if checkpoint_cb else False
    )

    if evalset_loader:
      trainer.fit(
          ann_model,
          train_dataloaders=trainset_loader,
          val_dataloaders=evalset_loader
      )
      ann_model = ann_model.load_from_checkpoint(checkpoint_cb.best_model_path)
    else:
      trainer.fit(
          ann_model,
          train_dataloaders=trainset_loader,
      )
      save_path = str(self.ouput_dir.joinpath("torch_models", f"{save_model_file}.ckpt"))
      trainer.save_checkpoint(save_path)
      ann_model = ann_model.load_from_checkpoint(save_path)

    return trainer, ann_model

  def train_xgboost(
      self,
      param: dict,
      tree_method: str,
      eval_set: list = None,
      n_estimators: int = 10000,
      save_model_file: str = None
  ):
    xgb_model = xgboost.XGBRegressor(
        objective="reg:squarederror",
        random_state=0,
        n_estimators=n_estimators,
        tree_method=tree_method,
        **param
    )

    if eval_set:
      xgb_model.fit(
          self.X_train, self.y_train,
          early_stopping_rounds=1000,
          eval_set=eval_set,
          eval_metric="rmse",
          verbose=False
      )
    else:
      xgb_model.fit(
          self.X_train, self.y_train,
          verbose=False
      )

    if save_model_file:
      save_path = str(self.ouput_dir.joinpath("traditional_ml_models", f"{save_model_file}.json"))
      xgb_model.save_model(save_path)

    return xgb_model

  def train_linear_regression(self, save_model_file: str = None):
    lr_model = LinearRegression()
    lr_model.fit(self.X_train, self.y_train)

    if save_model_file:
      save_path = str(self.ouput_dir.joinpath("traditional_ml_models", f"{save_model_file}.joblib"))
      joblib.dump(lr_model, save_path)

    return lr_model

  def train_random_forest(
      self,
      param: dict,
      save_model_file: str = None
  ):
    rf_model = RandomForestRegressor(random_state=0, n_jobs=-1, **param)
    rf_model.fit(self.X_train, self.y_train)

    if save_model_file:
      save_path = str(self.ouput_dir.joinpath("traditional_ml_models", f"{save_model_file}.joblib"))
      joblib.dump(rf_model, save_path)

    return rf_model
