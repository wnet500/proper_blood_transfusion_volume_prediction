import pytest
import torch

from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

from main.model_trainer import ModelTrainer
from main.torch_base import CustomDataset
from main.utils import adjust_pred_value, get_adjusted_r2, disable_logging_and_userwaring


@pytest.fixture
def data_processor():
  from main.data_processor import DataProcessor
  return DataProcessor()


@pytest.fixture
def X_y_datasets(data_processor):
  X_trainval, X_test, y_trainval, y_test = data_processor.make_modeling_X_y_datasets()
  data_dict = {
      "X_trainval": X_trainval.values, "X_test": X_test.values,
      "y_trainval": y_trainval.values, "y_test": y_test.values
  }
  return data_dict


@pytest.fixture
def model_trainer(X_y_datasets):
  return ModelTrainer(X_y_datasets["X_trainval"], X_y_datasets["y_trainval"])


def test_ann_evaluation(X_y_datasets, model_trainer):

  X_test, y_test = X_y_datasets["X_test"], X_y_datasets["y_test"]
  test_dataset = CustomDataset(X_test, y_test)
  testset_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=2)

  disable_logging_and_userwaring()

  ann_trainer, ann_model = model_trainer.train_ann(
      param={'num_layers': 2, 'num_units': 50},
      num_epochs=5,
      has_bar_callback=False,
      save_model_file="ann_model_test"
  )
  results = ann_trainer.predict(ann_model, dataloaders=testset_loader)
  y_pred = torch.vstack([results[i][0] for i in range(len(results))]).cpu().numpy()[:, 0]
  print()
  print(f"ann_mse: {mean_squared_error(y_test, adjust_pred_value(y_pred)):.3f}")
  print(f"ann_adj_r2: {get_adjusted_r2(y_test, adjust_pred_value(y_pred), X_test.shape[1]):.3f}")
