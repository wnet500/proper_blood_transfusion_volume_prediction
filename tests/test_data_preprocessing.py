import numpy as np
import pandas as pd
import pytest

from tests.config import *


@pytest.fixture
def data_processor():
  from main.data_processor import DataProcessor
  return DataProcessor(train_data_file, test_data_file)


def test_get_datasets_info(data_processor):
  datasets_info = data_processor.datasets_info

  assert isinstance(datasets_info, dict)
  assert set(["column_preprocessing_info", "outcome_info"]).issubset(set(datasets_info.keys()))
  assert set(["one_hot_columns", "numeric_columns", "drop_columns"]).issubset(
      set(datasets_info["column_preprocessing_info"].keys()))
  assert set(["target_outcome", "current_practice"]).issubset(set(datasets_info["outcome_info"].keys()))


@pytest.mark.parametrize(
    "sample_dataset_dict",
    [
        (
            {"float_col": [1.5, 24.78], "int_col": [3, 7], "cat_col": ["M", "F"], "boolean_col": [1, 0]}
        )
    ]
)
def test_convert_into_dummy_coded_datasets(data_processor, sample_dataset_dict):
  from pandas.api.types import is_numeric_dtype

  dummy_df = data_processor._convert_into_dummy_coded_datasets(
      df=pd.DataFrame(sample_dataset_dict),
      one_hot_var_list=["cat_col"],
      numeric_var_list=["float_col", "int_col", "boolean_col"]
  )

  assert np.all(dummy_df.apply(is_numeric_dtype, axis=0).values)


@pytest.mark.parametrize(
    "sample_dataset_dict, target_column, target_prop",
    [
        (
            {"cat_col": [*["A"] * 10, *["B"] * 7, *["C"] * 5, *["D"] * 3, *["E"] * 2, None]},
            "cat_col",
            0.2
        )
    ]
)
def test_reduce_num_of_categories_in_cols(data_processor, sample_dataset_dict, target_column, target_prop):
  df = pd.DataFrame(sample_dataset_dict)
  processed_df = data_processor._reduce_num_of_categories_in_cols(
      df=df.copy(),
      target_column=target_column,
      target_prop=target_prop
  )

  assert len(processed_df[target_column].value_counts().index) == \
      len(df[target_column].value_counts().index) * target_prop + 1
