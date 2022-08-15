import numpy as np
import pandas as pd

from pathlib import Path
from typing import List, Union

from main.config import train_dataset_path, test_dataset_path, dataset_info


class DataProcessor:
  def __init__(
      self
  ) -> None:
      self.data_dir = Path(__file__).parent.parent.joinpath("datasets")
      self.train_dataset_path = train_dataset_path
      self.test_dataset_path = test_dataset_path
      self.datasets_info = dataset_info

  def _load_raw_datasets(self) -> List[pd.DataFrame]:
    """모델 train과 test에 활용할 raw 데이터셋을 판다스 데이터프레임 타입으로 불러옵니다.

    Returns:
        List[pd.DataFrame]: 모델 train과 test에 활용될 raw 데이터셋
    """
    train_data = pd.read_csv(str(self.data_dir.joinpath(self.train_dataset_path)))
    test_data = pd.read_csv(str(self.data_dir.joinpath(self.test_dataset_path)))

    return [train_data, test_data]

  def _convert_into_dummy_coded_datasets(
      self,
      df: pd.DataFrame,
      one_hot_var_list: List[str],
      numeric_var_list: List[str]
  ) -> pd.DataFrame:
    """one hot encoding과 float type conversion을 수행합니다.

    Args:
        df (pd.DataFrame): 모델링 데이터셋
        one_hot_vars (List[str]): one hot encoding이 필요한 컬럼 리스트 (3개 이상의 범주형 변수)
        float_vars (List[str]): float로 type conversion이 필요한 컬럼 리스트 (True/False categorical, continuous 변수)

    Returns:
        pd.DataFrame: one hot encoding과 float type conversion이 적용된 모델링 데이터셋
    """
    if one_hot_var_list:
      df[one_hot_var_list] = df[one_hot_var_list].astype(object)
    if numeric_var_list:
      df[numeric_var_list] = df[numeric_var_list].astype(float)
    df = pd.get_dummies(df)

    return df

  def _reduce_num_of_categories_in_cols(
      self,
      df: pd.DataFrame,
      target_column: str,
      target_prop: float
  ) -> pd.DataFrame:
    """데이터 프레임 내 범주형 컬럼에서 카테고리의 갯수를 줄입니다.
    카테고리 카운트를 내림차순으로 정렬하여 일정수준의 카테고리만 남기고, 나머지 카테고리는 "others"로 병합합니다.
    이때 target_prop 값이 남길 카테고리의 갯수를 결정합니다 (예. target_prop=0.5 -> 전체 카테고리 갯수의 50%만 남기기).

    Args:
        df (pd.DataFrame): 모델링 데이터셋
        target_column (str): 타겟 범주형 컬럼 이름
        target_prop (float): 전체 카테고리의 갯수 대비 남길 카테고리의 갯수 비율 (예. target_prop=0.3 -> 전체 카테고리 갯수의 30%만 남기기)

    Returns:
        pd.DataFrame: 타겟 컬럼에 대해 카테고리 갯수를 줄인 모델링 데이터셋
    """
    df = df.copy()
    cat_count_ser = df[target_column].value_counts(ascending=False)
    preserved_cat_list = cat_count_ser.head(int(len(cat_count_ser) * target_prop)).index
    df[target_column] = np.where(
        df[target_column].isin(preserved_cat_list),
        df[target_column].astype(object),
        "others"
    )

    return df

  def make_ml_datasets(self) -> List[pd.DataFrame]:
    """모델 train과 test 데이터셋에서 모델링에 사용하지 않는 컬럼들을 제외하고,
    타겟 범주 변수에 대해 카테고리의 갯수를 줄이고,
    컬럼에 one hot encoding과 float type conversion을 수행하여 모델링 데이터셋을 생성합니다.

    Returns:
        List[pd.DataFrame]: 모델링에 활용될 수 있는 형태의 train과 test 데이터셋
    """
    train_data, test_data = self._load_raw_datasets()

    datasets = pd.concat([train_data.assign(train_data_yn=1), test_data.assign(train_data_yn=0)])

    if self.datasets_info["column_preprocessing_info"]["cat_num_reduction_info"]:
      for info_dict in self.datasets_info["column_preprocessing_info"]["cat_num_reduction_info"]:
        datasets = self._reduce_num_of_categories_in_cols(
            datasets,
            info_dict["column"],
            info_dict["target_remained_prop"]
        )

    datasets = datasets.drop(columns=self.datasets_info["column_preprocessing_info"]["drop_columns"])

    dummy_datasets = self._convert_into_dummy_coded_datasets(
        df=datasets,
        one_hot_var_list=self.datasets_info["column_preprocessing_info"]["one_hot_columns"],
        numeric_var_list=self.datasets_info["column_preprocessing_info"]["numeric_columns"]
    )

    train_data = dummy_datasets.query("train_data_yn == 1").drop(columns=["train_data_yn"])
    test_data = dummy_datasets.query("train_data_yn == 0").drop(columns=["train_data_yn"])

    return [train_data, test_data]

  def make_modeling_X_y_datasets(self) -> List[Union[pd.DataFrame, pd.Series]]:
    """모델링에 활용될 수 있는 형태의 train과 test 데이터셋에서
    [X_train(or X_trainval), X_test, y_train(or y_trainval), y_test] 데이터셋을 생성합니다.
    
    X: input 변수들, y: output 변수

    Returns:
        List[Union[pd.DataFrame, pd.Series]]: 모델링에 활용될 수 있는 train과 test 데이터셋의 X, y 데이터셋
    """
    trainval_datasets, test_datasets = self.make_ml_datasets()
    outcome_related_column_list = np.array(list(self.datasets_info["outcome_info"].values())).flatten().tolist()
    X_trainval = trainval_datasets.drop(columns=outcome_related_column_list)
    y_trainval = trainval_datasets.filter(
        items=self.datasets_info["outcome_info"]["target_outcome"],
        axis="columns"
    ).squeeze()
    X_test = test_datasets.drop(columns=outcome_related_column_list)
    y_test = test_datasets.filter(
        items=self.datasets_info["outcome_info"]["target_outcome"],
        axis="columns"
    ).squeeze()

    return [X_trainval, X_test, y_trainval, y_test]

  def make_current_practice_X_y_datasets(self) -> List[Union[pd.DataFrame, pd.Series]]:
    """현재 practice로 사용되고 있는 방법과 실제 outcome에 대한 X, y 데이터셋을 생성합니다.

    Returns:
        List[Union[pd.DataFrame, pd.Series]]: 모델 성능 비교에 활용될 수 있는 현재 practice X, y 데이터셋
    """
    train_data, test_data = self._load_raw_datasets()

    x_trainval = train_data[self.datasets_info["outcome_info"]["current_practice"]].squeeze()
    y_trainval = train_data[self.datasets_info["outcome_info"]["target_outcome"]].squeeze()
    x_test = test_data[self.datasets_info["outcome_info"]["current_practice"]].squeeze()
    y_test = test_data[self.datasets_info["outcome_info"]["target_outcome"]].squeeze()

    return [x_trainval, x_test, y_trainval, y_test]
