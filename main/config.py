# dataset 정보

## column_preprocessing_info: 데이터 컬럼 전처리 정보
### drop_columns: 제외할 컬럼 리스트
### one_hot_columns: one-hot 인코딩을 수행할 컬럼 리스트
### numeric_columns: 숫자로 type 캐스팅할 컬럼 (수치형 변수, 0/1 binary 변수 포함)

### cat_num_reduction_info
#### column: 변수 내 카테고리가 많아, 카테고리 갯수를 줄일 변수
#### target_remained_prop: 전체 카테고리에서 남길 갯수 비율 ex. 0.5 -> 전체 카테고리 갯수의 50%만 남기기

## outcome_info: 모델 아웃컴 정보
### target_outcome: 모델 아웃컴 컬럼
### current_practice: 모델의 성능과 비교할 현재 프렉티스 컬럼

train_dataset_path = "" # csv
test_dataset_path = "" # csv

dataset_info = {
  "column_preprocessing_info": {
    "drop_columns": [""],
    "one_hot_columns": [""],
    "numeric_columns": [""],
    "cat_num_reduction_info": [
      {
        "column": "",
        "target_remained_prop": 1
      }
    ]
  },
  "outcome_info": {
    "target_outcome": [""],
    "current_practice": [""]
  }
}

## pytorch_lightning Trainer 설정
pl_accelerator="gpu"
pl_devices=[0]
pl_global_seed=0
data_loader_num_workwers=2

evalset_loader_batch_size=2**12
testset_loader_batch_size=2**13
