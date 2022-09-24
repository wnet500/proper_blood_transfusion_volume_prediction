# proper blood transfusion volume prediction
## A. Preperation
### 1. initial python setting
- install required python libraries
- see `requirements.txt`

### 2. set data and model training config
- see `main/config.py`
- check a example config `main/config_exmple.py`

### 3. upload datasets
Here, the datasets we used in the research paper can not be released for personal information protection
- upload csv datasets (model_train_set and model_test_set) into `main/datasets`   
ex. `train_df.csv`, `test_df.csv`

## B. Model param gridsearch
- model types: `ann (torch)`, `ann (sklean)`, `xgboost`, `linear regression`, `random forest`
- see `model_parm_gridsearch.py` and set search param sets
- you can set CV info (ex. folds, repeats, validation proportion)

## C. Model train with the best param based on gridsearch results
- see `model_training_with_best_param.py`

## D. Model evaluation
- see `trained_model_evaluation.py`
- eval metrics: `mse`, `r2 (adjusted)`
- eval plot: `blan altman plot`

## E. Output
- you can see the saved files
  - gridsearch results (csv files)
  - plots (model evaluation with test sets, blan altman)
  - torch models (final torch model file)
  - traditional ml models (xgboost and sklearn-based model files)

---
*** Note that all codes are executable ONLY if your own data exist
