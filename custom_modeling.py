# %% Import libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, r2_score
from main.utils import adjust_pred_value, get_adjusted_r2

# %% Load data
train_df = pd.read_csv("datasets/train_df3.csv").drop(columns=["PT_pct", "PT_sec"])
test_df = pd.read_csv("datasets/test_df3.csv").drop(columns=["PT_pct", "PT_sec"])

# %% Get true, practice data
y_train = train_df["use_quan"]
y_test = test_df["use_quan"]

msbos_train = train_df["msbos"]
msbos_test = test_df["msbos"]

# %% Create X dataset
X_train = train_df.drop(columns=["use_quan", "msbos"])
X_test = test_df.drop(columns=["use_quan", "msbos"])

# %%
one_hot_columns = [
    "procedure_cd_fin",
    "anesthesia_cd",
    "sex"
] # 3
numeric_columns = [
    "age",
    "ami",
    "chf",
    "pud",
    "mld",
    "diab",
    "diabwc",
    "hp",
    "rend",
    "canc",
    "msld",
    "metacanc",
    "aids",
    "score",
    "aPTT",
    "Hb",
    "Plt",
    "PT_inr",
    "heparins",
    "direct_factor_Xa_inhibitors",
    "coumarin_deriatives",
    "miscellaneous_anticoagulants",
    "direct_thrombin_inhibitors"
] # 23
# %%
datasets = pd.concat([
    X_train.assign(train_data_yn=1),
    X_test.assign(train_data_yn=0)
])
datasets[one_hot_columns] = datasets[one_hot_columns].astype(object)
datasets[numeric_columns] = datasets[numeric_columns].astype(float)
datasets = pd.get_dummies(datasets)

X_train = datasets.query("train_data_yn == 1").drop(columns=["train_data_yn"])
X_test = datasets.query("train_data_yn == 0").drop(columns=["train_data_yn"])

# %%
X_test

# %% LR
lr_model = LinearRegression()
lr_model.fit(X_train.values, y_train.values)
# %%
y_pred = adjust_pred_value(lr_model.predict(X_test.values))
print(f"lr_mse: {mean_squared_error(y_test.values, y_pred):.3f}")
print(f"lr_adj_r2: {get_adjusted_r2(y_test.values, y_pred, X_test.shape[1]):.3f}")

# %% MSBOS
print(f"msbos_mse: {mean_squared_error(y_test.values, msbos_test.values):.3f}")
print(f"msbos_r2: {r2_score(y_test.values, msbos_test.values):.3f}")

# %%
rf_model = RandomForestRegressor(random_state=0, n_jobs=-1)
rf_model.fit(X_train.values, y_train.values)

y_pred_rf = adjust_pred_value(rf_model.predict(X_test.values))
print(f"rf_mse: {mean_squared_error(y_test.values, y_pred_rf):.3f}")
print(f"rf_adj_r2: {get_adjusted_r2(y_test.values, y_pred_rf, X_test.shape[1]):.3f}")
# %%
df = pd.concat([train_df, test_df])
# %%
len(df["procedure_cd_fin"].unique())
# %%
