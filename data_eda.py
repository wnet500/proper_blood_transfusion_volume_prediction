# %% 패키지 로드 
import pandas as pd
import numpy as np

# %%
pd.set_option("display.max_columns", None)
train_df = pd.read_csv("datasets/train_df3.csv")
test_df = pd.read_csv("datasets/test_df3.csv")
# %%
train_df
# %%
test_df.columns
# %%
data_info = \
{
  "column_preprocessing_info": {
    "one_hot_columns": [
      "procedure_cd_fin",
      "anesthesia_cd",
      "sex"
    ],
    "numeric_columns": [
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
      "PT_pct",
      "PT_sec",
      "heparins",
      "direct_factor_Xa_inhibitors",
      "coumarin_deriatives",
      "miscellaneous_anticoagulants",
      "direct_thrombin_inhibitors",
      
      "use_quan",
      "msbos"
      ],
    "drop_columns": [
      "PT_pct",
      "PT_sec"
    ]
  },
  "outcome_info":{
    "target_outcome": ["use_quan"],
    "current_practice": ["msbos"]
  }
}
# %%
test_df[data_info["column_preprocessing_info"]["one_hot_columns"]].astype(object)
# %%
test_df[data_info["column_preprocessing_info"]["numeric_columns"]].astype(float)
# %%
df = pd.concat([train_df, test_df])
# %%
others_cat_list = df["procedure_cd_fin"].value_counts()[df["procedure_cd_fin"].value_counts() < 100].index
df["procedure_cd_fin"] = np.where(df["procedure_cd_fin"].isin(others_cat_list), "others", df["procedure_cd_fin"].astype(object))
df["procedure_cd_fin"].value_counts()
# %%
cat_count_ser = df["procedure_cd_fin"].value_counts()
others_cat_list = cat_count_ser.head(int(len(cat_count_ser) * 0.5)).index

df["procedure_cd_fin"] = np.where(df["procedure_cd_fin"].isin(others_cat_list), df["procedure_cd_fin"].astype(object), "others")
df["procedure_cd_fin"].value_counts()
# %%
