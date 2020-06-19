# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
# import lightgbm as lgb
from matplotlib_venn import venn2
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder

train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data.csv')

train_y = train_df['y']
del train_df["y"]

df = pd.concat([train_df, test_df])

# ### preprocessing

# channelId
le_channelId = LabelEncoder()
df['channelId'] = le_channelId.fit_transform(df['channelId'])

# drop id , video_id  and thumbnail_link
del df['id']
del df['video_id']
del df['thumbnail_link']

# publishedAt
df["publishedAt"] = pd.to_datetime(df["publishedAt"])
df["year"] = df["publishedAt"].dt.year
df["month"] = df["publishedAt"].dt.month
df["day"] = df["publishedAt"].dt.day
df["hour"] = df["publishedAt"].dt.hour
df["minute"] = df["publishedAt"].dt.minute
del df["publishedAt"]

# collection_date
df["collection_date"] = "20" + df["collection_date"]
df["collection_date"] = pd.to_datetime(df["collection_date"], format="%Y.%d.%m")
df["c_year"] = df["collection_date"].dt.year
df["c_month"] = df["collection_date"].dt.month
df["c_day"] = df["collection_date"].dt.day
del df["collection_date"]

# evaluation
df['evaluation'] = df['likes'] + df['dislikes']

# 言語処理はあとで行う
df = df.drop(["title","channelTitle","tags","description"], axis=1)

df.head(2)

df.info()

# データの分割
train = df.iloc[:len(train_y), :]
test = df.iloc[len(train_y):, :]

# kfold
cv = KFold(n_splits=5, shuffle=True, random_state=10)
params = {
    'boosting_type': 'gbdt',
    'metric': 'rmse',
    'objective': 'regression',
    'seed': 20,
    'learning_rate': 0.01,
    "n_jobs": -1,
    "verbose": -1
}

y = np.log(train_y)

# ###  training

for tr_idx, val_idx in cv.split(train):
    x_train, x_val = train.iloc[tr_idx], train.iloc[val_idx]
    y_train, y_val = y[tr_idx], y[val_idx]
    
    # Datasetに入れて学習させる
    train_set = lgb.Dataset(x_train, y_train)
    val_set = lgb.Dataset(x_val, y_val, reference=train_set)
        
    # Training
    model = lgb.train(params, train_set, num_boost_round=24000, early_stopping_rounds=100,
                      valid_sets=[train_set, val_set], verbose_eval=500)
    
    # 予測したらexpで元に戻す
    test_pred = np.exp(model.predict(test))
    # 0より小さな値があるとエラーになるので補正
    test_pred = np.where(test_pred < 0, 0, test_pred)
    pred += test_pred / 5  #  5fold回すので

    oof = np.exp(model.predict(x_val))
    oof = np.where(oof < 0, 0, oof)
    rmsle = np.sqrt(mean_squared_log_error(np.exp(y_val), oof))
    print(f"RMSLE : {rmsle}")
    score += model.best_score["valid_1"]["rmse"] / 5

lgb.plot_importance(model, importance_type="gain", max_num_features=20)

print(f"Mean RMSLE SCORE :{score}")

submit_df = pd.DataFrame({"y": pred})
submit_df.index.name = "id"
submit_df.to_csv("submit.csv")
