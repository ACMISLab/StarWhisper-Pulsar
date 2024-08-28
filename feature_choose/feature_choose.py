#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/7/1 14:53
# @Author : 桐
# @QQ:1041264242
# 注意事项：
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import Ridge
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from boruta import BorutaPy

# 假设你的数据是DataFrame格式，并且列名为'feature_group1', 'feature_group2', 'feature_group3', 'feature_group4', 'target'
# 读取数据
data = pd.read_csv(r'E:\data_sets\PICS-ResNet_data\test\HTRU\result/HTRU_PULSAR_Thornton_2.csv')

# 分离特征和目标变量
X = data[['feature_group1', 'feature_group2', 'feature_group3', 'feature_group4']]
y = data['target']
feature_names = ['feature_group1', 'feature_group2', 'feature_group3', 'feature_group4']
X = pd.DataFrame(X, columns=feature_names)

# 6. 随机森林特征重要性
rf = RandomForestClassifier()
rf.fit(X, y)
rf_importance = pd.Series(rf.feature_importances_, index=feature_names)
print("\n随机森林特征重要性:")
print(rf_importance)

# 7. L1正则化的逻辑回归
log_reg = LogisticRegression(penalty='l1', solver='liblinear')
log_reg.fit(X, y)
log_reg_importance = pd.Series(log_reg.coef_[0], index=feature_names)
print("\nL1正则化的逻辑回归特征重要性:")
print(log_reg_importance)

# 9. 支持向量机（SVM）
svm = SVC(kernel='linear')
svm.fit(X, y)
svm_importance = pd.Series(svm.coef_[0], index=feature_names)
print("\n支持向量机（SVM）特征重要性:")
print(svm_importance)

# 11. LightGBM
lgbm = LGBMClassifier()
lgbm.fit(X, y)
lgbm_importance = pd.Series(lgbm.feature_importances_, index=feature_names)
print("\nLightGBM特征重要性:")
print(lgbm_importance)

# 12. CatBoost
catboost = CatBoostClassifier(verbose=0)
catboost.fit(X, y)
catboost_importance = pd.Series(catboost.feature_importances_, index=feature_names)
print("\nCatBoost特征重要性:")
print(catboost_importance)

# 13. 岭回归（L2正则化）
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)
ridge_importance = pd.Series(ridge.coef_, index=feature_names)
print("\n岭回归（L2正则化）特征重要性:")
print(ridge_importance)