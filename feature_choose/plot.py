#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/7/1 20:58
# @Author : 桐
# @QQ:1041264242
# 注意事项：
# 画图显示所有特征重要性评分



# LogisticRegression=[0.000065,-0.000080,-2.064417,0.001587]
# RidgeRegression=[0.000009,0.000013,-0.035877,0.000030]
# RandomForest=[0.108301,0.471284,0.200271,0.220143]
# SVM=[0.017977,0.018257,-11.227060,0.048726]
# LightGBM=[937,717,96,1250]
# CatBoost=[25.107600,27.059708,14.389658,33.443034]
# 数据
import matplotlib.pyplot as plt
import numpy as np

# 数据
features = ['Best Period', 'Best DM', 'Best SNR', 'Pulse Width']
LogisticRegression = [0.000065, -0.000080, -2.064417, 0.001587]
RidgeRegression = [0.000009, 0.000013, -0.035877, 0.000030]
RandomForest = [0.108301, 0.471284, 0.200271, 0.220143]
SVM = [0.017977, 0.018257, -11.227060, 0.048726]
LightGBM = [937, 717, 96, 1250]
CatBoost = [25.107600, 27.059708, 14.389658, 33.443034]

data = [LogisticRegression, RidgeRegression, RandomForest, SVM, LightGBM, CatBoost]
titles = ['Logistic Regression', 'Ridge Regression', 'Random Forest', 'SVM', 'LightGBM', 'CatBoost']

# 创建图形
fig, axs = plt.subplots(3, 2, figsize=(8, 10))
fig.subplots_adjust(hspace=0.6, wspace=0.4)

# 画图
for i, ax in enumerate(axs.flat):
    values = data[i]
    colors = ['blue' if v > 0 else 'red' for v in values]
    ax.bar(features, values, color=colors)
    ax.set_title(titles[i])

    y_min = min(values) - abs(min(values)) * 0.1  # 设置 y 轴最小值
    y_max = max(values) + abs(max(values)) * 0.1  # 设置 y 轴最大值

    if min(values) < -y_max * 0.1:  # 如果负值特别低，则缩小负值的 y 轴范围
        y_min = -y_max * 0.2

    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Feature',fontsize=14)
    ax.set_ylabel('Importance of features', fontsize=12)
    ax.xaxis.set_tick_params(labelsize=10, labelrotation=20)
# 保存为 PDF
plt.savefig('feature_importance_plots.pdf', format='pdf')

plt.show()
