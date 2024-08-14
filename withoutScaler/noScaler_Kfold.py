from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from imb_validationIndex import minority_class_accuracy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np


def out_of_Scaler_cross_validation(data, labels, model_func, k=5, balance_function=None, **kwargs):
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    f1_scores = []
    aucs = []
    minority_accuracies = []

    # 進行 k 次分割
    for train_index, test_index in kf.split(data, labels):
        # 根據分割的索引獲取訓練集和測試集
        X_train = pd.DataFrame(data.iloc[train_index], columns=data.columns)
        X_test = pd.DataFrame(data.iloc[test_index], columns=data.columns)

        # 將標籤轉換為 Pandas Series
        labels = pd.Series(labels)
        # 根據分割的索引獲取訓練集和測試集的標籤
        y_train, y_test = labels.iloc[train_index].reset_index(drop=True), labels.iloc[test_index].reset_index(drop=True)

        # 如果提供了平衡函數，則對訓練集進行平衡處理
        if balance_function is not None:
            X_train, y_train = balance_function(X_train, y_train)

        # 移除了MinMaxScaler相關代碼

        # 使用提供的模型函數訓練模型
        model = model_func(X_train, y_train, **kwargs)
        # 進行預測
        y_pred = model.predict(X_test)
        # 獲取預測概率
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # 計算評估指標
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else np.nan
        minority_acc = minority_class_accuracy(y_test, y_pred)

        # 將評估指標添加到對應的列表中
        f1_scores.append(f1)
        aucs.append(auc)
        minority_accuracies.append(minority_acc)

    # 計算平均評估指標
    mean_f1_score = sum(f1_scores) / len(f1_scores)
    mean_auc = sum(aucs) / len(aucs)
    mean_minority_accuracy = sum(minority_accuracies) / len(minority_accuracies)

    # 返回平均評估指標
    return mean_f1_score, mean_auc, mean_minority_accuracy