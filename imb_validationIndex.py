# miniority_class_accuracy: 計算少數類別的準確率。
import numpy as np
def minority_class_accuracy(y_true, y_pred):
    """
    計算少數類別的準確率。

    參數：
    - y_true: 真實標籤。
    - y_pred: 預測標籤。
    """
    # 找出少數類別
    classes, counts = np.unique(y_true, return_counts=True)
    minority_class = classes[np.argmin(counts)]

    # 計算少數類別預測正確的數量
    true_positives = np.sum((y_true == minority_class) & (y_pred == minority_class))

    # 計算少數類別總數
    total_minority_samples = np.sum(y_true == minority_class)

    # 計算少數類別召回率
    minority_recall = true_positives / total_minority_samples

    return minority_recall