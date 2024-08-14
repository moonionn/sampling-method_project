from sklearn.preprocessing import MinMaxScaler

# no_Sampling
def no_sampling(X, y):
    return X, y

# SMOTE
def smote_balance(X, y):
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

# ADASYN
def adasyn_balance(X, y):
    from imblearn.over_sampling import ADASYN
    adasyn = ADASYN(random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)
    return X_resampled, y_resampled

# RUS (Random Under Sampling)
def undersample_balance(X, y):
    from imblearn.under_sampling import RandomUnderSampler
    undersample = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = undersample.fit_resample(X, y)
    return X_resampled, y_resampled

# ROS (Random Over Sampling)
def oversample_balance(X, y):
    from imblearn.over_sampling import RandomOverSampler
    oversample = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversample.fit_resample(X, y)
    return X_resampled, y_resampled

# Gamma
import numpy as np
from scipy.special import gamma
from sklearn.neighbors import NearestNeighbors
import pandas as pd


def gamma_sampling(X, y, alpha=5.0, theta=0.125):
    # 确保 X 是 DataFrame，y 是 Series，并且它们的索引匹配
    X = pd.DataFrame(X).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    # 判断少数和多数类别的标签
    counts = y.value_counts()
    minority_class_label = counts.idxmin()
    majority_class_label = counts.idxmax()

    # 判断要生成的新样本数量
    n_samples = counts[majority_class_label] - counts[minority_class_label]

    # 找到少数类别的样本
    minority_mask = y == minority_class_label
    minority_samples = X[minority_mask]

    # 初始化新样本的列表
    new_samples = []

    # 使用最近邻居算法找到每个少数类别样本的邻居
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)

    # 生成新样本
    for _ in range(n_samples):
        # 隨機選擇一個少數類別樣本
        sample_index = np.random.choice(minority_samples.index)
        sample = minority_samples.loc[sample_index]

        # 選擇一個鄰居
        neighbor_index = indices[sample_index][1]
        neighbor = X.iloc[neighbor_index]

        # 定義向量V
        v = neighbor - sample

        # 使用伽瑪分佈生成值 t
        t = np.random.gamma(alpha, theta, 1)[0]

        # 定義新的少數點
        new_sample = sample + t * v

        # 將新的少數點加入新樣本中
        new_samples.append(new_sample)

    # 將新樣本轉換為 DataFrame
    new_samples_df = pd.DataFrame(new_samples, columns=X.columns)

    # 創建一個新的 y Series，包含少數類別的標籤
    new_y = pd.Series([minority_class_label] * n_samples)

    # 將新生成的樣本、原始的少數類別樣本和原始的多數類別樣本合併
    X_resampled = pd.concat([X, new_samples_df], ignore_index=True)
    y_resampled = pd.concat([y, new_y], ignore_index=True)

    return X_resampled, y_resampled