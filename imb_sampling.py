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

def gamma_sampling(X, y, alpha=5.0 , theta=0.125):
    # 判斷少數和多數類別的標籤
    counts = np.bincount(y)
    minority_class_label = np.argmin(counts)
    majority_class_label = np.argmax(counts)

    # 判斷要生成的新樣本數量
    n_samples = counts[majority_class_label] - counts[minority_class_label]

    # 找到少數類別的樣本
    y = pd.Series(y).reset_index(drop=True)  # 將 y 轉換為 pandas Series 並重置索引
    minority_samples = X[y == minority_class_label]  # 直接使用 X

    # 初始化新樣本的數組
    new_samples = np.zeros((n_samples, X.shape[1]))

    # 使用最近鄰居算法找到每個少數類別樣本的鄰居
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)

    # 生成新樣本
    for i in range(n_samples):
        # 隨機選擇一個少數類別樣本的索引
        sample_index = np.random.randint(low=0, high=minority_samples.shape[0])

        # 使用索引來從 minority_samples 中選擇一個樣本
        sample = minority_samples[sample_index]

        # 選擇一個鄰居
        neighbor = X[indices[sample_index][1]]

        # 定義向量 v
        v = neighbor - sample

        # 使用伽馬分佈生成值 t
        t = np.random.gamma(alpha, theta, 1)[0]

        # 定義新的少數點
        new_sample = sample + t * v

        # 將新的少數點加入新樣本中
        new_samples[i] = new_sample

    # # 使用 StandardScaler 進行縮放
    # scaler = StandardScaler()
    # new_samples = scaler.fit_transform(new_samples)

    # 創建一個新的 y 數組，包含少數類別的標籤
    new_y = np.full((n_samples,), minority_class_label)

    # 將新生成的樣本、原始的少數類別樣本和原始的多數類別樣本合併
    X_resampled = np.concatenate((X, new_samples))  # Convert to numpy array
    y_resampled = np.concatenate((y, new_y))

    return X_resampled, y_resampled