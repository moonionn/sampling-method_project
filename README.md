## 提升不平衡資料集之分類效能－應用不同採樣技術
探討不同採樣技術對機器學習模型在不平衡資料集上分類效能的影響，比較Random Oversampling、Random Undersampling、SMOTE、ADASYN和伽瑪分佈採樣 (gamma distribution sampling) 等方法，並結合Logistic Regression、Random Forest、SVM 和 KNN 等模型進行實驗。並使用 Minority Recall 作為新的評估指標，以更準確反映模型對少數類別的識別能力。

本專案有兩大主要探討重點
1. 不同採樣技術對機器學習模型在不平衡資料集上分類效能的影響
2. 根據資料不平衡程度， 分析在不同的平衡程度下的分類性能

Dataset
---
數據來源：[imblearn.dataset](https://imbalanced-learn.org/stable/datasets/index.html) 
使用了14項不同領域的資料集

不同採樣技術對機器學習模型在不平衡資料集上分類效能的影響
---
### 實驗過流程：
資料預處理: 清洗數據,進行特徵縮放
採樣: 僅對訓練集進行採樣
模型訓練與評估: 使用5折交叉驗證 (在每一回僅對訓練集做採樣，確保測試及不被污染)
結果分析: 比較不同採樣方法和模型組合的表現

### 實驗過流程：
