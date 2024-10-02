## 提升不平衡資料集之分類效能－應用不同採樣技術
探討不同採樣技術對機器學習模型在不平衡資料集上分類效能的影響，比較Random Oversampling、Random Undersampling、SMOTE、ADASYN和伽瑪分佈採樣 (gamma distribution sampling) 等方法，並結合Logistic Regression、Random Forest、SVM 和 KNN 等模型進行實驗。並使用 Minority Recall 作為新的評估指標，以更準確反映模型對少數類別的識別能力。

本專案有兩大主要探討重點
1. 不同採樣技術對機器學習模型在不平衡資料集上分類效能的影響
2. 根據資料不平衡程度， 分析在不同的平衡程度下的分類性能

Dataset
---
[imblearn.dataset](https://imbalanced-learn.org/stable/datasets/index.html) 中截取其中14項資料集

不同採樣技術對機器學習模型在不平衡資料集上分類效能的影響
---
#### 流程：
先將資料清洗移除不需要的資料，接著進行採樣、縮放、最後預測，並且採5折交叉驗證，在每一次的迴圈中，只會對訓練集進行採樣，以確保測試集不被污染。
![截圖 2024-10-03 凌晨3 17 09](https://github.com/user-attachments/assets/2d4a5f46-9f35-4075-ae8a-1bbba52bd5b1) 

<div align="center">
	<img src="[./raw-docs/img/editor.png](https://github.com/user-attachments/assets/2d4a5f46-9f35-4075-ae8a-1bbba52bd5b1)" alt="Editor" width="500">
</div>
