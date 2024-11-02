## 提升不平衡資料集之分類效能－應用不同採樣技術
探討不同採樣技術對機器學習模型在不平衡資料集上分類效能的影響，比較Random Oversampling (ROS)、Random Undersampling (RUS)、SMOTE、ADASYN和伽瑪分佈採樣 (gamma distribution sampling) 等方法，並結合Logistic Regression、Random Forest、SVM 和 KNN 等模型進行實驗。並使用 Minority Recall 作為新的評估指標，以更準確反映模型對少數類別的識別能力。

> 本專案有兩大主要探討重點
> 1. 不同採樣技術對機器學習模型在不平衡資料集上分類效能的影響 (投稿至TAAI 2024) 
> 2. 根據資料不平衡程度， 分析在不同的平衡程度下的分類性能 (本文主要說明)

### Dataset
數據來源：[imblearn.dataset](https://imbalanced-learn.org/stable/datasets/index.html) 
使用了14項不同領域的資料集，涵蓋的領域廣泛，如科技、生醫、酒精等等。不平衡程度 (Ratio) 從8.6:1至28:1不等。

### 實驗流程：
<img src="https://github.com/moonionn/sampling-method_project/blob/main/plot/flowchart.png" width="400">

### 評估指標：
比較不同採樣方法和模型組合的表現，特別關注以下評估指標：
ROC-AUC
Minority Recall（少數類別召回率）：作為新的評估指標，更準確地反映模型對少數類別的識別能力。

### 實驗結果：
根據資料集的不平衡比例將其分為低 (0-10)、中 (10-20) 和高 (20-30) 三組。我們計算每組各採樣方法相對於基準 (No Sampling) 的性能提升百分比。
1. 分類性能分析
資料集在未採樣前對於少數類別的分類Recall極低，基本上不超過50％，甚至多數落在30％左右。
但在但在經過採樣後，對於少數類別的分類Recall大幅提升，尤其是 RUS。
2. 資料可視化分析
<img src="https://github.com/moonionn/sampling-method_project/blob/main/heatmap_mean_minority_recall_square_layout_large_font.png" width="500"/>  
此圖為Minority Recall提升的百分比，有此圖可得知：

#### 從採樣方法來看
- 所有採樣方法都顯著提升了Minority Recall。  
- RUS在大多數情況下呈現最深的顏色，表示其在提高Minority Recall方面效果最為顯著。
#### 從模型來看
- LR及SVM模型在任何不平衡程度下都有優秀的分類性能，且高度越不平衡其分類性能越優秀。
- 使用 RF 及 KNN 對於提升少數類別辨識較不具優勢。    

> [!IMPORTANT]
> **分類模型的選擇較採樣方式的選擇，影像更大。**

### 結論：
本研究透過分析不同領域的資料集，探討各種採樣方法對處理不平衡資料集的影響。  
我們引入Minority Recall作為新的評估指標，以更準確地評估模型在不平衡資料集上對少數類別的識別能力。  
研究結果表明，所有採樣方法都能顯著提高模型性能，尤其在Minority Recall方面。  
對於高度不平衡的資料集，RUS在Minority Recall分類性能最好，但需要權衡丟失重要資訊的風險，伽瑪分佈採樣是個相對保守且優秀的選擇。  
值得注意的是，模型的選擇可能比採樣方法更能決定整體分類效能。因此，在處理不平衡資料時，選擇合適的模型至關重要。在實際應用中，需要根據具體需求和資料集特性來選擇適當的方法和模型組合。!
