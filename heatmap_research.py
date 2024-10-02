import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 读取CSV文件
df = pd.read_csv('formatted_results.csv')

# 定义所有模型和采样方法
models = ['Logistic Regression', 'Random Forest', 'KNN', 'SVM']
sampling_methods = ['No Sampling', 'SMOTE', 'ADASYN', 'Undersampling', 'Oversampling', 'Gamma']
metrics = ['F1', 'AUC', 'Mean Minority Recall']

# 更新后的不平衡比例
imbalance_ratios = {
    'abalone': 9.7,
    'abalone_19': 130,
    'car_eval_34': 12,
    'car_eval_4': 26,
    'ecoli': 8.6,
    'letter_img': 26,
    'optical_digits': 9.1,
    'pen_digits': 9.4,
    'satimage': 9.3,
    'sick_euthyroid': 9.8,
    'solar_flare_m0': 19,
    'thyroid_sick': 15,
    'us_crime': 12,
    'wine_quality': 26,
    'yeast_me2': 28
}


# 为每个数据集创建热图
def create_heatmap(dataset_name, metric):
    dataset_df = df[df['Dataset'] == dataset_name]
    pivot_table = dataset_df.pivot(index='Model', columns='Sampling', values=metric)
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='.3f')
    plt.title(f'{metric} for {dataset_name} (Imbalance Ratio: {imbalance_ratios[dataset_name]}:1)')
    plt.tight_layout()
    plt.savefig(f'heatmap_{dataset_name}_{metric}.png')
    plt.close()


# 分析不平衡程度对性能的影响
def analyze_imbalance_impact(metric):
    plt.figure(figsize=(12, 8))
    for sampling in sampling_methods:
        x = []
        y = []
        for dataset in df['Dataset'].unique():
            if dataset in imbalance_ratios:
                ratio = imbalance_ratios[dataset]
                performance = df[(df['Dataset'] == dataset) & (df['Sampling'] == sampling)][metric].mean()
                x.append(ratio)
                y.append(performance)
        plt.scatter(x, y, label=sampling)

    plt.xlabel('Imbalance Ratio')
    plt.ylabel(metric)
    plt.title(f'Impact of Imbalance Ratio on {metric}')
    plt.legend()
    plt.xscale('log')  # 使用对数刻度
    plt.savefig(f'imbalance_impact_{metric}.png')
    plt.close()


# 数据集验证
datasets_in_df = set(df['Dataset'].unique())
datasets_in_ratios = set(imbalance_ratios.keys())

if datasets_in_df != datasets_in_ratios:
    print("警告：数据集不匹配")
    print("在DataFrame中但不在比例字典中:", datasets_in_df - datasets_in_ratios)
    print("在比例字典中但不在DataFrame中:", datasets_in_ratios - datasets_in_df)

# 执行分析
for dataset in df['Dataset'].unique():
    if dataset in imbalance_ratios:
        for metric in metrics:
            create_heatmap(dataset, metric)
    else:
        print(f"警告：数据集 {dataset} 没有对应的不平衡比例")

for metric in metrics:
    analyze_imbalance_impact(metric)

print("分析完成。请检查生成的图表文件。")