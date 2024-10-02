import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 读取CSV文件
df = pd.read_csv('output_0815/car_eval_formatted_results.csv')

# 定义不平衡比例
imbalance_ratios = {
    'abalone': 9.7, 'car_eval_34': 12, 'car_eval_4': 26,
    'ecoli': 8.6, 'letter_img': 26, 'optical_digits': 9.1, 'pen_digits': 9.4,
    'satimage': 9.3, 'sick_euthyroid': 9.8, 'solar_flare_m0': 19,
    'thyroid_sick': 15, 'us_crime': 12, 'wine_quality': 26, 'yeast_me2': 28
}

# 将不平衡比例添加到数据框中
df['Imbalance Ratio'] = df['Dataset'].map(imbalance_ratios)

# 定义不平衡比例的范围
bins = [0, 10, 20, 30]
labels = ['Low (0-10)', 'Medium (10-20)', 'High (20-30)']
df['Imbalance Group'] = pd.cut(df['Imbalance Ratio'], bins=bins, labels=labels, include_lowest=True, right=False)

# 定义所有模型和采样方法
models = ['Logistic Regression', 'Random Forest', 'KNN', 'SVM']
sampling_methods = ['No Sampling', 'SMOTE', 'ADASYN', 'RUS', 'ROS', 'Gamma']
metrics = ['F1', 'AUC', 'Mean Minority Recall']

# 创建结果字典
result_dicts = {metric: {group: {model: {} for model in models} for group in labels} for metric in metrics}

# 计算平均绝对性能值
for group in labels:
    group_df = df[df['Imbalance Group'] == group]
    for model in models:
        model_df = group_df[group_df['Model'] == model]
        for method in sampling_methods:
            method_df = model_df[model_df['Sampling'] == method]
            if not method_df.empty:
                for metric in metrics:
                    result_dicts[metric][group][model][method] = method_df[metric].mean()

# 创建结果数据框
result_dfs = {}
for metric in metrics:
    result_df = pd.DataFrame(index=pd.MultiIndex.from_product([labels, models], names=['Imbalance Group', 'Model']),
                             columns=sampling_methods)
    for group in labels:
        for model in models:
            for method in sampling_methods:
                if method in result_dicts[metric][group][model]:
                    result_df.at[(group, model), method] = result_dicts[metric][group][model][method]
    result_dfs[metric] = result_df


# 格式化和打印结果表格
def format_value(x):
    if pd.isna(x):
        return "N/A"
    else:
        return f"{x:.3f}"


for metric in metrics:
    print(f"\n{metric} Absolute Performance:")
    print(result_dfs[metric].map(format_value).to_string())
    print("\n" + "=" * 50)

# 找出每个不平衡组和模型的最佳采样方法
for metric in metrics:
    print(f"\nBest sampling method for each imbalance group and model based on {metric}:")
    best_methods = result_dfs[metric].idxmax(axis=1)
    for (group, model), method in best_methods.items():
        performance = result_dfs[metric].at[(group, model), method]
        print(f"{group} - {model}: {method} ({format_value(performance)})")
    print("\n" + "=" * 50)


# 创建热力图函数
def create_heatmap(data, title, filename):
    plt.figure(figsize=(12, 8))
    clean_data = data.dropna(how='all').dropna(axis=1, how='all')

    # 确保数据是浮点型
    clean_data = clean_data.astype(float)

    # 使用 robust=True 来处理异常值
    sns.heatmap(clean_data, annot=True, cmap='YlGnBu', fmt='.3f',
                cbar_kws={'label': 'Absolute Performance'}, robust=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# 为每个指标创建热力图
for metric in metrics:
    for group in labels:
        group_data = result_dfs[metric].loc[group]
        title = f"{metric} Absolute Performance - {group} Imbalance Ratio"
        filename = f"car_heatmap_absolute_{metric.lower().replace(' ', '_')}_{group.lower().replace(' ', '_')}.png"
        create_heatmap(group_data, title, filename)

print("Absolute performance heatmaps have been generated and saved.")