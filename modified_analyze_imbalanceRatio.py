import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('output_0815_ratio/ratio_formatted_results.csv')

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
sampling_methods = ['SMOTE', 'ADASYN', 'RUS', 'ROS', 'Gamma']
metrics = ['F1', 'AUC', 'Mean Minority Recall']

# 创建一个函数来计算改进百分比
def calculate_improvement(baseline, sampled):
    if baseline == 0:
        return sampled * 100  # 返回绝对增加值的百分比
    return (sampled - baseline) / baseline * 100

# 定义去除极端值的函数
def remove_outliers(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return [x for x in data if lower_bound <= x <= upper_bound]

# 为每个指标创建一个结果字典
result_dicts = {metric: {group: {model: {method: [] for method in sampling_methods} for model in models} for group in labels} for metric in metrics}

# 遍历每个数据集、不平衡组和模型
for group in labels:
    group_df = df[df['Imbalance Group'] == group]
    for dataset in group_df['Dataset'].unique():
        dataset_df = group_df[group_df['Dataset'] == dataset]
        for model in models:
            model_df = dataset_df[dataset_df['Model'] == model]
            baseline_df = model_df[model_df['Sampling'] == 'No Sampling']
            if baseline_df.empty:
                continue  # Skip if no baseline data
            baseline = baseline_df.iloc[0]

            # 计算每种采样方法的改进
            for method in sampling_methods:
                sampled_df = model_df[model_df['Sampling'] == method]
                if sampled_df.empty:
                    continue  # Skip if no sampled data
                sampled = sampled_df.iloc[0]

                for metric in metrics:
                    improvement = calculate_improvement(baseline[metric], sampled[metric])
                    result_dicts[metric][group][model][method].append(improvement)

# 去除极端值并计算平均改进百分比
result_dfs = {}
for metric in metrics:
    result_df = pd.DataFrame(index=pd.MultiIndex.from_product([labels, models], names=['Imbalance Group', 'Model']),
                             columns=sampling_methods)
    for group in labels:
        for model in models:
            for method in sampling_methods:
                improvements = result_dicts[metric][group][model][method]
                if improvements:
                    cleaned_improvements = remove_outliers(improvements)
                    if cleaned_improvements:
                        result_df.at[(group, model), method] = np.mean(cleaned_improvements)
    result_dfs[metric] = result_df

# 格式化和打印结果表格
def format_value(x):
    if pd.isna(x):
        return "N/A"
    elif x > 1000:
        return "Large Improvement"
    elif x < -1000:
        return "Large Decrease"
    else:
        return f"{x:.2f}%"

for metric in metrics:
    print(f"\n{metric} Improvement (%) after removing outliers:")
    print(result_dfs[metric].map(format_value).to_string())
    print("\n" + "=" * 50)

# 找出每个不平衡组和模型的最佳采样方法
for metric in metrics:
    print(f"\nBest sampling method for each imbalance group and model based on {metric}:")
    best_methods = result_dfs[metric].idxmax(axis=1)
    for (group, model), method in best_methods.items():
        improvement = result_dfs[metric].at[(group, model), method]
        print(f"{group} - {model}: {method} ({format_value(improvement)})")
    print("\n" + "=" * 50)

