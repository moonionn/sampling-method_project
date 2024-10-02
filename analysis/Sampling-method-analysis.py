import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('../output_0815_ratio/ratio_formatted_results.csv')

# 定义所有模型和采样方法
models = ['Logistic Regression', 'Random Forest', 'KNN', 'SVM']
sampling_methods = ['SMOTE', 'ADASYN', 'RUS', 'ROS', 'Gamma']
metrics = ['F1', 'AUC', 'Mean Minority Recall']

# 创建一个函数来计算改进百分比
def calculate_improvement(baseline, sampled):
    if baseline == 0:
        return sampled * 100  # 返回绝对增加值的百分比
    return (sampled - baseline) / baseline * 100

# 为每个指标创建一个结果字典，用于存储所有改进百分比
result_dicts = {metric: {model: {method: [] for method in sampling_methods} for model in models} for metric in metrics}

# 遍历每个数据集和模型
for dataset in df['Dataset'].unique():
    dataset_df = df[df['Dataset'] == dataset]

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
                result_dicts[metric][model][method].append(improvement)

# 定义去除极端值的函数
def remove_outliers(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return [x for x in data if lower_bound <= x <= upper_bound]

# 对每个指标的结果应用极端值去除，然后计算平均值
result_dfs = {}
for metric in metrics:
    result_df = pd.DataFrame(index=models, columns=sampling_methods)
    for model in models:
        for method in sampling_methods:
            data = result_dicts[metric][model][method]
            if data:
                cleaned_data = remove_outliers(data)
                result_df.at[model, method] = np.mean(cleaned_data) if cleaned_data else np.nan
            else:
                result_df.at[model, method] = np.nan
    result_dfs[metric] = result_df

# 格式化和打印结果表格
def format_value(x):
    if pd.isna(x):
        return "N/A"
    else:
        return f"{x:.2f}%"

for metric in metrics:
    print(f"\n{metric} Improvement (%) after removing outliers:")
    print(result_dfs[metric].map(format_value).to_string())
    print("\n" + "=" * 50)

# 找出每个模型的最佳采样方法
for metric in metrics:
    print(f"\nBest sampling method for each model based on {metric}:")
    best_methods = result_dfs[metric].idxmax(axis=1)
    for model, method in best_methods.items():
        improvement = result_dfs[metric].at[model, method]
        print(f"{model}: {method} ({format_value(improvement)})")
    print("\n" + "=" * 50)