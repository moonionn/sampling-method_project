import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('../formatted_results.csv')

# 定义所有模型和采样方法
models = ['Logistic Regression', 'Random Forest', 'KNN', 'SVM']
sampling_methods = ['SMOTE', 'ADASYN', 'Undersampling', 'Oversampling', 'Gamma']
metrics = ['F1', 'AUC', 'Mean Minority Recall']

# 创建一个函数来计算改进百分比
def calculate_improvement(baseline, sampled):
    if baseline == 0:
        return sampled * 100  # 返回绝对增加值的百分比
    return (sampled - baseline) / baseline * 100

# 为每个指标创建一个结果DataFrame
result_dfs = {metric: pd.DataFrame(index=models, columns=sampling_methods) for metric in metrics}

# 遍历每个数据集和模型
for dataset in df['Dataset'].unique():
    dataset_df = df[df['Dataset'] == dataset]

    for model in models:
        model_df = dataset_df[dataset_df['Model'] == model]
        baseline = model_df[model_df['Sampling'] == 'No Sampling'].iloc[0]

        # 计算每种采样方法的改进
        for method in sampling_methods:
            sampled = model_df[model_df['Sampling'] == method].iloc[0]

            for metric in metrics:
                improvement = calculate_improvement(baseline[metric], sampled[metric])

                # 将改进添加到相应的DataFrame中
                current_value = result_dfs[metric].at[model, method]
                if isinstance(current_value, list):
                    current_value.append(improvement)
                elif pd.isna(current_value):
                    result_dfs[metric].at[model, method] = [improvement]
                else:
                    result_dfs[metric].at[model, method] = [current_value, improvement]

# 计算平均改进百分比
for metric in metrics:
    result_dfs[metric] = result_dfs[metric].apply(lambda x: x.apply(lambda y: np.mean(y) if isinstance(y, list) else y))

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
    print(f"\n{metric} Improvement (%)")
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