import pandas as pd

# 读取CSV数据
df = pd.read_csv('../formatted_results.csv')

# 模型和指标列表
models = ['Logistic Regression', 'Random Forest', 'SVM', 'KNN']
metrics = ['F1', 'AUC', 'Mean Minority Recall']

# 列名映射
column_mapping = {
    'No Sampling': 'Original',
    'Undersampling': 'RUS',
    'Oversampling': 'ROS'
}

# 新的列顺序
new_order = ['Original', 'SMOTE', 'ADASYN', 'RUS', 'ROS', 'Gamma']

# 对于每个模型
for model in models:
    print(f"\n{model}")

    # 对于每个指标
    for metric in metrics:
        print(f"\n{metric}")

        # 创建一个新的DataFrame，只包含当前模型和指标
        model_metric_df = df[df['Model'] == model][['Dataset', 'Sampling', metric]]

        # 将DataFrame重塑为所需的格式
        pivot_df = model_metric_df.pivot(index='Dataset', columns='Sampling', values=metric)

        # 重命名列
        pivot_df = pivot_df.rename(columns=column_mapping)

        # 重新排序列
        pivot_df = pivot_df.reindex(columns=new_order)

        # 打印结果表格
        print(pivot_df.to_string())
        print("\n")

        # 保存到CSV
        pivot_df.to_csv(f'../forclassification/{model}_{metric}_data.csv', index=True)