import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_imbalance_ratio_impact(df, imbalance_ratios):
    # 将不平衡比例添加到数据框中
    df['Imbalance Ratio'] = df['Dataset'].map(imbalance_ratios)

    # 定义不平衡比例的范围
    bins = [0, 10, 20, 50, float('inf')]
    labels = ['Low (<=10)', 'Medium (10-20)', 'High (20-50)', 'Very High (>50)']
    df['Imbalance Group'] = pd.cut(df['Imbalance Ratio'], bins=bins, labels=labels, include_lowest=True)

    metrics = ['F1', 'AUC', 'Mean Minority Recall']
    sampling_methods = df['Sampling'].unique()

    results = {}
    for metric in metrics:
        # 计算每个不平衡组和采样方法的平均性能
        grouped = df.groupby(['Imbalance Group', 'Sampling'])[metric].mean().unstack()
        results[metric] = grouped

        # 创建热图
        plt.figure(figsize=(12, 8))
        sns.heatmap(grouped, annot=True, cmap='YlGnBu', fmt='.3f')
        plt.title(f'Average {metric} for Different Imbalance Ratios and Sampling Methods')
        plt.tight_layout()
        plt.savefig(f'heatmap_{metric}_imbalance_impact.png')
        plt.close()

        # 打印表格结果
        print(f"\nAverage {metric} for Different Imbalance Ratios and Sampling Methods:")
        print(grouped.to_string())
        print("\n" + "="*80)

    return results

# 假设我们有以下数据
df = pd.read_csv('output_0815_ratio/ratio_formatted_results.csv')
imbalance_ratios = {
    'abalone': 9.7, 'car_eval_34': 12, 'car_eval_4': 26,
    'ecoli': 8.6, 'letter_img': 26, 'optical_digits': 9.1, 'pen_digits': 9.4,
    'satimage': 9.3, 'sick_euthyroid': 9.8, 'solar_flare_m0': 19,
    'thyroid_sick': 15, 'us_crime': 12, 'wine_quality': 26, 'yeast_me2': 28
}

# 运行分析
results = analyze_imbalance_ratio_impact(df, imbalance_ratios)

# 输出总结
print("\nSummary of Sampling Methods Impact:")
for metric, result in results.items():
    print(f"\n{metric}:")
    best_method = result.idxmax()
    print("Best performing sampling method for each imbalance group:")
    print(best_method.to_string())
    print("\nPerformance improvement over No Sampling:")
    improvement = (result.subtract(result['No Sampling'], axis=0).div(result['No Sampling'], axis=0) * 100).round(2)
    print(improvement.to_string())
    print("\n" + "="*80)