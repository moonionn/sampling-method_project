import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

# 读取CSV文件
data = pd.read_csv('../output_0815_ratio/ratio_formatted_results.csv')

# 检查DataFrame的列名
print(data.columns)

# 确保列名与绘图代码中的列名一致
data.rename(columns={'Sampling': 'Method'}, inplace=True)

# 准备雷达图数据
categories = ['F1', 'AUC', 'Mean Minority Recall']
num_categories = len(categories)

# 计算平均值
avg_values = data.groupby('Method')[categories].mean()

# 创建雷达图
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
angles = [n / float(num_categories) * 2 * pi for n in range(num_categories)]
angles += angles[:1]

# 初始化雷达图
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# 设置标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

# 绘制每种方法的雷达图
for method in avg_values.index:
    values = avg_values.loc[method].tolist()
    values += values[:1]  # 将第一个值附加到末尾，使雷达图闭合
    label = f"{method} (F1: {values[0]:.2f}, AUC: {values[1]:.2f}, Mean Minority Recall: {values[2]:.2f})"
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=label)
    ax.fill(angles, values, alpha=0.1)  # 使用透明度区分，不用颜色

# 添加标题和图例
plt.title('Average Performance of Different Sampling Methods')
plt.legend(loc='lower right')
plt.savefig('../output_0815_ratio/ratio_radar_chart.png')
plt.show()

# 设置画布大小
plt.figure(figsize=(21, 8))

# 创建F1 Score的折线图
plt.subplot(1, 3, 1)
sns.lineplot(x="Method", y="F1", hue="Model", marker="o", data=data)
plt.title('F1 Score Trends for Different Sampling Methods and Models')
plt.xlabel('Sampling Method')
plt.ylabel('F1 Score')
plt.legend(title='Model')
plt.xticks(rotation=45)

# 创建AUC的折线图
plt.subplot(1, 3, 2)
sns.lineplot(x="Method", y="AUC", hue="Model", marker="o", data=data)
plt.title('AUC Trends for Different Sampling Methods and Models')
plt.xlabel('Sampling Method')
plt.ylabel('AUC')
plt.legend(title='Model')
plt.xticks(rotation=45)

# 创建Minority Recall的折线图
plt.subplot(1, 3, 3)
sns.lineplot(x="Method", y="Mean Minority Recall", hue="Model", marker="o", data=data)
plt.title('Minority Recall Trends for Different Sampling Methods and Models')
plt.xlabel('Sampling Method')
plt.ylabel('Minority Recall')
plt.legend(title='Model')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('../output_0815_ratio/ratio_eval_line_plots.png')
plt.show()

