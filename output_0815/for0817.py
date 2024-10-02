import pandas as pd
from io import StringIO

# 读取CSV数据
df = pd.read_csv('car_eval_formatted_results.csv')

# 计算每个Model和Sampling组合的AUC和Minority Recall平均值
auc_df = df.groupby(['Model', 'Sampling'])['AUC'].mean().unstack()
minority_recall_df = df.groupby(['Model', 'Sampling'])['Mean Minority Recall'].mean().unstack()

# 重新排序列，使得"No Sampling"在第一列
column_order = ['No Sampling', 'SMOTE', 'ADASYN', 'RUS', 'ROS', 'Gamma']
auc_df = auc_df[column_order]
minority_recall_df = minority_recall_df[column_order]

# 重命名索引，使其更简洁
auc_df.index = auc_df.index.str.replace('Logistic Regression', 'LR')
minority_recall_df.index = minority_recall_df.index.str.replace('Logistic Regression', 'LR')

# 打印结果
print("AUC DataFrame:")
print(auc_df)
print("\nMinority Recall DataFrame:")
print(minority_recall_df)