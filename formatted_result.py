import pandas as pd

# 讀取原始數據
final_result = pd.read_csv('output_0815/car_eval_results.csv')

# 定義一個函數來四捨五入到小數點後三位
def round_to_three(x):
    return f"{float(x):.3f}"

# 應用四捨五入函數到 F1, AUC, 和 Mean Minority Recall 列
columns_to_format = ['F1', 'AUC', 'Mean Minority Recall']
for col in columns_to_format:
    final_result[col] = final_result[col].apply(round_to_three)

# 獲取第一列的名稱（可能是未命名的）
first_column = final_result.columns[0]

# 拆分第一列
final_result[['Dataset', 'Model', 'Sampling']] = final_result[first_column].str.extract(r"\('(.+)', '(.+)', '(.+)'\)")

# 刪除原始的第一列
final_result = final_result.drop(first_column, axis=1)

# 重新排列列的順序
final_result = final_result[['Dataset', 'Model', 'Sampling', 'F1', 'AUC', 'Mean Minority Recall']]

# 保存結果
final_result.to_csv('output_0815/car_eval_formatted_results.csv', index=False)