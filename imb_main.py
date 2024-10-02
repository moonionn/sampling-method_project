# main.py
import pandas as pd
from imb_preprocessing import load_data
from multiprocessing import Pool, cpu_count
from imb_kfold import cross_validation
from imb_model import logistic_regression_model, random_forest_model, svm_model, knn_model
from imb_sampling import no_sampling, oversample_balance, undersample_balance, smote_balance, adasyn_balance, gamma_sampling

# --- 參數配置 ---
K_FOLD = 5
RANDOM_STATE = 42

# --- 資料集 ---
datasets = {
    # 'wine': ('dataset/new_winequality.csv', 'quality'),
    # 'abalone': ('dataset/abalone.csv', 'target'),
    # 'abalone_19': ('dataset/abalone_19.csv', 'target'),
    'car_eval_4': ('dataset/car_eval_4.csv', 'target'),
    'car_eval_34': ('dataset/car_eval_34.csv', 'target'),
    # 'ecoli': ('dataset/ecoli.csv', 'target'),
    # 'letter_img': ('dataset/letter_img.csv', 'target'),
    # 'optical_digits': ('dataset/optical_digits.csv', 'target'),
    # 'pen_digits': ('dataset/pen_digits.csv', 'target'),
    # 'satimage': ('dataset/satimage.csv', 'target'),
    # 'sick_euthyroid': ('dataset/sick_euthyroid.csv', 'target'),
    # 'solar_flare_m0': ('dataset/solar_flare_m0.csv', 'target'),
    # 'thyroid_sick': ('dataset/thyroid_sick.csv', 'target'),
    # 'us_crime': ('dataset/us_crime.csv', 'target'),
    # 'wine_quality': ('dataset/wine_quality.csv', 'target'),
    # 'yeast_me2': ('dataset/yeast_me2.csv', 'target')
}

# --- 採樣方法和模型 ---
sampling_methods = {
    'No Sampling': no_sampling,
    'SMOTE': smote_balance,
    'ADASYN': adasyn_balance,
    'RUS': undersample_balance,
    'ROS': oversample_balance,
    'Gamma': gamma_sampling
}

models = {
    'Logistic Regression': logistic_regression_model,
    'Random Forest': random_forest_model,
    'SVM': svm_model,
    'KNN': knn_model
}

# --- 儲存结果 ---
results = {}

# --- 平行處理 ---
def process_task(args):
    dataset_name, file_path, target_column, model_name, model_func, sampling_name, sampling_func = args
    print(f"\n--- Processing dataset: {dataset_name}, Model: {model_name}, Sampling method: {sampling_name} ---")
    df = pd.read_csv(file_path)
    X, y = load_data(df, target_column)
    f1, auc, mean_minority_recall = cross_validation(X, y, model_func, k=K_FOLD, balance_function=sampling_func)
    return (dataset_name, model_name, sampling_name), (f1, auc, mean_minority_recall)

# --- 建立任務列表 ---
tasks = [(dataset_name, file_path, target_column, model_name, model_func, sampling_name, sampling_func)
         for dataset_name, (file_path, target_column) in datasets.items()
         for model_name, model_func in models.items()
         for sampling_name, sampling_func in sampling_methods.items()]

if __name__ == '__main__':
    # --- 建立進程池並執行任務 ---
    with Pool(cpu_count()) as p:
        results = dict(p.map(process_task, tasks))

    # --- 結果 ---
    print("\n--- Results ---")
    for (dataset, model, sampling), (f1, auc, mean_minority_recall) in results.items():
        print(f"{dataset} - {model} ({sampling}): F1 = {f1:.4f}, AUC = {auc:.4f}, Mean Minority Recall = {mean_minority_recall:.4f}")
    # 將結果轉換為 DataFrame
    df = pd.DataFrame.from_dict(results, orient='index', columns=['F1', 'AUC', 'Mean Minority Recall'])

    # 將 DataFrame 寫入 CSV 文件 # change
    df.to_csv('output_0815/car_eval_results.csv')

# # --- 循環遍歷資料集、採樣方法和模型 ---
# for dataset_name, (file_path, target_column) in datasets.items():
#     print(f"\n--- Processing dataset: {dataset_name} ---")
#     df = pd.read_csv(file_path)
#     X, y = load_data(df, target_column)
#
#     for model_name, model_func in models.items():
#         print(f"    * Model: {model_name} ")
#         for sampling_name, sampling_func in sampling_methods.items():
#             print(f"      - Sampling method: {sampling_name}")
#             f1, auc, mean_minority_recall = cross_validation(X, y, model_func, k=K_FOLD,
#                                                                   balance_function=sampling_func)
#             results[(dataset_name, model_name, sampling_name)] = (f1, auc, mean_minority_recall)
#
# # --- 結果 ---
# print("\n--- Results ---")
# for (dataset, model, sampling), (f1, auc, mean_minority_recall) in results.items():
#     print(f"{dataset} - {model} ({sampling}): F1 = {f1:.4f}, AUC = {auc:.4f}, Mean Minority Recall = {mean_minority_recall:.4f}")