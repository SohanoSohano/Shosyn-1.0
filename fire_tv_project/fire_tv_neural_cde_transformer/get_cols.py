import csv

def get_column_names(csv_path):
    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Read only the first row (column headers)
        return headers

# 🔁 Replace with your actual path
csv_file_path = r"C:\Users\solos\OneDrive\Documents\College\Projects\Advanced Behavioural Analysis for Content Recommendation\Shosyn\fire_tv_neural_cde_transformer_instance_version\Shosyn-1.0\fire_tv_project\fire_tv_neural_cde_transformer\fire_tv_synthetic_dataset_v3_tmdb.csv"

columns = get_column_names(csv_file_path)
print("Column Names:")
for col in columns:
    print(col)
