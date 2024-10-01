import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from imblearn.over_sampling import SMOTE
import numpy as np

# Đọc dữ liệu từ file CSV và chuẩn bị tập huấn luyện, kiểm tra, xác thực
file_path = './mynewdata.csv'  # Thay đổi đường dẫn tới tệp của bạn
data = pd.read_csv(file_path)
data.drop([0, 1, 2], inplace=True)  # xóa 3 mẫu cho tròn 300 mẫu dữ liệu

# Bỏ cột 'STT' và 'target' khỏi tập đặc trưng X
X = data.drop(columns=['STT', 'target']).values
y = data['target'].values

# Chia dữ liệu thành các tập huấn luyện (70%), kiểm tra (15%) và xác thực (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Áp dụng SMOTE để xử lý mất cân bằng dữ liệu
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Chuẩn hóa dữ liệu bằng StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)  # Sử dụng dữ liệu đã được resample
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Khởi tạo mô hình Logistic Regression
model = LogisticRegression(max_iter=1000, random_state=42)

# Định nghĩa các tham số cần tìm kiếm
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Tham số điều chỉnh độ phức tạp
    'solver': ['lbfgs', 'liblinear'],  # Các phương pháp tối ưu
    'penalty': ['l2', 'none'],  # Các phương pháp regularization
    'max_iter': [100, 500, 1000] 
}

# Sử dụng GridSearchCV để tìm tham số tối ưu
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train_scaled, y_train_resampled)

# In tham số tối ưu
print("Tham số tối ưu:", grid_search.best_params_)
print("Điểm độ chính xác tối ưu:", grid_search.best_score_)

# Sử dụng mô hình với tham số tối ưu
best_model = grid_search.best_estimator_

# Dự đoán trên tập xác thực
y_val_pred = best_model.predict(X_val_scaled)

# Tính độ chính xác trên tập xác thực
accuracy_val = accuracy_score(y_val, y_val_pred)
print(f'Dộ chính xác trên tập xác thực: {accuracy_val:.10f}')

# Dự đoán trên tập kiểm tra
y_test_pred = best_model.predict(X_test_scaled)
y_test_proba = best_model.predict_proba(X_test_scaled)[:, 1]  # Lấy xác suất cho lớp 1

# Tính độ chính xác trên tập kiểm tra
accuracy_test = accuracy_score(y_test, y_test_pred)
print(f'Dộ chính xác trên tập kiểm tra: {accuracy_test:.10f}')

# Ma trận nhầm lẫn
conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Ma trận nhầm lẫn:\n", conf_matrix)

# Báo cáo phân loại trên tập kiểm tra
class_report = classification_report(y_test, y_test_pred)
print("Báo cáo phân loại trên tập kiểm tra:\n", class_report)

# Tính ROC AUC
roc_auc = roc_auc_score(y_test, y_test_proba)
print(f'ROC AUC: {roc_auc:.10f}')

# Vẽ đường cong ROC
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
plt.figure()
plt.plot(fpr, tpr, label='Đường ROC (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'k--')  # Đường tham chiếu
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tỷ lệ dương tính giả')
plt.ylabel('Tỷ lệ dương tính thật')
plt.title('Đường cong ROC')
plt.legend(loc='lower right')
plt.show()

# Vẽ ma trận nhầm lẫn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Dự đoán 0', 'Dự đoán 1'],
            yticklabels=['Thực tế 0', 'Thực tế 1'])
plt.ylabel('Giá trị thực tế')
plt.xlabel('Giá trị dự đoán')
plt.title('Ma trận nhầm lẫn')
plt.show()

# Dữ liệu mới để dự đoán (cần thay đổi cho phù hợp với số lượng đặc trưng)
new_data = np.array([[51,1,0,140,299,0,1,173,1,1.6,2,0,3]])  # Thay đổi kích thước thành 2D

# Chuẩn hóa dữ liệu mới
new_data_scaled = scaler.transform(new_data)

# Dự đoán với mô hình
new_prediction = best_model.predict(new_data_scaled)
new_prediction_proba = best_model.predict_proba(new_data_scaled)[:, 1]  # Lấy xác suất cho lớp 1

# In kết quả dự đoán
print(f'Dự đoán cho mẫu mới: {new_prediction[0]}')  # 0 hoặc 1
print(f'Xác suất dự đoán cho lớp 1: {new_prediction_proba[0]:.4f}')  # Xác suất cho lớp 1
