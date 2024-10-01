import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from imblearn.over_sampling import SMOTE

# Đọc dữ liệu từ file CSV và chuẩn bị tập huấn luyện, kiểm tra, xác thực
file_path = './mynewdata.csv'  # Thay đổi đường dẫn tới tệp của bạn
data = pd.read_csv(file_path)
data.drop([0, 1, 2], inplace=True)  # xóa 3 mẫu cho tròn 300 mẫu dữ liệu

# Bỏ cột 'STT' và 'target' khỏi tập đặc trưng X
X = data.drop(columns=['STT', 'target']).values

# Cột 'target' là đầu ra y
y = data['target'].values

# Chia dữ liệu thành các tập huấn luyện (70%), kiểm tra (15%) và xác thực (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Áp dụng SMOTE để cân bằng tập huấn luyện
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Chuẩn hóa dữ liệu bằng StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Khởi tạo mô hình Perceptron
model = Perceptron(random_state=42)

# Định nghĩa tham số cho GridSearchCV
param_grid = {
    'max_iter': [1000, 1500, 2000],
    'tol': [1e-3, 1e-4],
    'eta0': [0.1, 0.01],
    'alpha': [0.001, 0.01]
}

# Sử dụng GridSearchCV để tìm tham số tối ưu
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train_resampled)

# Lấy mô hình tối ưu
best_model = grid_search.best_estimator_

# In tham số tốt nhất
print("Tham số tốt nhất:", grid_search.best_params_)

# Huấn luyện mô hình tối ưu trên dữ liệu đã được cân bằng
best_model.fit(X_train_resampled, y_train_resampled)

# Dự đoán trên tập huấn luyện đã được cân bằng
y_train_pred_resampled = best_model.predict(X_train_resampled)

# Dự đoán trên tập xác thực
y_val_pred = best_model.predict(X_val_scaled)

# Dự đoán trên tập kiểm tra
y_test_pred = best_model.predict(X_test_scaled)
y_test_proba = best_model.decision_function(X_test_scaled)  # Lấy xác suất

# Tính độ chính xác
accuracy_train = accuracy_score(y_train_resampled, y_train_pred_resampled)  # Độ chính xác trên tập huấn luyện
accuracy_val = accuracy_score(y_val, y_val_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)

# In kết quả
print(f'Dộ chính xác trên tập huấn luyện: {accuracy_train:.10f}')  # Kết quả độ chính xác trên tập huấn luyện
print(f'Dộ chính xác trên tập xác thực: {accuracy_val:.10f}')
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
new_data = np.array([[51, 1, 0, 140, 299, 0, 1, 173, 1, 1.6, 2, 0, 3]])  # Thay đổi kích thước thành 2D

# Chuẩn hóa dữ liệu mới
new_data_scaled = scaler.transform(new_data)

# Dự đoán với mô hình
new_prediction = best_model.predict(new_data_scaled)
new_prediction_proba = best_model.decision_function(new_data_scaled)  # Lấy giá trị quyết định

# Chuyển đổi giá trị quyết định thành xác suất
probability = 1 / (1 + np.exp(-new_prediction_proba))  # Hàm sigmoid

# In kết quả dự đoán
print(f'Dự đoán cho mẫu mới: {new_prediction[0]}')  # 0 hoặc 1
print(f'Xác suất dự đoán cho lớp 1: {probability[0]:.4f}')  # Xác suất cho lớp 1
