import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
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

# Cột 'target' là đầu ra y
y = data['target'].values

# Chia dữ liệu thành các tập huấn luyện (70%), kiểm tra (15%) và xác thực (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# In số mẫu của các tập
print(f'Số mẫu của tập huấn luyện: {len(X_train)}')
print(f'Số mẫu của tập xác thực: {len(X_val)}')
print(f'Số mẫu của tập kiểm tra: {len(X_test)}')

# Chuẩn hóa dữ liệu bằng StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# Khởi tạo mô hình Perceptron
model = Perceptron(max_iter=1000, tol=0.001,eta0=0.1,alpha=0.001, random_state=42)

# Huấn luyện mô hình
model.fit(X_train_scaled, y_train)

# Lưu mô hình vào tệp
joblib.dump(model, 'perceptron_model.pkl')  # Hoặc sử dụng pickle
# Lưu scaler vào tệp
joblib.dump(scaler, 'scaler.pkl')

#dư đoán trên tập huấn luyện
y_train_pred = model.predict(X_train_scaled)

# Dự đoán trên tập xác thực
y_val_pred = model.predict(X_val_scaled)

# Dự đoán xác suất trên tập kiểm tra
y_test_pred = model.predict(X_test_scaled)
y_test_proba = model.decision_function(X_test_scaled)  # Lấy xác suất

# Tính độ chính xác
accuracy_train = accuracy_score(y_train, y_train_pred) 
accuracy_val = accuracy_score(y_val, y_val_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)

# In kết quả
print(f'Dộ chính xác trên tập huấn luyện: {accuracy_train:.10f}')
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
new_data = np.array([[60,0,3,150,240,0,1,171,0,0.9,2,0,2]])  # Thay đổi kích thước thành 2D

# Chuẩn hóa dữ liệu mới
new_data_scaled = scaler.transform(new_data)

# Dự đoán với mô hình
new_prediction = model.predict(new_data_scaled)
new_prediction_proba = model.decision_function(new_data_scaled)  # Lấy giá trị quyết định

# Chuyển đổi giá trị quyết định thành xác suất
probability = 1 / (1 + np.exp(-new_prediction_proba))  # Hàm sigmoid

# In kết quả dự đoán
print(f'Dự đoán cho mẫu mới: {new_prediction[0]}')  # 0 hoặc 1
print(f'Xác suất dự đoán cho lớp 1: {probability[0]:.4f}')  # Xác suất cho lớp 1



# Lấy giá trị quyết định cho tập kiểm tra
y_test_proba = model.decision_function(X_test_scaled)

# Thay đổi ngưỡng quyết định (ví dụ: ngưỡng là 0.3)
threshold = 0.65
y_test_pred_custom_threshold = (y_test_proba >= threshold).astype(int)

# In kết quả mới
# print(f'Dự đoán với ngưỡng {threshold}: {y_test_pred_custom_threshold}')

# Tính lại độ chính xác và báo cáo phân loại với ngưỡng tùy chỉnh
accuracy_test_custom_threshold = accuracy_score(y_test, y_test_pred_custom_threshold)
print(f'Dộ chính xác trên tập kiểm tra với ngưỡng {threshold}: {accuracy_test_custom_threshold:.10f}')

# Báo cáo phân loại mới
class_report_custom = classification_report(y_test, y_test_pred_custom_threshold)
print("Báo cáo phân loại trên tập kiểm tra với ngưỡng tùy chỉnh:\n", class_report_custom)

# Vẽ lại ma trận nhầm lẫn
conf_matrix_custom = confusion_matrix(y_test, y_test_pred_custom_threshold)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_custom, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Dự đoán 0', 'Dự đoán 1'],
            yticklabels=['Thực tế 0', 'Thực tế 1'])
plt.ylabel('Giá trị thực tế')
plt.xlabel('Giá trị dự đoán')
plt.title(f'Ma trận nhầm lẫn (Ngưỡng {threshold})')
plt.show()

new_data = np.array([[62,0,1,124,281,1,1,103,0,1.4,1,1,1]])  # Thay đổi kích thước thành 2D

# Chuẩn hóa dữ liệu mới
new_data_scaled = scaler.transform(new_data)

# Dự đoán với mô hình
new_prediction = model.predict(new_data_scaled)
new_prediction_proba = model.decision_function(new_data_scaled)  # Lấy giá trị quyết định

# Chuyển đổi giá trị quyết định thành xác suất
probability = 1 / (1 + np.exp(-new_prediction_proba))  # Hàm sigmoid

# In kết quả dự đoán
print(f'Dự đoán cho mẫu mới: {new_prediction[0]}')  # 0 hoặc 1
print(f'Xác suất dự đoán cho lớp 1: {probability[0]:.4f}')  # Xác suất cho lớp 1