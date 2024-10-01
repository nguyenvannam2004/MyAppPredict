import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import joblib

# Đọc dữ liệu từ file CSV
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

# In số mẫu của các tập
print(f'Số mẫu của tập huấn luyện: {len(X_train)}')
print(f'Số mẫu của tập xác thực: {len(X_val)}')
print(f'Số mẫu của tập kiểm tra: {len(X_test)}')

# Áp dụng SMOTE để xử lý mất cân bằng dữ liệu
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Chuẩn hóa dữ liệu bằng StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)  # Sử dụng dữ liệu đã được resample
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Khởi tạo các mô hình
svm_model = SVC(probability=True, kernel='poly', degree=4, C=0.01, gamma='scale', random_state=42)
logreg_model = LogisticRegression(random_state=42)

# Sử dụng mô hình Voting Classifier để kết hợp các mô hình
ensemble_model = VotingClassifier(estimators=[
    ('svm', svm_model),
    ('logreg', logreg_model)
], voting='soft')

# Huấn luyện mô hình với dữ liệu đã cân bằng
ensemble_model.fit(X_train_scaled, y_train_resampled)

# Dự đoán trên tập huấn luyện
y_train_pred = ensemble_model.predict(X_train_scaled)
accuracy_train = accuracy_score(y_train_resampled, y_train_pred) 
print(f'Dộ chính xác trên tập huấn luyện: {accuracy_train:.10f}')

# Dự đoán trên tập xác thực
y_val_pred = ensemble_model.predict(X_val_scaled)

# Đánh giá mô hình trên tập xác thực
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Dộ chính xác trên tập xác thực: {val_accuracy:.10f}')

# Dự đoán trên tập kiểm tra
y_test_pred = ensemble_model.predict(X_test_scaled)

# Đánh giá mô hình trên tập kiểm tra
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Dộ chính xác trên tập kiểm tra: {test_accuracy:.10f}')

# Đánh giá báo cáo phân loại
print("Báo cáo phân loại cho tập kiểm tra:")
print(classification_report(y_test, y_test_pred))

# Ma trận nhầm lẫn
conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Ma trận nhầm lẫn:\n", conf_matrix)

# Vẽ ma trận nhầm lẫn
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn')
plt.show()

# Đánh giá ROC và AUC
y_test_prob = ensemble_model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
roc_auc = roc_auc_score(y_test, y_test_prob)

plt.figure()
plt.plot(fpr, tpr, color='blue', label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('Tỷ lệ dương giả')
plt.ylabel('Tỷ lệ dương thật')
plt.title('Biểu đồ ROC')
plt.legend(loc='lower right')
plt.show()

# Vẽ phân phối xác suất dự đoán cho lớp 1
plt.figure(figsize=(10, 6))
sns.histplot(y_test_pred, bins=30, kde=True)  # Vẽ histogram với KDE
plt.xlabel('Xác suất dự đoán cho lớp 1')
plt.ylabel('Tần suất')
plt.title('Phân phối xác suất dự đoán cho lớp 1 ')
plt.axvline(x=0.5, color='red', linestyle='--', label='Ngưỡng 0.5')  # Đường ngưỡng
plt.legend()
plt.show()




# Hiệu chỉnh xác suất dự đoán của mô hình
calibrated_model = CalibratedClassifierCV(estimator=ensemble_model, method='sigmoid', cv='prefit')
calibrated_model.fit(X_val_scaled, y_val)

# Dự đoán trên tập kiểm tra với mô hình đã hiệu chỉnh
y_test_pred_calibrated = calibrated_model.predict(X_test_scaled)
#y_test_pred_calibrated = (y_test_pred_calibrated > 0.5).astype(int)
#Đánh giá mô hình đã hiệu chỉnh
test_accuracy_calibrated = accuracy_score(y_test, y_test_pred_calibrated)
print(f'Dộ chính xác trên tập kiểm tra sau khi hiệu chỉnh: {test_accuracy_calibrated:.10f}')

# Đánh giá báo cáo phân loại
print("Báo cáo phân loại cho tập kiểm tra sau khi hiệu chỉnh:")
print(classification_report(y_test, y_test_pred_calibrated))

# Ma trận nhầm lẫn
conf_matrix_calibrated = confusion_matrix(y_test, y_test_pred_calibrated)
print("Ma trận nhầm lẫn sau khi hiệu chỉnh:\n", conf_matrix_calibrated)

# Vẽ ma trận nhầm lẫn
sns.heatmap(conf_matrix_calibrated, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn sau khi hiệu chỉnh')
plt.show()

# Đánh giá ROC và AUC cho mô hình đã hiệu chỉnh
y_test_prob_calibrated = calibrated_model.predict_proba(X_test_scaled)[:, 1]
fpr_calibrated, tpr_calibrated, thresholds_calibrated = roc_curve(y_test, y_test_prob_calibrated)
roc_auc_calibrated = roc_auc_score(y_test, y_test_prob_calibrated)

plt.figure()
plt.plot(fpr_calibrated, tpr_calibrated, color='blue', label='ROC curve (AUC = %0.2f)' % roc_auc_calibrated)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('Tỷ lệ dương giả')
plt.ylabel('Tỷ lệ dương thật')
plt.title('Biểu đồ ROC sau khi hiệu chỉnh')
plt.legend(loc='lower right')
plt.show()

# Vẽ phân phối xác suất dự đoán cho lớp 1
plt.figure(figsize=(10, 6))
sns.histplot(y_test_prob_calibrated, bins=30, kde=True)  # Vẽ histogram với KDE
plt.xlabel('Xác suất dự đoán cho lớp 1')
plt.ylabel('Tần suất')
plt.title('Phân phối xác suất dự đoán cho lớp 1 sau khi hiệu chỉnh')
plt.axvline(x=0.5, color='red', linestyle='--', label='Ngưỡng 0.5')  # Đường ngưỡng
plt.legend()
plt.show()


joblib.dump(calibrated_model, 'ensemble_model.pkl') 
# Lưu scaler vào tệp
joblib.dump(scaler, 'scaler_ensemble.pkl')
# ensemble_model
# joblib.dump(ensemble_model, 'ensemble_model.pkl')

# Giả sử ngưỡng mới mà bạn muốn sử dụng là 0.3
# custom_threshold = 0.47

# # Dự đoán xác suất cho lớp dương trên tập kiểm tra
# y_test_prob = ensemble_model.predict_proba(X_test_scaled)[:, 1]

# # Dùng ngưỡng tùy chỉnh để xác định dự đoán lớp
# y_test_pred_custom = (y_test_prob > custom_threshold).astype(int)

# # Đánh giá mô hình với ngưỡng mới
# test_accuracy_custom = accuracy_score(y_test, y_test_pred_custom)
# print(f'Dộ chính xác trên tập kiểm tra với ngưỡng {custom_threshold}: {test_accuracy_custom:.10f}')

# # Đánh giá báo cáo phân loại
# print("Báo cáo phân loại cho tập kiểm tra với ngưỡng tùy chỉnh:")
# print(classification_report(y_test, y_test_pred_custom))

# # Ma trận nhầm lẫn
# conf_matrix_custom = confusion_matrix(y_test, y_test_pred_custom)
# print("Ma trận nhầm lẫn với ngưỡng tùy chỉnh:\n", conf_matrix_custom)

# # Vẽ ma trận nhầm lẫn
# sns.heatmap(conf_matrix_custom, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Dự đoán')
# plt.ylabel('Thực tế')
# plt.title(f'Ma trận nhầm lẫn với ngưỡng {custom_threshold}')
# plt.show()


# Dự đoán cho dữ liệu mới
# Giả sử dữ liệu mới có đặc trưng tương tự như dữ liệu đã huấn luyện
# Bạn cần cung cấp dữ liệu mới (thay đổi các giá trị cho phù hợp với dữ liệu của bạn)
X_new = np.array([[61,0,0,145,307,0,0,146,1,1.0,1,0,3]])

# Chuẩn hóa dữ liệu mới sử dụng scaler đã được huấn luyện
X_new_scaled = scaler.transform(X_new)

# Dự đoán lớp cho dữ liệu mới
y_new_pred = calibrated_model.predict(X_new_scaled)
print(f'Dự đoán lớp cho dữ liệu mới: {y_new_pred}')


