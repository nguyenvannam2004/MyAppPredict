import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from imblearn.over_sampling import SMOTE
import numpy as np

file_path = './mynewdata.csv'  
data = pd.read_csv(file_path)
data.drop([0, 1, 2], inplace=True)  

X = data.drop(columns=['STT', 'target']).values

y = data['target'].values

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f'Số mẫu của tập huấn luyện: {len(X_train)}')
print(f'Số mẫu của tập xác thực: {len(X_val)}')
print(f'Số mẫu của tập kiểm tra: {len(X_test)}')

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)  
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=100,C=0.01,penalty='l2',solver='lbfgs', random_state=42)

model.fit(X_train_scaled, y_train_resampled)

joblib.dump(model, 'logistic_regression_model.pkl')  

y_train_pred = model.predict(X_train_scaled)
y_val_pred = model.predict(X_val_scaled)

y_test_pred = model.predict(X_test_scaled)
y_test_proba = model.predict_proba(X_test_scaled)[:, 1]  # Lấy xác suất cho lớp 1

accuracy_train = accuracy_score(y_train_resampled, y_train_pred) 
accuracy_val = accuracy_score(y_val, y_val_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)

print(f'Dộ chính xác trên tập huấn luyện: {accuracy_train:.10f}')
print(f'Dộ chính xác trên tập xác thực: {accuracy_val:.10f}')
print(f'Dộ chính xác trên tập kiểm tra: {accuracy_test:.10f}')

conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Ma trận nhầm lẫn:\n", conf_matrix)

class_report = classification_report(y_test, y_test_pred)
print("Báo cáo phân loại trên tập kiểm tra:\n", class_report)

roc_auc = roc_auc_score(y_test, y_test_proba)
print(f'ROC AUC: {roc_auc:.10f}')

fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)

plt.figure()
plt.plot(fpr, tpr, label='Đường ROC (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'k--')  
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tỷ lệ dương tính giả')
plt.ylabel('Tỷ lệ dương tính thật')
plt.title('Đường cong ROC')
plt.legend(loc='lower right')
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Dự đoán 0', 'Dự đoán 1'],
            yticklabels=['Thực tế 0', 'Thực tế 1'])
plt.ylabel('Giá trị thực tế')
plt.xlabel('Giá trị dự đoán')
plt.title('Ma trận nhầm lẫn')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(y_test_proba, bins=30, kde=True)  
plt.xlabel('Xác suất dự đoán cho lớp 1')
plt.ylabel('Tần suất')
plt.title('Phân phối xác suất dự đoán cho lớp 1')
plt.axvline(x=0.5, color='red', linestyle='--', label='Ngưỡng 0.5')  
plt.legend()
plt.show()

new_data = np.array([[62,1,1,120,281,0,0,103,0,1.4,1,1,3]])  

new_data_scaled = scaler.transform(new_data)

new_prediction = model.predict(new_data_scaled)
new_prediction_proba = model.predict_proba(new_data_scaled)[:, 1]  

print(f'Dự đoán cho mẫu mới: {new_prediction[0]}')  
print(f'Xác suất dự đoán cho lớp 1: {new_prediction_proba[0]:.4f}')  
