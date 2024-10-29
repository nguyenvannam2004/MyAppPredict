# Import các thư viện cần thiết
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Perceptron
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

file_path = './mynewdata.csv'
data = pd.read_csv(file_path)

data.drop([0, 1, 2], inplace=True)

X = data.drop(columns=['STT', 'target']).values
y = data['target'].values

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f'Số mẫu của tập huấn luyện: {len(X_train)}')
print(f'Số mẫu của tập xác thực: {len(X_val)}')
print(f'Số mẫu của tập kiểm tra: {len(X_test)}')

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


#svm_model = SVC(kernel='poly', degree=3, C=0.001, gamma='scale', random_state=42)
model_pla = Perceptron(max_iter=1000, tol=0.001, eta0=0.1, alpha=0.001, random_state=42)
model_logic = LogisticRegression(max_iter=100, C=0.01, penalty='l2', solver='lbfgs', random_state=42)
model_noron = MLPClassifier(hidden_layer_sizes=(10,), max_iter=2000, learning_rate_init=0.1, alpha=0.03, random_state=42)
#gradient_model = GradientBoostingClassifier(random_state=42)

ensemble_model = VotingClassifier(estimators=[
    #('svm', svm_model),
    ('logreg', model_logic),
    ('pla', model_pla),
    ('noron', model_noron),
    #('grdien', gradient_model)
], voting='hard')

ensemble_model.fit(X_train_scaled, y_train_resampled)

y_train_pred = ensemble_model.predict(X_train_scaled)
accuracy_train = accuracy_score(y_train_resampled, y_train_pred)
print(f'Độ chính xác trên tập huấn luyện: {accuracy_train:.10f}')

y_val_pred = ensemble_model.predict(X_val_scaled)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Độ chính xác trên tập xác thực: {val_accuracy:.10f}')

y_test_pred = ensemble_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Độ chính xác trên tập kiểm tra: {test_accuracy:.10f}')

print("Báo cáo phân loại cho tập kiểm tra:")
print(classification_report(y_test, y_test_pred))

conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Ma trận nhầm lẫn:\n", conf_matrix)

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn')
plt.show()

joblib.dump(ensemble_model, 'ensemble_model_hard_voting.pkl')
joblib.dump(scaler, 'scaler_ensemble.pkl')

X_new = np.array([[57, 1, 1, 124, 261, 0, 1, 141, 0, 0.3, 2, 0, 3]])
X_new_scaled = scaler.transform(X_new)
y_new_pred = ensemble_model.predict(X_new_scaled)
print(f'Dự đoán lớp cho dữ liệu mới: {y_new_pred}')

# dữ liệu mới
file_path_new = './heart.csv' 
data_new = pd.read_csv(file_path_new)
X_new_data = data_new.drop(columns=['target']).values
y_new_data = data_new['target'].values

X_train_new, X_temp_new, y_train_new, y_temp_new = train_test_split(X_new_data, y_new_data, test_size=0.4, random_state=42)
X_val_new, X_test_new, y_val_new, y_test_new = train_test_split(X_temp_new, y_temp_new, test_size=0.5, random_state=42)

print(f'Số mẫu của tập huấn luyện mới: {len(X_train_new)}')
print(f'Số mẫu của tập xác thực mới: {len(X_val_new)}')
print(f'Số mẫu của tập kiểm tra mới: {len(X_test_new)}')

smote_new = SMOTE(random_state=42)
X_train_resampled_new, y_train_resampled_new = smote_new.fit_resample(X_train_new, y_train_new)

X_train_scaled_new = scaler.transform(X_train_resampled_new)  
X_val_scaled_new = scaler.transform(X_val_new)
X_test_scaled_new = scaler.transform(X_test_new)

ensemble_model = joblib.load('ensemble_model_hard_voting.pkl')

model_pla.fit(X_train_scaled_new, y_train_resampled_new)  
model_logic.fit(X_train_scaled_new, y_train_resampled_new)  
model_noron.fit(X_train_scaled_new, y_train_resampled_new)  

ensemble_model.fit(X_train_scaled_new, y_train_resampled_new)

#
y_train_pred = ensemble_model.predict(X_train_scaled_new)
accuracy_train = accuracy_score(y_train_resampled_new, y_train_pred) 
print(f'Dộ chính xác trên tập huấn luyện: {accuracy_train:.10f}')

y_val_pred = ensemble_model.predict(X_val_scaled_new)
val_accuracy = accuracy_score(y_val_new, y_val_pred)
print(f'Dộ chính xác trên tập xác thực: {val_accuracy:.10f}')

y_new_pred_finetune = ensemble_model.predict(X_test_scaled_new)
print(f'Độ chính xác trên dữ liệu mới: {accuracy_score(y_test_new, y_new_pred_finetune):.10f}')
#
print("Báo cáo phân loại cho dữ liệu mới:")
print(classification_report(y_test_new, y_new_pred_finetune))

conf_matrix_new = confusion_matrix(y_test_new, y_new_pred_finetune)
print("Ma trận nhầm lẫn cho dữ liệu mới:\n", conf_matrix_new)

sns.heatmap(conf_matrix_new, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn cho dữ liệu mới')
plt.show()

joblib.dump(ensemble_model, 'new_ensemble_model_hard_voting.pkl')