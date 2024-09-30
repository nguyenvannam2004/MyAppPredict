import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import numpy as np
import joblib

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, lambda_reg=0.03):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg

        # Khởi tạo trọng số ngẫu nhiên
        np.random.seed(42)
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)

    # Hàm kích hoạt Sigmoid
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Đạo hàm của hàm kích hoạt Sigmoid
    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    # Hàm huấn luyện mạng nơ-ron
    def train(self, X_train, y_train, epochs=2000):
        for epoch in range(epochs):
            # Giai đoạn lan truyền tiến
            hidden_layer_input = np.dot(X_train, self.weights_input_hidden)
            hidden_layer_output = self.sigmoid(hidden_layer_input)
            final_input = np.dot(hidden_layer_output, self.weights_hidden_output)
            final_output = self.sigmoid(final_input)

            # Tính toán lỗi
            error = y_train.reshape(-1, 1) - final_output

            # Lan truyền ngược
            d_final_output = error * self.sigmoid_derivative(final_output)
            error_hidden_layer = d_final_output.dot(self.weights_hidden_output.T)
            d_hidden_layer = error_hidden_layer * self.sigmoid_derivative(hidden_layer_output)

            # Cập nhật trọng số với regularization
            self.weights_hidden_output += hidden_layer_output.T.dot(d_final_output) * self.learning_rate - self.lambda_reg * self.weights_hidden_output
            self.weights_input_hidden += X_train.T.dot(d_hidden_layer) * self.learning_rate - self.lambda_reg * self.weights_input_hidden

    # Hàm dự đoán
    def predict(self, X_new):
        # Chuẩn hóa dữ liệu mới
        X_new = (X_new - np.mean(X_new, axis=0)) / np.std(X_new, axis=0)

        # Giai đoạn lan truyền tiến
        hidden_layer_input = np.dot(X_new, self.weights_input_hidden)
        hidden_layer_output = self.sigmoid(hidden_layer_input)
        final_input = np.dot(hidden_layer_output, self.weights_hidden_output)
        final_output = self.sigmoid(final_input)

        #Chuyển đổi đầu ra thành nhãn dự đoán
        y_pred = (final_output > 0.57).astype(int)
        return y_pred

    # Lưu mô hình vào file
    def save_model(self, filename):
        model = {
            "weights_input_hidden": self.weights_input_hidden,
            "weights_hidden_output": self.weights_hidden_output
        }
        joblib.dump(model, filename)

    # Tải mô hình từ file
    @classmethod
    def load_model(cls, filename):
        model = joblib.load(filename)
        nn = cls(model['weights_input_hidden'].shape[0], model['weights_hidden_output'].shape[0], model['weights_hidden_output'].shape[1])
        nn.weights_input_hidden = model['weights_input_hidden']
        nn.weights_hidden_output = model['weights_hidden_output']
        return nn

# Đọc dữ liệu từ file CSV và chuẩn bị tập huấn luyện, kiểm tra, xác thực
file_path = './mynewdata.csv'  # Thay đổi đường dẫn tới tệp của bạn
data = pd.read_csv(file_path)
data.drop([0, 1, 2], inplace=True)  # xóa 3 mẫu cho tròn 300 mẫu dữ liệu

# Bỏ cột 'STT' và 'target' khỏi tập đặc trưng X
X = data.drop(columns=['STT', 'target']).values

# Cột 'target' là đầu ra y
y = data['target'].values

# Chuẩn hóa dữ liệu
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Chia dữ liệu thành các tập huấn luyện (70%), kiểm tra (15%) và xác thực (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Áp dụng SMOTE để xử lý mất cân bằng dữ liệu
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# In số mẫu của các tập
print(f'Số mẫu của tập huấn luyện: {len(X_train)}')
print(f'Số mẫu của tập xác thực: {len(X_val)}')
print(f'Số mẫu của tập kiểm tra: {len(X_test)}')

# Chuẩn hóa dữ liệu bằng StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)  # Sử dụng dữ liệu đã được resample
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Khởi tạo mạng nơ-ron
input_size = X_train.shape[1]  # Kích thước đầu vào
hidden_size = 10  # Số lượng nơ-ron trong lớp ẩn
output_size = 1  # Số lượng nơ-ron đầu ra

nn = NeuralNetwork(input_size, hidden_size, output_size)

# Huấn luyện mạng nơ-ron
nn.train(X_train_resampled, y_train_resampled)

# Dự đoán trên tập huấn luyện
y_train_pred = nn.predict(X_train_resampled)

# Tính độ chính xác trên tập huấn luyện
train_accuracy = accuracy_score(y_train_resampled, y_train_pred)
print(f'Dộ chính xác trên tập huấn luyện: {train_accuracy:.10f}')

# Dự đoán trên tập xác thực
y_val_pred = nn.predict(X_val)

# Tính độ chính xác trên tập xác thực
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Dộ chính xác trên tập xác thực: {val_accuracy:.10f}')

# Dự đoán và đánh giá trên tập kiểm tra
y_test_pred = nn.predict(X_test)

# Tính độ chính xác trên tập kiểm tra
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Dộ chính xác trên tập kiểm tra: {test_accuracy:.10f}')

# Tính toán xác suất dự đoán
y_test_proba = nn.sigmoid(np.dot(nn.sigmoid(np.dot(X_test, nn.weights_input_hidden)), nn.weights_hidden_output)).flatten()  

# Tính và vẽ ROC Curve (chỉ áp dụng với bài toán nhị phân)
if len(set(y)) == 2:
    # Tính toán ROC và AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    auc = roc_auc_score(y_test, y_test_proba)
    print(f'Giá trị AUC: {auc:.2f}')
    
    # Vẽ ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Đường tham chiếu
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

# Vẽ biểu đồ phân phối xác suất dự đoán
plt.figure(figsize=(8, 6))
sns.histplot(y_test_proba, bins=20, kde=True, color='blue')
plt.title("Distribution of Prediction Probabilities")
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.show()

# Tính toán và in ra ma trận nhầm lẫn
conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Ma trận nhầm lẫn:")
print(conf_matrix)

# Vẽ ma trận nhầm lẫn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Không có bệnh', 'Có bệnh'], yticklabels=['Không có bệnh', 'Có bệnh'])
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn')
plt.show()

class_report = classification_report(y_test, y_test_pred)
print('Báo cáo phân loại:')
print(class_report)

new_data = np.array([62,0,1,124,281,1,1,103,0,1.4,1,1,1])

# Dự đoán với dữ liệu mới
predictions = nn.predict(new_data)

# In kết quả dự đoán
print("Dự đoán:", predictions)

# lưu mô hình 
joblib.dump(nn, 'neural_network_model.pkl')  
