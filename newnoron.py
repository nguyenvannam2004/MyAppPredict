import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import numpy as np
import joblib

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, lambda_reg=0.03):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        np.random.seed(42)
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    def train(self, X_train, y_train, epochs=2000):
        for epoch in range(epochs):
           
            hidden_layer_input = np.dot(X_train, self.weights_input_hidden)
            hidden_layer_output = self.sigmoid(hidden_layer_input)
            final_input = np.dot(hidden_layer_output, self.weights_hidden_output)
            final_output = self.sigmoid(final_input)

            error = y_train.reshape(-1, 1) - final_output

            d_final_output = error * self.sigmoid_derivative(final_output)
            error_hidden_layer = d_final_output.dot(self.weights_hidden_output.T)
            d_hidden_layer = error_hidden_layer * self.sigmoid_derivative(hidden_layer_output)

            self.weights_hidden_output += hidden_layer_output.T.dot(d_final_output) * self.learning_rate - self.lambda_reg * self.weights_hidden_output
            self.weights_input_hidden += X_train.T.dot(d_hidden_layer) * self.learning_rate - self.lambda_reg * self.weights_input_hidden

    def predict(self, X_new, threshold):
        X_new = (X_new - np.mean(X_new, axis=0)) / np.std(X_new, axis=0)

        hidden_layer_input = np.dot(X_new, self.weights_input_hidden)
        hidden_layer_output = self.sigmoid(hidden_layer_input)
        final_input = np.dot(hidden_layer_output, self.weights_hidden_output)
        final_output = self.sigmoid(final_input)

        y_pred = (final_output > threshold).astype(int)
        return y_pred
file_path = './mynewdata.csv'  
data = pd.read_csv(file_path)
data.drop([0, 1, 2], inplace=True)  

X = data.drop(columns=['STT', 'target']).values

y = data['target'].values

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
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

input_size = X_train.shape[1]  
hidden_size = 10  
output_size = 1 

nn = NeuralNetwork(input_size, hidden_size, output_size)

nn.train(X_train_resampled, y_train_resampled)
y_train_pred = nn.predict(X_train_resampled, 0.5)
train_accuracy = accuracy_score(y_train_resampled, y_train_pred)
print(f'Dộ chính xác trên tập huấn luyện: {train_accuracy:.10f}')

y_val_pred = nn.predict(X_val, 0.5)

val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Dộ chính xác trên tập xác thực: {val_accuracy:.10f}')

y_test_pred = nn.predict(X_test, 0.5)  
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Dộ chính xác trên tập kiểm tra: {test_accuracy:.10f}')

