import streamlit as st
import joblib
import pandas as pd
from NeurualNetwork import NeuralNetwork 
import numpy as np
# Tải các mô hình đã lưu
model_pla = joblib.load('perceptron_model.pkl')
model_logistic = joblib.load('logistic_regression_model.pkl')
model_neural = joblib.load('neural_network_model.pkl')
model_ensemble = joblib.load('ensemble_model.pkl')

# Tạo tiêu đề cho ứng dụng
st.title("Dự Đoán Bệnh Tim")

# Tạo một dropdown để chọn mô hình
model_options = {
    'PLA Model': model_pla,
    'Logistic Regression Model': model_logistic,
    'Neural Network Model': model_neural,
    'Ensemble Model': model_ensemble
}

# Nhập dữ liệu đầu vào
age = st.number_input("Tuổi:", min_value=0, max_value=120)
sex = st.selectbox("Giới tính:", options=[0, 1], format_func=lambda x: "Nam" if x == 1 else "Nữ")
cp = st.selectbox("Chỉ số đau ngực:", options=[0, 1, 2, 3],format_func=lambda x: "Không đau" if x==0 else "Đau ngực nhẹ" if x==1 else "Đau ngực vừa" if x==2 else "Đau ngực nặng")
trestbps = st.number_input("Huyết áp nghỉ (mm Hg):", min_value=0)
chol = st.number_input("Cholesterol (mg/dl):", min_value=0)
fbs = st.selectbox("Đường huyết lúc nhịn ăn:", options=[0, 1],format_func = lambda x : "Dưới 120 mg/dl" if x==0 else "120 mg/dl hoặc cao hơn")
restecg = st.selectbox("Kết quả điện tâm đồ:", options=[0, 1, 2], format_func=lambda x: "Bình thường" if x==0 else "Có sóng ST chênh lên" if x==1 else "Có sóng ST chênh xuống")
thalach = st.number_input("Tần số tim tối đa:", min_value=0)
exang = st.selectbox("Đau thắt ngực khi hoạt động:", options=[0, 1], format_func = lambda x: "Không đau" if x==0 else "Có đau")
oldpeak = st.number_input("Độ dốc ST (0.0 - 6.2):", min_value=0.0, max_value=6.2)
slope = st.selectbox("Độ dốc của đỉnh ST:", options=[0, 1, 2],format_func=lambda x: "Dốc lên" if x==0 else "Bằng phẳng" if x==1 else "Dốc xuống")
ca = st.selectbox("Số mạch máu lớn:", options=[0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia:", options=[0, 1, 2, 3],format_func=lambda x: "Bình thường" if x==0 else "Khuyết tật cố hữu" if x==1 else "Khuyết tật tạm thời" if x==2 else "Không rõ")
selected_model = st.selectbox("Chọn mô hình:", list(model_options.keys()))
# Nút dự đoán
if st.button("Dự đoán"):
    features = [
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]
   
    # Chuyển đổi thành DataFrame với tên cột
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                     'restecg', 'thalach', 'exang', 'oldpeak', 
                     'slope', 'ca', 'thal']
    features_df = pd.DataFrame([features], columns=feature_names)

    # Dự đoán bằng mô hình đã chọn
    model = model_options[selected_model]

    if model == model_pla:
        # Sử dụng decision_function cho mô hình PLA
        # Tải scaler từ tệp (đối với dữ liệu mới)
        scaler_loaded = joblib.load('scaler.pkl')
        features_df = scaler_loaded.transform(features_df)  # Chuẩn hóa dữ liệu
        prediction = model.decision_function(features_df)  
        probability = 1 / (1 + np.exp(-prediction))  # Chuyển đổi thành xác suất
        
        if probability >= 0.65:
            result = 1
        else:
            result = 0
    elif model == model_ensemble:
        scaler_loaded = joblib.load('scaler_ensemble.pkl')
        features_df = scaler_loaded.transform(features_df)
        prediction = model.predict(features_df)
        result = prediction[0]
    else:
        # Dự đoán cho các mô hình còn lại
        prediction = model.predict(features_df)
        result = prediction[0]

    # Kết quả dự đoán
    if result == 1:
        result_message = (
            "Có nguy cơ mắc bệnh tim.\n\n"
            "Đừng buồn, bạn hãy chạy vào trong vườn hái một quả chanh.\n"
            "Bổ đôi nó ra (nhớ là phải bổ ngang nha) xong vắt nó vào cốc, cho 2 thìa đường,\n"
            "500ml nước đun sôi để nguội.Cho thêm 2 viên đá nữa cho mát rồi khuấy đều lên sẽ thu đc dung dịch hay còn gọi là nước đường.\n"
            "Uống nó! Nó sẽ không giúp bạn hết bị bệnh tim đâu nhưng mà nước đường thì rất là ngọt, với lại biết đâu đó có thể sẽ là lần cuối cùng mà bạn đc uống nước đường thì sao =))))"
        )
    else:
        result_message = "Không có nguy cơ mắc bệnh tim."
    
    st.success(result_message)
