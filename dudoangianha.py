import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Tải dữ liệu
file_path = "C:/Users/vietn/OneDrive/Desktop/BTLAI/Data_Set.csv"  # Đổi thành đường dẫn file của bạn
df = pd.read_csv(file_path)

# Giả sử cột "Price" là giá nhà (label), các cột còn lại là feature
X = df.drop(columns=["Giá"])  # Bỏ cột giá
y = df["Giá"]  # Chỉ lấy cột giá

# Chia train/test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Dữ liệu đã sẵn sàng để huấn luyện mô hình!")

# Định nghĩa mô hình mạng nơ-ron
model = keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)),  # Hidden Layer 1
    layers.Dropout(0.2),  # Dropout để tránh overfitting
    layers.Dense(64, activation="relu"),  # Hidden Layer 2
    layers.Dense(32, activation="relu"),  # Hidden Layer 3
    layers.Dense(16, activation="relu"),  # Hidden Layer 4 (thêm lớp này)
    layers.Dense(1)  # Output Layer
])



# Compile mô hình
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss="mse", metrics=["mae"])

# In thông tin về mô hình
model.summary()

# Huấn luyện mô hình
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=150, batch_size=64)

# Lưu mô hình sau khi train (tùy chọn)
model.save("house_price_model.h5")


# Đánh giá trên tập test
loss, mae = model.evaluate(X_test, y_test)
print(f"Mean Absolute Error trên tập test: {mae:.2f}")


# Lấy 5 mẫu ngẫu nhiên
sample_indices = np.random.choice(X_test.shape[0], 5, replace=False)

# Lấy dữ liệu mẫu từ tập test
sample_houses = X_test[sample_indices]

# Dự đoán toàn bộ 5 mẫu cùng lúc
predicted_prices = model.predict(sample_houses).flatten()

# In kết quả
for idx, predicted_price in zip(sample_indices, predicted_prices):
    actual_price = y_test.iloc[idx]
    print(f"🏠 Mẫu {idx} - Giá thực tế: {actual_price:.2f}, Giá dự đoán: {predicted_price:.2f}")

loss, mae = model.evaluate(X_test, y_test)
print(f"Mean Absolute Error (MAE) trên tập test: {mae:.2f}")

