import pandas as pd

# Đọc file sentiment_data.csv
data = pd.read_csv("sentiment_data.csv")

# Tách 400 dòng đầu thành train_data.csv
train_data = data.iloc[:400]
train_data.to_csv("train_data.csv", index=False)

# Tách 100 dòng cuối thành test_data.csv
test_data = data.iloc[400:]
test_data.to_csv("test_data.csv", index=False)

print("Đã tạo thành công train_data.csv và test_data.csv!")