import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 데이터
housing = fetch_california_housing(as_frame=True).frame

# 데이터 분리(학습/정답)
# eda에서 확인한 중복 변수 제거 할당
X = housing.drop(columns=['MedHouseVal', 'AveBedrms'])
y = housing['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 모델
m=LinearRegression()
# 학습
m.fit(X_train_scaled, y_train)
# 예측
y_pred = m.predict(X_test_scaled)

# 평가
mae=mean_absolute_error(y_test, y_pred)
rmse=root_mean_squared_error(y_test, y_pred)
r2=r2_score(y_test, y_pred)

print(mae)
print(rmse)
print(r2)