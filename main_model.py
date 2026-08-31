import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 데이터
housing = fetch_california_housing(as_frame=True).frame
# 개선 1 : eda에서 확인한 MedHouseVal 상한선 5.0에 다수 분포된 이상치 제거
housing_clean = housing[housing['MedHouseVal']<5.0].copy()

# 개선 2 : eda에서 확인한 편향치 큰 4개 변수 처리
skew_cols = ['MedInc','AveRooms','Population','AveOccup']
for col in skew_cols:
    housing_clean[col] = np.log1p(housing_clean[col])

# 데이터 분리(학습/정답)
# eda에서 확인한 중복 변수 제거 할당
# 개선 1 : housing -> housing_clean
X = housing_clean.drop(columns=['MedHouseVal', 'AveBedrms'])
y = housing_clean['MedHouseVal']

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

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")
