import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.feature_selection import mutual_info_regression

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# DataFrame 으로 만들어서 현재 경로에 california_housing.csv 이름의 파일로 저장
housing = fetch_california_housing(as_frame=True).frame
housing.to_csv('california_housing.csv', index=False)

# 정보 확인(shape / describe)
# print(housing.shape)
# 개선 2를 위한 정보 확인(편향치 확인)
# describe() -> 생략 없이 표기
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
print(housing.describe())
# 개선 2 : 위에서 확인한 평균<->최댓값 격차 큰 변수 편향치 확인
skew_features = housing[['MedInc','AveRooms','Population','AveOccup']].skew()
print(skew_features)

# 결측치 확인
# housing.info()
# print(housing.isnull().sum())

# 데이터 분리(학습/정답)
X = housing.drop(columns=['MedHouseVal'])
y = housing['MedHouseVal']

# 1. 전체 데이터 중 상관계수 확인
mi_scores = mutual_info_regression(X, y, random_state=42)
# 높은 순 정렬
mi_scores = (pd.Series(mi_scores, index=X.columns, name='MI scores')
             .sort_values(ascending=False))
# print()
# print(mi_scores.head(10))

# 2. 중복 변수 확인
# 수치형 변수 상관계수 절대값
corr_matrix = X.corr().abs()
# 자기자신과의 비교 / 중복 비교치 삭제
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
# 상관계수 높은(중복인) 데이터 추출
high_corr = upper_tri.stack()[upper_tri.stack() >= 0.8]

# 방/침실 수 중복 노이즈 확인
# 위도/경도는 서로 영향치가 높지만 하나만 해서는 위치값으로 의미가 없음.
# print()
# print(high_corr)

# 3. 가격 분포 히스토그램
plt.figure(figsize=(10, 5))
sns.histplot(y, bins=50, color='green', edgecolor='black', kde=True)
plt.title('캘리포니아 주택 가격 분포 - MedHouseVal')
plt.xlabel('주택 가격 / $100,000 단위')
plt.ylabel('구역(동네) 수')

plt.tight_layout()
plt.savefig('주택_가격분포_히스토그램.png')

# 4. 확인된 상위 상관계수 영향치 그래프
# 4-1. 소득 x 집값 산점도
fig, axes =plt.subplots(1, 2, figsize=(15, 5))
sns.scatterplot(data=housing, x='MedInc', y='MedHouseVal', ax=axes[0])
axes[0].set_title('소득 x 주택가격')

# 4-2. 위도/경도 x 집값
# 위
sns.scatterplot(data=housing, x='Longitude', y='Latitude',
                hue='MedHouseVal', ax=axes[1])
axes[1].set_title('위도/경도 x 주택가격')
axes[1].set_xlabel('경도 (-124:서 / -114:동)')
axes[1].set_ylabel('위도 (34:남 / 42:북)')

plt.tight_layout()
plt.savefig('주택_가격_상관계수.png')
# plt.show()