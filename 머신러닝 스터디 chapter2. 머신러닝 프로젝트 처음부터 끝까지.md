# 머신러닝 스터디 chapter2. 머신러닝 프로젝트 처음부터 끝까지

> 1. 큰 그림을 봅니다
> 2. 데이터를 구합니다
> 3. 데이터로부터 통찰을 얻기 위해 탐색하고 시각화합니다
> 4. 머신러닝 알고리즘을 위해 데이터를 준비합니다
> 5. 모델을 선택하고 훈련시킵니다
> 6. 모델을 상세하게 조정합니다
> 7. 솔루션을 제시합니다
> 8. 시스템을 론칭하고 모니터링하고 유지보수합니다

## 2.1 실제 데이터로 작업하기

1. 유명한 공개 데이터 저장소
   - UC얼바인 머신러닝 저장소(http://archive.ics.uci.edu/ml)
   - 캐글 데이터셋(http://www.kaggle.com/datasets)
   - 아마존 AWS 데이터셋(https://registry.opendata.aws)

2. 메타 포털
   - 데이터 포털(http://dataportals.org)
   - 오픈 데이터 모니터(http://opendatamonitor.eu)
   - 퀀들(http://quandl.com)

3. 인기있는 공개 데이터 저장소가 나열되어 있는 다른 페이지
   - 위키백과 머신러닝 데이터셋 목록(https://goo.gl/SJHN2k)
   - Quora.com(https://homl.info/10)
   - 데이터셋 서브레딧(http://www.reddit.com/r/datasets)

## 2.2 큰 그림 보기

캘리포니아 인구조사 데이터를 사용해 캘리포니아의 주택 가격 모델 만들기

이 데이터로 모델을 학습시켜서 다른 측정 데이터가 주어졌을 때 구역의 중간 주택 가격을 예측

### 2.2.1 문제 정의

레이블된 훈련 샘플(구역의 중간 주택 가격을 가지고 있음)이 있으므로 지도 학습

+값을 예측해야 하므로 회귀 문제

+예측에 사용할 특성이 여러개(구역의 인구, 중간 소득 등)이므로 다중 회귀 문제

+각 구역마다 하나의 값을 예측하므로 단변량 회귀 문제

### 2.2.2 성능 측정 지표 선택

회귀 문제의 전형적인 성능 지표는 평균 제곱근 오차(root mean square error, RMSE), 오차가 커질수록 이 값은 커지므로 예측에 얼마나 많은 오류가 있는 가늠
$$
RMSE(X, h)=\sqrt{{1\over m}\sum_{i=1}^m(h(x^i)-y^i)^2}\\
m=RMSE를\ 측정할\ 데이터셋에\ 있는\ 샘플\ 수\\
x^i=데이터셋에\ 있는\ i번째\ 샘플(레이블을\ 제외한)의\ 전체\ 특성값의\ 벡터\\
y^i=해당\ 레이블(해당\ 샘플의\ 기대\ 출력값)\\
X=데이터셋에\ 있는\ 모든\ 샘플의\ 모든\ 특성값(레이블은\ 제외)을\ 포함하는\ 행렬\\
h=시스템의\ 예측\ 함수(=가설)
$$
이상치가 많은 경우에는 평균 절대 오차(mean absolute error, MAE(=평균 절대 편차, mean absolute deviation))도 고려
$$
MAE(X, h)={1\over m}\sum_{i=1}^m\left| h(x^i)-y^i \right|\\
$$
RMSE, MAE 모두 예측값의 벡터와 타깃값의 벡터 사이의 거리를 재는 방법

거리 측정 방법(=노름)의 종류 : 유클리디안 노름(제곱항을 합한 것의 제곱근(RMSE)), 맨해튼 노름(절댓값의 합을 계산)

### 2.2.3 가정 검사

가정을 나열하고 검사해보는 것이 좋음

## 2.3 데이터 가져오기

### 2.3.1 작업환경 만들기

### 2.3.2 데이터 다운로드

```python
import os
import tarfile
import urllib
import pandas as pd

download_root="https://raw.githubusercontent.com/rickiepark/handson-ml2/master/"
housing_path=os.path.join("datasets","housing")
housing_url=download_root+"datasets/housing/housing.tgz"

#코드를 추출하는 함수
def fetch_housing_data(house_url=housing_url, house_path=housing_path):
    #현재 작업공간에 datasets/housing 디렉터리를 만들고
    os.makedir(house_path, exist_ok=True)
    #housing.tgz파일을 내려받고
    tgz_path=os.path.join(house_path, "housing.tgz")
    urllib.request.urlretrieve(house_url, tgz_path)
    # 같은 디렉터리에 압축을 풀어 housing.csv파일을 만듦
    housing_tgz=tarfile.open(tgz_path)
    housing_tgz.extractall(path=house_path)
    housing_tgz.close()

#데이터 읽어들이는 함수
def load_housing_data(house_path=housing_path):
    #모든 데이터를 담은 판다스의 데이터 프레임 객체를 반환
    csv_path=os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
```

### 2.3.2 데이터 구조 훑어보기

```python
import matplotlib.pyplot as plt

'''
longitude : 경도
latitude : 위도
housing_median_age : 주택 평균 연식
total_rooms : 방의 갯수
total_bedrooms : 침실의 갯수
population : 인구
households : 가구
median_income : 평균 수입
median_house_value : 주택 평균 가치
ocean_proximity : 바다에 근접한 정도
'''
housing=load_housing_data()
housing.head()  #head()->처음 다섯행 확인하는 함수
housing.info()  #info()->데이터에 대한 간략한 설명(전체 행 수, 각 특성의 데이터 타입, 널이 아닌 값의 개수 확인)
housing["ocean_proximity"].value_counts()   #value_counts()->어떤 카테고리가 있고 각 카테고리마다 인스턴스가 몇 개 있는지 확인
housing.describe()  #describe()->숫자형 특성의 요약 정보

#히스토그램
housing.hist(bins=50, figsize=(20,15))
plt.show()
```

### 2.3.4  테스트 세트 만들기

데이터 일부를 미리 테스트 세트로 떼어두기, 무작위로 데이터셋의 20%정도를 떼어둠

Why?테스트 세트로 일반화 오차를 추정하면 매우 낙관적인 추정이 되며 시스템을 론칭했을 때 기대한 성능이 나오지 않을 것=>데이터 스누핑(data snooping)편향

```python
import numpy as np
'''
프로그램을 다시 실행하면 다른 테스트 세트가 생성됨->여러번하면 전체 데이터셋을 보게됨
해결법1. 처음 실행에서 테스트 세트를 저장하고 다음번 실행에서 이를 불러들이는것
해결법2. 항상 같은 난수 인데긋가 생성되도록 np.random.permutation()을 호출하기 전 난수 발생기의 초깃값을 지정
하지만 해결법1,2 둘다 다음번에 업데이트된 데이터셋을 사용하려면 문제가 됨
def split_train_test(data, test_ratio):
    shuffled_indices=np.random.permutation(len(data))
    test_set_size=int(len(data)*test_ratio)
    test_indices=shuffled_indices[:test_set_size]
    train_indices=shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set=split_train_test(housing.0.2)
print(len(train_set))
print(len(test_set))
'''

'''
위의 코드의 문제점을 해결한 코드
샘플의 식별자를 사용하여 테스트 세트로 보낼지 말지 정하는 것
'''
from zlib import crc32

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier))&0xffffffff<test_ratio*2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids=data[id_column]
    in_test_set=ids.apply(lambda id_:test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id=housing.reset_index()   #'index'열이 추가된 데이터 프레임 반환
train_set, test_set=split_train_test_by_id(housing_with_id, 0.2, "index")

#사이킷런으로 데이터셋을 여러 서브셋으로 나눌수도 있음
from sklearn.model_selection import train_test_split
train_set, test_set=train_test_split(housing, test_size=0.2, random_state=42)

'''
위의 방법들은 무작위로 샘플을 뽑는 것
데이터셋이 충분히 크다면 문제가 없지만 그렇지 않다면 샘플링 편향이 생길 가능성
=>계층적 샘플링을 통해 비율을 유지해야함
예를 들어 중간 소득이 예측에 중요한 변수라고 치면 이를 5개의 카테고리로 나눔
0~1.5/1.5~3.0/3.0~4.5/4.5~6.0/6.0~
'''
housing["income_cat"]=pd.cut(housing["median_income"],
                             bins=[0.,1.5,3.0,4.5,6.,np.inf],
                             labels=[1,2,3,4,5])
housing["income_cat"].hist()

#소득 카테고리를 위에서 나눴으므로 계층 샘플링
from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]
strat_test_set["income_cat"].value_counts()/len(strat_test_set)

#income_cat특성삭제해서 데이터를 원래대로 되돌림
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
```

## 2.4 데이터 이해를 위한 탐색과 시각화

훈련 세트만 탐색, 훈련 세트를 손상시키지 않기 위해 복사본 만들어 사용

```python
housing=strat_train_set.copy()
```



### 2.4.1지리적 데이터 시각화

지리정보(위도, 경도)가 있으니 모든 구역을 산점도로 만들어 데이터 시각화

```python
housing.plot(kind="scatter", x="longitude", y="latitude")
```

위의 시각화는 지역을 잘 나타내지만 특별한 패턴을 찾기는 힘듦

alpha옵션을 0.1로 주면 포인트가 밀집된 영역을 잘 보여줌

```python
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
```

원의 반지름은 구역의 인구를 나타내고(s) 색상은 가격을 나타냄(c)

미리 정의된 컬러 맵 중 파란색(낮은 가격)에서 빨간색(높은 가격)까지 범위를 가지는 jet사용(cmap)

```python
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["popilation"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,)
plt.legend()
```

시각화된 그래프를 보면 주택가격은 지역과 인구 밀도에 관련이 매우 크다는 사실을 알 수 있음, 군집 알고리즘을 사용해 주요 군집을 찾고 군집의 중심까지의 거리를 재는 특성 추가가능

### 2.4.2 상관관계 조사

1. 표준 상관계수(standart correlation coefficient, 피어슨의 r)

데이터 셋이 너무 크지 않으므로 모든 특성 간의 표준 상관계수(standart correlation coefficient, 피어슨의 r)를 corr()메서드를 이용해 계산

```python
corr_matrix=housing.corr()
```

상관관계 범위는 -1~1까지

1에 가까우면 강한 양의 상관관계를 가진다는 뜻 하나가 올라가면 다른 하나도 같이 오름=>비례관계

-1에 가까우면 강한 음의 상관관계를 가진다는 뜻 하나가 올라가면 다른 하나는 내려감=>반비례관계

0에 가까우면 선형적인 상관관계가 없다는 뜻

*상관계수는 선형적인 상관관계만 측정가능, 비선형적인 관계는 측정 불가

2. scatter_matrix

숫자형 특성 사이에 산점도를 그려주는 판다스의 scatter_matrix함수 사용

```python
from pandas.plotting import scatter_matrix
attributes=["median_house_value", "median_income", 
            "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12,8))
```

### 2.4.3 특성 조합으로 실험

