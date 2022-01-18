# -*- coding: utf-8 -*-
"""TimeArray_RNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1quurOCFo6ueVy9CtJLJCkSih6YnEkkw_

# 전기 사용량을 예측해보자~
"""

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %cd ./drive/MyDrive/Colab Notebooks
!pwd

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 전기 사용량, 온도 데이터 per 1hour
data = pd.read_csv("./data/energy.csv")
print(data.shape)
data.tail() # 데이터 끝부분 확인 <-> head()

energy = pd.read_csv("./data/energy.csv", parse_dates=["timestamp"], index_col="timestamp") # timestamp 를 날짜 타입으로 읽으며, index로 설정한다. 시간을 계산하기에 용이하다.
energy.head()

# 필요없는 온도 값 삭제
del energy["temp"]

plt.figure(figsize=(10,5)) # 가로가 좀 더 길게~
plt.plot(energy)
plt.show()

"""# 시간의 흐름을 판단해야 하기에 데이터를 랜덤 샘플링 하지 않아야 한다.
- (train, validation, text)
# 스케일링
- 딥러닝 학습 시 최적점을 찾아가는 속도 개선을 위해 컬럼 스케일링 진행
- RNN 데이터 구조로 변경 (sample, time step, feature)
"""

from sklearn.preprocessing import MinMaxScaler# 최대값과 최소값을 이용해서 스케일링 : 0~1 사이값 객체

scaler = MinMaxScaler() # 인스턴스화
scaler.fit(energy) # 현재 데이터의 최소값, 최대값 추출
energy["load"] = scaler.transform(energy) #스케일링 후 load 컬럼에 덮어씌우기
energy.head()

"""### RNN 으로 학습하기 좋게 데이터를 바꿔보자
- 과거 6시간을 학습하고 다음 1시간을 정답 y로 학습하자
"""

energy_shifted = energy.copy() # 원본 데이터 그~대로 복사
energy_shifted["y+1"] = energy_shifted["load"].shift(-1, freq="H") # shifted 밀다 : 데이터를 미는 것! +1 한 칸만 밀어라 -1 한 칸만 당겨라 , freq : 어떤 주기로 밀래? = "시간기준 H"
energy_shifted

# load_t 5 ~ 0 : 학습 데이터
# y+1 : 정답 데이터
energy_shifted["load_t-5"] = energy_shifted["load"].shift(5, freq="H") # 아래로 5번 밀기 
energy_shifted["load_t-4"] = energy_shifted["load"].shift(4, freq="H") 
energy_shifted["load_t-3"] = energy_shifted["load"].shift(3, freq="H") 
energy_shifted["load_t-2"] = energy_shifted["load"].shift(2, freq="H") 
energy_shifted["load_t-1"] = energy_shifted["load"].shift(1, freq="H") 
energy_shifted["load_t-0"] = energy_shifted["load"].shift(0, freq="H") 

# timestamp 가 6이기 때문에 5시부터 학습을 시작한다.
energy_shifted.head(10) # 앞 10행 확인!

#총 데이터가 약 2만개, 손실하는 데이터 5개 뿐!

energy_shifted.dropna(inplace=True) # inplace : 현재 데이터에 적용 및 저장까지 할래? True
energy_shifted.head(10)

"""### 데이터 나누기
- 2012~ 2014
"""

validation_start = "2014-09-01 00:00:00" # 검증 데이터 시작
test_start = "2014-11-01 00:00:00"

train = energy_shifted[energy_shifted.index < validation_start] # 불리언 인덱싱
validation = energy_shifted[(energy_shifted.index > validation_start) & (energy_shifted .index < test_start)] # 검증보다 크고 테스트보다 작은 검증 데이터 색출
test = energy_shifted[energy_shifted.index > test_start] # 테스트 날짜보다 큰 거 : 테스 날짜보다 나중에 온 것

plt.figure(figsize=(10,5))
plt.plot(train["load"])
plt.plot(validation["load"])
plt.plot(test["load"])
plt.show()

"""## 데이터 분리"""

# 현재 feature 수가 빠져 있음
X_train = train.loc[:, "load_t-5" : "load_t-0"] # 문제 데이터
y_train = train["y+1"] # 정답 데이터
X_val = validation.loc[:, "load_t-5" : "load_t-0"]
y_val = validation["y+1"]
X_test = test.loc[:, "load_t-5" : "load_t-0"]
y_test = test["y+1"] 

# 2차원
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)

"""### RNN 학습을 위한 2 >> 3 차원 변경
- 과거5시간, 과거4 시간, 과거 3시간, 과거 2시간 , 과거 1시간
"""

# 샘플 개수, timestamp, feature 개수 >> 3차원
# feature 순환하는 한 개의 퍼셉트론에 몇 개의 정보를 동시에 넣을지?
X_train = X_train.values.reshape(23371, 6, 1) # DataFrame 에는 reshape 가 없다. 따라서 values 를 거쳐 사용한다.
X_val = X_val.values.reshape(1463, 6, 1)
X_test = X_test.values.reshape(1462, 6, 1)

"""## 모델링 및 학습"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer, Dense, LSTM # LSTM : 순환하는 RNN 레이어

model1 = Sequential()
model1.add(InputLayer(input_shape=(6,1))) # input_shape : 샘플은 의미 없고 들어가는 한 개의 데이터에 관해 입력!
model1.add(LSTM(32, return_sequences=True))
model1.add(LSTM(16))
model1.add(Dense(1)) # 수치값을 예측하기에 데이터 그대로 보내고

model1.compile(loss="mean_squared_error", # 수치값이기에 MSE
              optimizer = "Adam") # 수치 데이터이기에 metrics 사용하지 않는다.

h1 = model1.fit(X_train, y_train,
           validation_data = (X_val, y_val), # 우리가 직접 쪼개놓았기에 내장함수를 사용하지 않는다.
          epochs = 20)

plt.figure(figsize=(15,5))
plt.plot(h1.history["loss"], label="loss")
plt.plot(h1.history["val_loss"], label="val_loss")
plt.legend()
plt.show()

"""## 모델 예측 및 결과 시각화"""

result = test[["y+1"]] # 테스트 데이터의 정답 추출
result["pre"] = model1.predict(X_test)
result

"""# 스케일링 전의 데이터 단위로 바꾸기 : inverse_scaling  """

result[["y+1", "pre"]] = scaler.inverse_transform(result)
result

plt.figure(figsize=(20,10))
plt.plot(result["pre"], label = "predict")
plt.plot(result["y+1"], label = "actual")
plt.legend()
plt.show()

"""# 온도 특성도 넣어서 예측해보자"""

# 데이터 가지고 오면서 동시에 파싱하기
energy = pd.read_csv("./data/energy.csv", parse_dates=["timestamp"], index_col="timestamp") # timestamp 를 날짜 타입으로 읽으며, index로 설정한다. 시간을 계산하기에 용이하다.

load_scaler = MinMaxScaler()
temp_scaler= MinMaxScaler()

load_scaler.fit(energy[["load"]]) # load 의 최대, 최소 값 확인
temp_scaler.fit(energy[["temp"]]) # temp 의 최대, 최소 값 확인

# 0~1 사이 값으로 스케일링
energy["load"] = load_scaler.transform(energy[["load"]])
energy["temp"] = temp_scaler.transform(energy[["temp"]])

energy.head(10)

# y+1 정답 특성 shift 밀어서 만들어주기~
energy_shifted = energy.copy() # 원본 카피
energy_shifted["y+1"] = energy_shifted["load"].shift(-1, freq="H")
energy_shifted.head(10)

# load, temp 를 번갈아 가면서 해야 feature 값을 2로 설정 하는 것만으로도 간편하게 데이터 분리가 2쌍씩 나누어질 수 있다.
energy_shifted["load_t-5"] = energy_shifted["load"].shift(5, freq="H")
energy_shifted["temp_t-5"] = energy_shifted["temp"].shift(5, freq="H")
energy_shifted["load_t-4"] = energy_shifted["load"].shift(4, freq="H")
energy_shifted["temp_t-4"] = energy_shifted["temp"].shift(4, freq="H")
energy_shifted["load_t-3"] = energy_shifted["load"].shift(3, freq="H")
energy_shifted["temp_t-3"] = energy_shifted["temp"].shift(3, freq="H")
energy_shifted["load_t-2"] = energy_shifted["load"].shift(2, freq="H")
energy_shifted["temp_t-2"] = energy_shifted["temp"].shift(2, freq="H")
energy_shifted["load_t-1"] = energy_shifted["load"].shift(1, freq="H")
energy_shifted["temp_t-1"] = energy_shifted["temp"].shift(1, freq="H")
energy_shifted["load_t-0"] = energy_shifted["load"].shift(0, freq="H")
energy_shifted["temp_t-0"] = energy_shifted["temp"].shift(0, freq="H")

# 과거에 대한 온도 값이 없는 경우가 많다
energy_shifted.head(10)

# 온도 결측치 제거
energy_shifted.dropna(inplace=True) # inplace : 실행 후 저장 여부

# train, validation, test 분리
validation_start = "2014-09-01 00:00:00" # 검증 데이터 시작
test_start = "2014-11-01 00:00:00"

train = energy_shifted[energy_shifted.index < validation_start] # 불리언 인덱싱
validation = energy_shifted[(energy_shifted.index > validation_start) & (energy_shifted .index < test_start)] # 검증보다 크고 테스트보다 작은 검증 데이터 색출
test = energy_shifted[energy_shifted.index > test_start] # 테스트 날짜보다 큰 거 : 테스 날짜보다 나중에 온 것

# 현재 feature 수가 빠져 있음
X_train = train.loc[:, "load_t-5" : "temp_t-0"] # 훈련용 문제 데이터
y_train = train["y+1"] # 훈련용 정답 데이터
X_val = validation.loc[:, "load_t-5" : "temp_t-0"] # 검증용 문제 데이터
y_val = validation["y+1"] # 검증용 정답 데이터
X_test = test.loc[:, "load_t-5" : "temp_t-0"] # 테스트용 문제 데이터
y_test = test["y+1"] # 테스트용 정답 데이터

# 2차원
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)

X_train.head(10)

# reshape
#샘플 개수, timestamp, feature 개수 >> 3차원
# feature 순환하는 한 개의 퍼셉트론에 몇 개의 정보를 동시에 넣을지?
X_train = X_train.values.reshape(23371, 6, 2) # DataFrame 에는 reshape 가 없다. 따라서 values 를 거쳐 사용한다.
X_val = X_val.values.reshape(1463, 6,  2)
X_test = X_test.values.reshape(1462, 6, 2)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer, Dense, LSTM # LSTM : 순환하는 RNN 레이어

model2 = Sequential()
model2.add(InputLayer(input_shape=(6,2))) # input_shape : 샘플은 의미 없고 들어가는 한 개의 데이터에 관해 입력!
model2.add(LSTM(128, return_sequences=True))
model2.add(LSTM(256, return_sequences=True))
model2.add(LSTM(512, return_sequences=True))
model2.add(LSTM(1024, return_sequences=True))
model2.add(LSTM(2048, return_sequences=True))
model2.add(LSTM(4096, return_sequences=True))
model2.add(LSTM(2048, return_sequences=True))
model2.add(LSTM(1024, return_sequences=True))
model2.add(LSTM(512, return_sequences=True))
model2.add(LSTM(256, return_sequences=True))
model2.add(LSTM(128, return_sequences=True))
model2.add(LSTM(64, return_sequences=True))
model2.add(LSTM(32, return_sequences=True))
model2.add(LSTM(16))
model2.add(Dense(1)) # 수치값을 예측하기에 데이터 그대로 보내고

model2.compile(loss="mean_squared_error", # 수치값이기에 MSE
              optimizer = "Adam") # 수치 데이터이기에 metrics 사용하지 않는다.

h2 = model2.fit(X_train, y_train,
           validation_data = (X_val, y_val), # 우리가 직접 쪼개놓았기에 내장함수를 사용하지 않는다.
          epochs = 20)

plt.figure(figsize=(20,10))
plt.plot(result["pre"], label = "predict")
plt.plot(result["y+1"], label = "actual")
plt.legend()
plt.show()

