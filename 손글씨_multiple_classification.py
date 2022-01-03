# -*- coding: utf-8 -*-
from tensorflow.keras.datasets import mnist # 손글씨 모듈
import numpy as np
data = mnist.load_data()

# 이중구조 [[]] train, test
print(len(data[0]))
print(len(data[1]))
# 이중튜플로 나누기
(X_train, y_train),(X_test,y_test) = data

# 튜플예시
a,b = 1,2

#데이터 모양 확인 / 사진 6만장 28px*28px
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# 데이터 시각화
import matplotlib.pyplot as plt
plt.imshow(X_train[15000], cmap="gray")
print(f"정답 : {y_train[15000]}") # 정답 분류
print(f"정답 유니크 : {np.unique(y_train)}")
print(f"유니크 카운트 : {np.bincount(y_train)}")

plt.hist(X_train[0])
plt.show()

# 모델 설계 준비
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.keras import Sequential # 핵심
from tensorflow.keras.layers import Dense, InputLayer, Flatten
from tensorflow.keras.utils import to_categorical # 다중분류

## 다중분류 시 정답데이터를 반드시 확률정보로 바꾸어야 학습을 시킬 수 있다. *****
print(f"정답 :{y_train[:10]}")
print(f"{to_categorical(y_train[:10])}") # 원핫인코딩과 비슷한 구조 / 10 종류 = 10 개의 선 = 10 개의 확률

y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

# 모델 구조 설계
mnist_model = Sequential()
mnist_model.add(InputLayer(input_shape=(28,28))) # 서비스 할 때 사용자가 1차원으로 펴서 넣어야 하는 불편함이 생기기에 모델 중간에 1차원으로 바꾸는 구조로 짠다.(편의성)
# CNN 2차원 데이터를 처리하는 알고리즘이 중간에 들어온다
mnist_model.add(Flatten()) # 데이터를 1차원으로 평평하게 만들어주는 레이어
mnist_model.add(Dense(32, activation="sigmoid"))
mnist_model.add(Dense(64, activation="sigmoid"))
mnist_model.add(Dense(64, activation="sigmoid"))
mnist_model.add(Dense(32, activation="sigmoid"))
# 출력층 / 다중분류 시 유니크 값에 해당하는 총 10개의 확률값이 필요하다.
mnist_model.add(Dense(10, activation="softmax")) # softmax : 현재/전체 데이터 비율 >> 사람이 보기 쉽게 합산을 1.0 으로 만들어 줌

# 학습/평가 방법 설정
mnist_model.compile(loss="categorical_crossentropy", # 다중분류용 손실함수 binary_crossentrophy 와 원리 동일, 개수만 늘어난 형태
                    optimizer="Adam", # default : GD / Adam : 최신 버전
                    metrics=["accuracy"]) # metrics : 정확도를 보여 줌

# 학습
mnist_history = mnist_model.fit(X_train, y_train_one_hot,
                validation_split=0.3, # 학습과 검증 분리 7:3
                epochs=10)

# 시각화
plt.figure(figsize=(15,5))
plt.plot(mnist_history.history["loss"],label="loss")
plt.plot(mnist_history.history["val_loss"],label="val_loss") # 과대적합 여부를 확인하기 위해 검증 데이터도 그래프 시각화
plt.legend()
plt.show()

# 모델 평가
mnist_model.evaluate(X_test, y_test_one_hot)
