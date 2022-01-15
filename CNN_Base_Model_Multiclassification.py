# -*- coding: utf-8 -*-
"""CNN Base Modeling_Multiclassification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wQ5PMqNUaoxh_59z4mF84ajCJYVwUpvt
"""

!pwd

# Commented out IPython magic to ensure Python compatibility.
# %cd drive/MyDrive/Colab\ Notebooks

import numpy as np

data = np.load("./data/teamFace.npz")

X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)

# 모델링

from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Flatten
from tensorflow.keras.optimizers import Adam

# CNN Layer
from tensorflow.keras.layers import Conv2D, MaxPool2D # Conv: 특징 추출, Maxpol: 필터링 
# Con1~3D , MaxPool1~3D

# 모델설계
CNN_model = Sequential()
CNN_model.add(InputLayer(input_shape=(480,480,3))) # 입력층 input_shape : 단 한 개의 데이터 차원에만 맞춰줘도 됨 3 : RGB
# 특징 추출부
CNN_model.add(Conv2D(kernel_size=(3,3), # 돋보기 크기 : 특징 추출 :  3*3 픽셀 범위를 변형시켜 가며 특징을 관찰 / 커널 단위 : 홀수 단위를 추천
                     filters=64, # 관찰 돋보기 개수 : 64개의 특징을 추출
                     activation = 'relu',
                     strides = 1, # strides = 3 : 3만큼 건너 뛰어서 특징 추출 / default 1 / (2,2) 가로세로 2칸씩 건너 뛴다
                    padding = "valid")) # default = "valid" , "same" : 크기가 줄어들지 않는다 
                    # padding : 외각이나 주변부분의 특징을 추출하는 데에 활용되며 최근에는 많이들 사용한다
# 필터링
CNN_model.add(MaxPool2D())  

CNN_model.add(Conv2D(kernel_size=(3,3),filters=128, activation = 'relu')) # 항아리 모양
CNN_model.add(MaxPool2D())

CNN_model.add(Conv2D(kernel_size=(3,3),filters=64, activation = 'relu')) 
CNN_model.add(MaxPool2D()) 

# 분류기
# flatten layer : 추출된 주요 특징을 전결합층에 전달하기 위해 1차원 자료로 바꿔주며, 이미지 형태의 데이터를 배열형태로 flatten하게 만들어준다
CNN_model.add(Flatten()) # 데이터 평탄화
CNN_model.add(Dense(64,activation='relu'))
CNN_model.add(Dense(128,activation='relu'))
CNN_model.add(Dense(64,activation='relu'))
CNN_model.add(Dense(3,activation='softmax')) # 출력층 / softmax : 총합이 항상 1이 되게하여 사용자로 하여금 % 를 쉽게 확인할 수 있도록 한다

# 2. 모델 학습/평가 방법
CNN_model.compile(loss = "sparse_categorical_crossentropy",# 확률 정보로 안 바꾼 상태라면 sparse_categorical_crossentropy
                  optimizer = Adam(lr = 0.00001),
                  metrics = "accuracy")
CNN_model.summary()

# 3. 모델 저장 설계
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# 체크포인트 저장
# 모델이 저장될 경로를 문자열로 저장
save_path = "./model/face/model_{epoch:03d}_{val_accuracy:.4f}.hdf5"  # d : 10진수 정수형태  / 03 : 3자리 수로 표현 / .4f : 소수점 4째 자리까지 표현 / {} 대소문자 확인
mckp = ModelCheckpoint(filepath = save_path,
                                                  monitor = "val_accuracy",
                                                    save_best_only = True, # 최고값을 갱신했을 때만 저장해라
                                                        verbose = 1)

# 조기 학습 중단
early = EarlyStopping(monitor = "val_accuracy",
                                          patience = 1)  # patience : ?번 만큼 봐주는 데 개선이 안 되면 중단!

# 4. 모델 학습 및 결과 시각화
CNN_h = CNN_model.fit(X_train, y_train,
                      validation_data = (X_val, y_val),
                      epochs = 50,
                     callbacks = [mckp, early]) # 체크포인트, 조기 학습 중단 설정)