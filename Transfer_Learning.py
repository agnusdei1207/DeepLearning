# -*- coding: utf-8 -*-

##### npz 로딩
"""

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %cd ./drive/MyDrive/Colab\ Notebooks

import numpy as np
data = np.load('./data/animal/animals.npz')
X_train,y_train = data['X_train'], data['y_train']
X_val,y_val = data['X_val'], data['y_val']

"""##### VGG16 모델 불러오기"""

from tensorflow.keras.applications import VGG16

vgg16_model = VGG16(input_shape=(224,224,3),
                    include_top=False,
                    weights='imagenet')

vgg16_model.summary()

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

cnn_model = Sequential()
cnn_model.add(vgg16_model)
cnn_model.add(Flatten())
cnn_model.add(Dense(128,activation='relu'))
cnn_model.add(Dense(64,activation='relu'))
cnn_model.add(Dense(3,activation='softmax'))

cnn_model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(lr=0.0001),
                  metrics='accuracy')

save_path = "./model/animal/model_{epoch:03d}_{val_accuracy:.4f}.hdf5"
mckp = ModelCheckpoint(filepath=save_path,
                       monitor="val_accuracy",
                       save_best_only=True,
                       verbose=1)
early = EarlyStopping(monitor="val_accuracy",
                      patience=10)

CNN_h = cnn_model.fit(X_train,y_train,
                      validation_data=(X_val,y_val),
                      epochs=500,
                      callbacks=[mckp,early])

"""##### 미세조정 방식"""

vgg16_model2 = VGG16(input_shape=(224,224,3),
                     include_top=False, weights='imagenet')

vgg16_model2.summary()

for layer in vgg16_model2.layers :
  if layer.name == 'block5_conv3' :
    layer.trainable = True # 학습 가능
  else :
    layer.trainable = False # 학습 불가능

vgg16_model2.summary()

model2 = Sequential()
model2.add(vgg16_model2)
model2.add(Flatten())
model2.add(Dense(128, activation='relu'))
model2.add(Dense(64, activation='relu'))
model2.add(Dense(3, activation='softmax'))
model2.compile(loss='sparse_categorical_crossentropy',
               optimizer=Adam(learning_rate=0.0001),
               metrics=['accuracy'])

save_path = "./model/animal/model2_{epoch:03d}_{val_accuracy:.4f}.hdf5"
mckp = ModelCheckpoint(filepath=save_path,
                       monitor="val_accuracy",
                       save_best_only=True,
                       verbose=1)
early = EarlyStopping(monitor="val_accuracy",
                      patience=10)

CNN_h = model2.fit(X_train,y_train,
                      validation_data=(X_val,y_val),
                      epochs=100,
                      callbacks=[mckp,early])

"""#### 모델로딩"""

from tensorflow.keras.models import load_model

m = load_model('./model/animal/model2_007_0.8342.hdf5')

m.trainable=True

m.summary()

