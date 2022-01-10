{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "ex09 전이학습.ipynb",
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "##### npz 로딩"
      ],
      "metadata": {
        "id": "7qXxzusXCLug"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Vv_THwbDB3ZO",
        "outputId": "b115b0a9-33da-440d-8811-88e862842b97"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "/content/drive/MyDrive/Colab Notebooks\n"
          ]
        }
      ],
      "source": [
        "%cd ./drive/MyDrive/Colab\\ Notebooks"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "data = np.load('./data/animal/animals.npz')\n",
        "X_train,y_train = data['X_train'], data['y_train']\n",
        "X_val,y_val = data['X_val'], data['y_val']"
      ],
      "metadata": {
        "id": "9HYZceztCOby"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "##### VGG16 모델 불러오기"
      ],
      "metadata": {
        "id": "KX3Iu4fsDvPs"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from tensorflow.keras.applications import VGG16"
      ],
      "metadata": {
        "id": "CK3CQ-aEDyNx"
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "vgg16_model = VGG16(input_shape=(224,224,3),\n",
        "                    include_top=False,\n",
        "                    weights='imagenet')"
      ],
      "metadata": {
        "id": "2tVodM0pD5nl"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "vgg16_model.summary()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "YlYUnlrfE9N8",
        "outputId": "270f100b-3fbb-4e6f-912d-dbf4a500b40e"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model: \"vgg16\"\n",
            "_________________________________________________________________\n",
            " Layer (type)                Output Shape              Param #   \n",
            "=================================================================\n",
            " input_2 (InputLayer)        [(None, 224, 224, 3)]     0         \n",
            "                                                                 \n",
            " block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792      \n",
            "                                                                 \n",
            " block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928     \n",
            "                                                                 \n",
            " block1_pool (MaxPooling2D)  (None, 112, 112, 64)      0         \n",
            "                                                                 \n",
            " block2_conv1 (Conv2D)       (None, 112, 112, 128)     73856     \n",
            "                                                                 \n",
            " block2_conv2 (Conv2D)       (None, 112, 112, 128)     147584    \n",
            "                                                                 \n",
            " block2_pool (MaxPooling2D)  (None, 56, 56, 128)       0         \n",
            "                                                                 \n",
            " block3_conv1 (Conv2D)       (None, 56, 56, 256)       295168    \n",
            "                                                                 \n",
            " block3_conv2 (Conv2D)       (None, 56, 56, 256)       590080    \n",
            "                                                                 \n",
            " block3_conv3 (Conv2D)       (None, 56, 56, 256)       590080    \n",
            "                                                                 \n",
            " block3_pool (MaxPooling2D)  (None, 28, 28, 256)       0         \n",
            "                                                                 \n",
            " block4_conv1 (Conv2D)       (None, 28, 28, 512)       1180160   \n",
            "                                                                 \n",
            " block4_conv2 (Conv2D)       (None, 28, 28, 512)       2359808   \n",
            "                                                                 \n",
            " block4_conv3 (Conv2D)       (None, 28, 28, 512)       2359808   \n",
            "                                                                 \n",
            " block4_pool (MaxPooling2D)  (None, 14, 14, 512)       0         \n",
            "                                                                 \n",
            " block5_conv1 (Conv2D)       (None, 14, 14, 512)       2359808   \n",
            "                                                                 \n",
            " block5_conv2 (Conv2D)       (None, 14, 14, 512)       2359808   \n",
            "                                                                 \n",
            " block5_conv3 (Conv2D)       (None, 14, 14, 512)       2359808   \n",
            "                                                                 \n",
            " block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0         \n",
            "                                                                 \n",
            "=================================================================\n",
            "Total params: 14,714,688\n",
            "Trainable params: 14,714,688\n",
            "Non-trainable params: 0\n",
            "_________________________________________________________________\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from tensorflow.keras import Sequential\n",
        "from tensorflow.keras.layers import Dense,Flatten\n",
        "from tensorflow.keras.optimizers import Adam\n",
        "from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping"
      ],
      "metadata": {
        "id": "-vyTwNNRFtqn"
      },
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn_model = Sequential()\n",
        "cnn_model.add(vgg16_model)\n",
        "cnn_model.add(Flatten())\n",
        "cnn_model.add(Dense(128,activation='relu'))\n",
        "cnn_model.add(Dense(64,activation='relu'))\n",
        "cnn_model.add(Dense(3,activation='softmax'))"
      ],
      "metadata": {
        "id": "kg1ybH8cF_qa"
      },
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn_model.compile(loss='sparse_categorical_crossentropy',\n",
        "                  optimizer=Adam(lr=0.0001),\n",
        "                  metrics='accuracy')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "5pxDmdfnF5jQ",
        "outputId": "3cbd7ccc-30a2-482e-b5ab-bd4467e7a47c"
      },
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/keras/optimizer_v2/adam.py:105: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.\n",
            "  super(Adam, self).__init__(name, **kwargs)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "save_path = \"./model/animal/model_{epoch:03d}_{val_accuracy:.4f}.hdf5\"\n",
        "mckp = ModelCheckpoint(filepath=save_path,\n",
        "                       monitor=\"val_accuracy\",\n",
        "                       save_best_only=True,\n",
        "                       verbose=1)\n",
        "early = EarlyStopping(monitor=\"val_accuracy\",\n",
        "                      patience=10)"
      ],
      "metadata": {
        "id": "RWB53o4OG9ID"
      },
      "execution_count": 12,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "CNN_h = cnn_model.fit(X_train,y_train,\n",
        "                      validation_data=(X_val,y_val),\n",
        "                      epochs=500,\n",
        "                      callbacks=[mckp,early])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "yQWiRH-lHFCo",
        "outputId": "8f66b253-d3b9-4744-a289-1564b2a4ab5d"
      },
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/500\n",
            "24/24 [==============================] - ETA: 0s - loss: 1.5856 - accuracy: 0.4556\n",
            "Epoch 00001: val_accuracy improved from -inf to 0.36898, saving model to ./model/animal/model_001_0.3690.hdf5\n",
            "24/24 [==============================] - 23s 420ms/step - loss: 1.5856 - accuracy: 0.4556 - val_loss: 1.2054 - val_accuracy: 0.3690\n",
            "Epoch 2/500\n",
            "24/24 [==============================] - ETA: 0s - loss: 1.1295 - accuracy: 0.4355\n",
            "Epoch 00002: val_accuracy improved from 0.36898 to 0.51337, saving model to ./model/animal/model_002_0.5134.hdf5\n",
            "24/24 [==============================] - 6s 247ms/step - loss: 1.1295 - accuracy: 0.4355 - val_loss: 0.9955 - val_accuracy: 0.5134\n",
            "Epoch 3/500\n",
            "24/24 [==============================] - ETA: 0s - loss: 0.9889 - accuracy: 0.5269\n",
            "Epoch 00003: val_accuracy did not improve from 0.51337\n",
            "24/24 [==============================] - 5s 214ms/step - loss: 0.9889 - accuracy: 0.5269 - val_loss: 0.9934 - val_accuracy: 0.4973\n",
            "Epoch 4/500\n",
            "24/24 [==============================] - ETA: 0s - loss: 0.8791 - accuracy: 0.5941\n",
            "Epoch 00004: val_accuracy improved from 0.51337 to 0.59358, saving model to ./model/animal/model_004_0.5936.hdf5\n",
            "24/24 [==============================] - 6s 248ms/step - loss: 0.8791 - accuracy: 0.5941 - val_loss: 0.8600 - val_accuracy: 0.5936\n",
            "Epoch 5/500\n",
            "24/24 [==============================] - ETA: 0s - loss: 0.7549 - accuracy: 0.6532\n",
            "Epoch 00005: val_accuracy improved from 0.59358 to 0.62567, saving model to ./model/animal/model_005_0.6257.hdf5\n",
            "24/24 [==============================] - 6s 248ms/step - loss: 0.7549 - accuracy: 0.6532 - val_loss: 0.8261 - val_accuracy: 0.6257\n",
            "Epoch 6/500\n",
            "24/24 [==============================] - ETA: 0s - loss: 0.6449 - accuracy: 0.7191\n",
            "Epoch 00006: val_accuracy improved from 0.62567 to 0.63102, saving model to ./model/animal/model_006_0.6310.hdf5\n",
            "24/24 [==============================] - 6s 254ms/step - loss: 0.6449 - accuracy: 0.7191 - val_loss: 0.7562 - val_accuracy: 0.6310\n",
            "Epoch 7/500\n",
            "24/24 [==============================] - ETA: 0s - loss: 0.5651 - accuracy: 0.7500\n",
            "Epoch 00007: val_accuracy improved from 0.63102 to 0.64706, saving model to ./model/animal/model_007_0.6471.hdf5\n",
            "24/24 [==============================] - 6s 249ms/step - loss: 0.5651 - accuracy: 0.7500 - val_loss: 0.7050 - val_accuracy: 0.6471\n",
            "Epoch 8/500\n",
            "24/24 [==============================] - ETA: 0s - loss: 0.4902 - accuracy: 0.7930\n",
            "Epoch 00008: val_accuracy improved from 0.64706 to 0.80749, saving model to ./model/animal/model_008_0.8075.hdf5\n",
            "24/24 [==============================] - 6s 249ms/step - loss: 0.4902 - accuracy: 0.7930 - val_loss: 0.6513 - val_accuracy: 0.8075\n",
            "Epoch 9/500\n",
            "24/24 [==============================] - ETA: 0s - loss: 0.3356 - accuracy: 0.8656\n",
            "Epoch 00009: val_accuracy did not improve from 0.80749\n",
            "24/24 [==============================] - 5s 215ms/step - loss: 0.3356 - accuracy: 0.8656 - val_loss: 0.7121 - val_accuracy: 0.7540\n",
            "Epoch 10/500\n",
            "24/24 [==============================] - ETA: 0s - loss: 0.3711 - accuracy: 0.8522\n",
            "Epoch 00010: val_accuracy did not improve from 0.80749\n",
            "24/24 [==============================] - 5s 213ms/step - loss: 0.3711 - accuracy: 0.8522 - val_loss: 0.6611 - val_accuracy: 0.7273\n",
            "Epoch 11/500\n",
            "24/24 [==============================] - ETA: 0s - loss: 0.4236 - accuracy: 0.8522\n",
            "Epoch 00011: val_accuracy did not improve from 0.80749\n",
            "24/24 [==============================] - 5s 213ms/step - loss: 0.4236 - accuracy: 0.8522 - val_loss: 0.6151 - val_accuracy: 0.7380\n",
            "Epoch 12/500\n",
            "24/24 [==============================] - ETA: 0s - loss: 0.2791 - accuracy: 0.8884\n",
            "Epoch 00012: val_accuracy did not improve from 0.80749\n",
            "24/24 [==============================] - 5s 213ms/step - loss: 0.2791 - accuracy: 0.8884 - val_loss: 0.5566 - val_accuracy: 0.7968\n",
            "Epoch 13/500\n",
            "24/24 [==============================] - ETA: 0s - loss: 0.1475 - accuracy: 0.9476\n",
            "Epoch 00013: val_accuracy did not improve from 0.80749\n",
            "24/24 [==============================] - 5s 213ms/step - loss: 0.1475 - accuracy: 0.9476 - val_loss: 0.8015 - val_accuracy: 0.7861\n",
            "Epoch 14/500\n",
            "24/24 [==============================] - ETA: 0s - loss: 0.2368 - accuracy: 0.8965\n",
            "Epoch 00014: val_accuracy did not improve from 0.80749\n",
            "24/24 [==============================] - 5s 213ms/step - loss: 0.2368 - accuracy: 0.8965 - val_loss: 0.6676 - val_accuracy: 0.7487\n",
            "Epoch 15/500\n",
            "24/24 [==============================] - ETA: 0s - loss: 0.0844 - accuracy: 0.9758\n",
            "Epoch 00015: val_accuracy did not improve from 0.80749\n",
            "24/24 [==============================] - 5s 213ms/step - loss: 0.0844 - accuracy: 0.9758 - val_loss: 0.7504 - val_accuracy: 0.8021\n",
            "Epoch 16/500\n",
            "24/24 [==============================] - ETA: 0s - loss: 0.0364 - accuracy: 0.9892\n",
            "Epoch 00016: val_accuracy improved from 0.80749 to 0.82353, saving model to ./model/animal/model_016_0.8235.hdf5\n",
            "24/24 [==============================] - 6s 253ms/step - loss: 0.0364 - accuracy: 0.9892 - val_loss: 0.7491 - val_accuracy: 0.8235\n",
            "Epoch 17/500\n",
            "24/24 [==============================] - ETA: 0s - loss: 0.0406 - accuracy: 0.9852\n",
            "Epoch 00017: val_accuracy did not improve from 0.82353\n",
            "24/24 [==============================] - 5s 213ms/step - loss: 0.0406 - accuracy: 0.9852 - val_loss: 0.7169 - val_accuracy: 0.8182\n",
            "Epoch 18/500\n",
            "24/24 [==============================] - ETA: 0s - loss: 0.0605 - accuracy: 0.9825\n",
            "Epoch 00018: val_accuracy did not improve from 0.82353\n",
            "24/24 [==============================] - 5s 213ms/step - loss: 0.0605 - accuracy: 0.9825 - val_loss: 0.8481 - val_accuracy: 0.7968\n",
            "Epoch 19/500\n",
            "24/24 [==============================] - ETA: 0s - loss: 0.0364 - accuracy: 0.9879\n",
            "Epoch 00019: val_accuracy did not improve from 0.82353\n",
            "24/24 [==============================] - 5s 213ms/step - loss: 0.0364 - accuracy: 0.9879 - val_loss: 1.2489 - val_accuracy: 0.7647\n",
            "Epoch 20/500\n",
            "24/24 [==============================] - ETA: 0s - loss: 0.0756 - accuracy: 0.9852\n",
            "Epoch 00020: val_accuracy did not improve from 0.82353\n",
            "24/24 [==============================] - 5s 213ms/step - loss: 0.0756 - accuracy: 0.9852 - val_loss: 0.7090 - val_accuracy: 0.7807\n",
            "Epoch 21/500\n",
            "24/24 [==============================] - ETA: 0s - loss: 0.0519 - accuracy: 0.9825\n",
            "Epoch 00021: val_accuracy improved from 0.82353 to 0.82888, saving model to ./model/animal/model_021_0.8289.hdf5\n",
            "24/24 [==============================] - 6s 252ms/step - loss: 0.0519 - accuracy: 0.9825 - val_loss: 0.6612 - val_accuracy: 0.8289\n",
            "Epoch 22/500\n",
            "24/24 [==============================] - ETA: 0s - loss: 0.0365 - accuracy: 0.9852\n",
            "Epoch 00022: val_accuracy did not improve from 0.82888\n",
            "24/24 [==============================] - 5s 214ms/step - loss: 0.0365 - accuracy: 0.9852 - val_loss: 0.9273 - val_accuracy: 0.7914\n",
            "Epoch 23/500\n",
            "24/24 [==============================] - ETA: 0s - loss: 0.0574 - accuracy: 0.9812\n",
            "Epoch 00023: val_accuracy did not improve from 0.82888\n",
            "24/24 [==============================] - 5s 213ms/step - loss: 0.0574 - accuracy: 0.9812 - val_loss: 1.2580 - val_accuracy: 0.7914\n",
            "Epoch 24/500\n",
            "24/24 [==============================] - ETA: 0s - loss: 0.1198 - accuracy: 0.9664\n",
            "Epoch 00024: val_accuracy did not improve from 0.82888\n",
            "24/24 [==============================] - 5s 214ms/step - loss: 0.1198 - accuracy: 0.9664 - val_loss: 0.4774 - val_accuracy: 0.8182\n",
            "Epoch 25/500\n",
            "24/24 [==============================] - ETA: 0s - loss: 0.0606 - accuracy: 0.9866\n",
            "Epoch 00025: val_accuracy did not improve from 0.82888\n",
            "24/24 [==============================] - 5s 213ms/step - loss: 0.0606 - accuracy: 0.9866 - val_loss: 0.8629 - val_accuracy: 0.8075\n",
            "Epoch 26/500\n",
            "24/24 [==============================] - ETA: 0s - loss: 0.0780 - accuracy: 0.9798\n",
            "Epoch 00026: val_accuracy improved from 0.82888 to 0.85561, saving model to ./model/animal/model_026_0.8556.hdf5\n",
            "24/24 [==============================] - 6s 257ms/step - loss: 0.0780 - accuracy: 0.9798 - val_loss: 0.6295 - val_accuracy: 0.8556\n",
            "Epoch 27/500\n",
            "24/24 [==============================] - ETA: 0s - loss: 0.0263 - accuracy: 0.9933\n",
            "Epoch 00027: val_accuracy did not improve from 0.85561\n",
            "24/24 [==============================] - 5s 213ms/step - loss: 0.0263 - accuracy: 0.9933 - val_loss: 1.1123 - val_accuracy: 0.7807\n",
            "Epoch 28/500\n",
            "24/24 [==============================] - ETA: 0s - loss: 0.0455 - accuracy: 0.9812\n",
            "Epoch 00028: val_accuracy improved from 0.85561 to 0.86096, saving model to ./model/animal/model_028_0.8610.hdf5\n",
            "24/24 [==============================] - 6s 253ms/step - loss: 0.0455 - accuracy: 0.9812 - val_loss: 0.9663 - val_accuracy: 0.8610\n",
            "Epoch 29/500\n",
            "24/24 [==============================] - ETA: 0s - loss: 0.0315 - accuracy: 0.9879\n",
            "Epoch 00029: val_accuracy did not improve from 0.86096\n",
            "24/24 [==============================] - 5s 214ms/step - loss: 0.0315 - accuracy: 0.9879 - val_loss: 0.7043 - val_accuracy: 0.8449\n",
            "Epoch 30/500\n",
            "24/24 [==============================] - ETA: 0s - loss: 0.0053 - accuracy: 0.9987\n",
            "Epoch 00030: val_accuracy did not improve from 0.86096\n",
            "24/24 [==============================] - 5s 214ms/step - loss: 0.0053 - accuracy: 0.9987 - val_loss: 0.8722 - val_accuracy: 0.8342\n",
            "Epoch 31/500\n",
            "24/24 [==============================] - ETA: 0s - loss: 7.4204e-04 - accuracy: 1.0000\n",
            "Epoch 00031: val_accuracy did not improve from 0.86096\n",
            "24/24 [==============================] - 5s 213ms/step - loss: 7.4204e-04 - accuracy: 1.0000 - val_loss: 0.8655 - val_accuracy: 0.8289\n",
            "Epoch 32/500\n",
            "24/24 [==============================] - ETA: 0s - loss: 2.3638e-04 - accuracy: 1.0000\n",
            "Epoch 00032: val_accuracy did not improve from 0.86096\n",
            "24/24 [==============================] - 5s 213ms/step - loss: 2.3638e-04 - accuracy: 1.0000 - val_loss: 0.8679 - val_accuracy: 0.8342\n",
            "Epoch 33/500\n",
            "24/24 [==============================] - ETA: 0s - loss: 1.3086e-04 - accuracy: 1.0000\n",
            "Epoch 00033: val_accuracy did not improve from 0.86096\n",
            "24/24 [==============================] - 5s 214ms/step - loss: 1.3086e-04 - accuracy: 1.0000 - val_loss: 0.8729 - val_accuracy: 0.8342\n",
            "Epoch 34/500\n",
            "24/24 [==============================] - ETA: 0s - loss: 9.9271e-05 - accuracy: 1.0000\n",
            "Epoch 00034: val_accuracy did not improve from 0.86096\n",
            "24/24 [==============================] - 5s 218ms/step - loss: 9.9271e-05 - accuracy: 1.0000 - val_loss: 0.8843 - val_accuracy: 0.8342\n",
            "Epoch 35/500\n",
            "24/24 [==============================] - ETA: 0s - loss: 8.0872e-05 - accuracy: 1.0000\n",
            "Epoch 00035: val_accuracy did not improve from 0.86096\n",
            "24/24 [==============================] - 5s 215ms/step - loss: 8.0872e-05 - accuracy: 1.0000 - val_loss: 0.8920 - val_accuracy: 0.8396\n",
            "Epoch 36/500\n",
            "24/24 [==============================] - ETA: 0s - loss: 6.8773e-05 - accuracy: 1.0000\n",
            "Epoch 00036: val_accuracy did not improve from 0.86096\n",
            "24/24 [==============================] - 5s 213ms/step - loss: 6.8773e-05 - accuracy: 1.0000 - val_loss: 0.8994 - val_accuracy: 0.8396\n",
            "Epoch 37/500\n",
            "24/24 [==============================] - ETA: 0s - loss: 5.9791e-05 - accuracy: 1.0000\n",
            "Epoch 00037: val_accuracy did not improve from 0.86096\n",
            "24/24 [==============================] - 5s 226ms/step - loss: 5.9791e-05 - accuracy: 1.0000 - val_loss: 0.9062 - val_accuracy: 0.8342\n",
            "Epoch 38/500\n",
            "24/24 [==============================] - ETA: 0s - loss: 5.2477e-05 - accuracy: 1.0000\n",
            "Epoch 00038: val_accuracy did not improve from 0.86096\n",
            "24/24 [==============================] - 5s 213ms/step - loss: 5.2477e-05 - accuracy: 1.0000 - val_loss: 0.9130 - val_accuracy: 0.8342\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "##### 미세조정 방식"
      ],
      "metadata": {
        "id": "H53P0g65rgcR"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "vgg16_model2 = VGG16(input_shape=(224,224,3),\n",
        "                     include_top=False, weights='imagenet')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "m075yityrgE1",
        "outputId": "1fe0c46e-573d-417f-c691-cc55ddaa9b5c"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5\n",
            "58892288/58889256 [==============================] - 0s 0us/step\n",
            "58900480/58889256 [==============================] - 0s 0us/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "vgg16_model2.summary()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "KekKn4Wpr4df",
        "outputId": "70f312fe-7d3c-4c20-d77c-8b30df2b1b72"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model: \"vgg16\"\n",
            "_________________________________________________________________\n",
            " Layer (type)                Output Shape              Param #   \n",
            "=================================================================\n",
            " input_1 (InputLayer)        [(None, 224, 224, 3)]     0         \n",
            "                                                                 \n",
            " block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792      \n",
            "                                                                 \n",
            " block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928     \n",
            "                                                                 \n",
            " block1_pool (MaxPooling2D)  (None, 112, 112, 64)      0         \n",
            "                                                                 \n",
            " block2_conv1 (Conv2D)       (None, 112, 112, 128)     73856     \n",
            "                                                                 \n",
            " block2_conv2 (Conv2D)       (None, 112, 112, 128)     147584    \n",
            "                                                                 \n",
            " block2_pool (MaxPooling2D)  (None, 56, 56, 128)       0         \n",
            "                                                                 \n",
            " block3_conv1 (Conv2D)       (None, 56, 56, 256)       295168    \n",
            "                                                                 \n",
            " block3_conv2 (Conv2D)       (None, 56, 56, 256)       590080    \n",
            "                                                                 \n",
            " block3_conv3 (Conv2D)       (None, 56, 56, 256)       590080    \n",
            "                                                                 \n",
            " block3_pool (MaxPooling2D)  (None, 28, 28, 256)       0         \n",
            "                                                                 \n",
            " block4_conv1 (Conv2D)       (None, 28, 28, 512)       1180160   \n",
            "                                                                 \n",
            " block4_conv2 (Conv2D)       (None, 28, 28, 512)       2359808   \n",
            "                                                                 \n",
            " block4_conv3 (Conv2D)       (None, 28, 28, 512)       2359808   \n",
            "                                                                 \n",
            " block4_pool (MaxPooling2D)  (None, 14, 14, 512)       0         \n",
            "                                                                 \n",
            " block5_conv1 (Conv2D)       (None, 14, 14, 512)       2359808   \n",
            "                                                                 \n",
            " block5_conv2 (Conv2D)       (None, 14, 14, 512)       2359808   \n",
            "                                                                 \n",
            " block5_conv3 (Conv2D)       (None, 14, 14, 512)       2359808   \n",
            "                                                                 \n",
            " block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0         \n",
            "                                                                 \n",
            "=================================================================\n",
            "Total params: 14,714,688\n",
            "Trainable params: 14,714,688\n",
            "Non-trainable params: 0\n",
            "_________________________________________________________________\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "for layer in vgg16_model2.layers :\n",
        "  if layer.name == 'block5_conv3' :\n",
        "    layer.trainable = True # 학습 가능\n",
        "  else :\n",
        "    layer.trainable = False # 학습 불가능"
      ],
      "metadata": {
        "id": "ykpcJKsPsNnO"
      },
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "vgg16_model2.summary()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "65CDxXe2s6OB",
        "outputId": "3f87b198-d3f4-4993-d86f-086dfcfb4562"
      },
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model: \"vgg16\"\n",
            "_________________________________________________________________\n",
            " Layer (type)                Output Shape              Param #   \n",
            "=================================================================\n",
            " input_1 (InputLayer)        [(None, 224, 224, 3)]     0         \n",
            "                                                                 \n",
            " block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792      \n",
            "                                                                 \n",
            " block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928     \n",
            "                                                                 \n",
            " block1_pool (MaxPooling2D)  (None, 112, 112, 64)      0         \n",
            "                                                                 \n",
            " block2_conv1 (Conv2D)       (None, 112, 112, 128)     73856     \n",
            "                                                                 \n",
            " block2_conv2 (Conv2D)       (None, 112, 112, 128)     147584    \n",
            "                                                                 \n",
            " block2_pool (MaxPooling2D)  (None, 56, 56, 128)       0         \n",
            "                                                                 \n",
            " block3_conv1 (Conv2D)       (None, 56, 56, 256)       295168    \n",
            "                                                                 \n",
            " block3_conv2 (Conv2D)       (None, 56, 56, 256)       590080    \n",
            "                                                                 \n",
            " block3_conv3 (Conv2D)       (None, 56, 56, 256)       590080    \n",
            "                                                                 \n",
            " block3_pool (MaxPooling2D)  (None, 28, 28, 256)       0         \n",
            "                                                                 \n",
            " block4_conv1 (Conv2D)       (None, 28, 28, 512)       1180160   \n",
            "                                                                 \n",
            " block4_conv2 (Conv2D)       (None, 28, 28, 512)       2359808   \n",
            "                                                                 \n",
            " block4_conv3 (Conv2D)       (None, 28, 28, 512)       2359808   \n",
            "                                                                 \n",
            " block4_pool (MaxPooling2D)  (None, 14, 14, 512)       0         \n",
            "                                                                 \n",
            " block5_conv1 (Conv2D)       (None, 14, 14, 512)       2359808   \n",
            "                                                                 \n",
            " block5_conv2 (Conv2D)       (None, 14, 14, 512)       2359808   \n",
            "                                                                 \n",
            " block5_conv3 (Conv2D)       (None, 14, 14, 512)       2359808   \n",
            "                                                                 \n",
            " block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0         \n",
            "                                                                 \n",
            "=================================================================\n",
            "Total params: 14,714,688\n",
            "Trainable params: 2,359,808\n",
            "Non-trainable params: 12,354,880\n",
            "_________________________________________________________________\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "model2 = Sequential()\n",
        "model2.add(vgg16_model2)\n",
        "model2.add(Flatten())\n",
        "model2.add(Dense(128, activation='relu'))\n",
        "model2.add(Dense(64, activation='relu'))\n",
        "model2.add(Dense(3, activation='softmax'))\n",
        "model2.compile(loss='sparse_categorical_crossentropy',\n",
        "               optimizer=Adam(learning_rate=0.0001),\n",
        "               metrics=['accuracy'])"
      ],
      "metadata": {
        "id": "nyUCT-Lrtl3m"
      },
      "execution_count": 12,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "save_path = \"./model/animal/model2_{epoch:03d}_{val_accuracy:.4f}.hdf5\"\n",
        "mckp = ModelCheckpoint(filepath=save_path,\n",
        "                       monitor=\"val_accuracy\",\n",
        "                       save_best_only=True,\n",
        "                       verbose=1)\n",
        "early = EarlyStopping(monitor=\"val_accuracy\",\n",
        "                      patience=10)"
      ],
      "metadata": {
        "id": "w9fHx-heuS9l"
      },
      "execution_count": 13,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "CNN_h = model2.fit(X_train,y_train,\n",
        "                      validation_data=(X_val,y_val),\n",
        "                      epochs=100,\n",
        "                      callbacks=[mckp,early])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "KyOdpVk-ugXS",
        "outputId": "0d517e1a-61c0-48b0-ae94-54722fc107e0"
      },
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/100\n",
            "24/24 [==============================] - ETA: 0s - loss: 2.6430 - accuracy: 0.6868\n",
            "Epoch 00001: val_accuracy improved from -inf to 0.78075, saving model to ./model/animal/model2_001_0.7807.hdf5\n",
            "24/24 [==============================] - 8s 272ms/step - loss: 2.6430 - accuracy: 0.6868 - val_loss: 1.4145 - val_accuracy: 0.7807\n",
            "Epoch 2/100\n",
            "23/24 [===========================>..] - ETA: 0s - loss: 0.1642 - accuracy: 0.9674\n",
            "Epoch 00002: val_accuracy improved from 0.78075 to 0.80749, saving model to ./model/animal/model2_002_0.8075.hdf5\n",
            "24/24 [==============================] - 2s 97ms/step - loss: 0.1625 - accuracy: 0.9677 - val_loss: 1.2743 - val_accuracy: 0.8075\n",
            "Epoch 3/100\n",
            "23/24 [===========================>..] - ETA: 0s - loss: 0.0102 - accuracy: 0.9973\n",
            "Epoch 00003: val_accuracy improved from 0.80749 to 0.82888, saving model to ./model/animal/model2_003_0.8289.hdf5\n",
            "24/24 [==============================] - 2s 98ms/step - loss: 0.0101 - accuracy: 0.9973 - val_loss: 1.0147 - val_accuracy: 0.8289\n",
            "Epoch 4/100\n",
            "23/24 [===========================>..] - ETA: 0s - loss: 4.7044e-04 - accuracy: 1.0000\n",
            "Epoch 00004: val_accuracy did not improve from 0.82888\n",
            "24/24 [==============================] - 2s 80ms/step - loss: 4.6665e-04 - accuracy: 1.0000 - val_loss: 1.0006 - val_accuracy: 0.8235\n",
            "Epoch 5/100\n",
            "23/24 [===========================>..] - ETA: 0s - loss: 1.2613e-04 - accuracy: 1.0000\n",
            "Epoch 00005: val_accuracy did not improve from 0.82888\n",
            "24/24 [==============================] - 2s 80ms/step - loss: 1.2882e-04 - accuracy: 1.0000 - val_loss: 1.0066 - val_accuracy: 0.8289\n",
            "Epoch 6/100\n",
            "23/24 [===========================>..] - ETA: 0s - loss: 9.7168e-05 - accuracy: 1.0000\n",
            "Epoch 00006: val_accuracy did not improve from 0.82888\n",
            "24/24 [==============================] - 2s 79ms/step - loss: 9.6124e-05 - accuracy: 1.0000 - val_loss: 1.0093 - val_accuracy: 0.8289\n",
            "Epoch 7/100\n",
            "23/24 [===========================>..] - ETA: 0s - loss: 8.1891e-05 - accuracy: 1.0000\n",
            "Epoch 00007: val_accuracy improved from 0.82888 to 0.83422, saving model to ./model/animal/model2_007_0.8342.hdf5\n",
            "24/24 [==============================] - 2s 104ms/step - loss: 8.1180e-05 - accuracy: 1.0000 - val_loss: 1.0110 - val_accuracy: 0.8342\n",
            "Epoch 8/100\n",
            "23/24 [===========================>..] - ETA: 0s - loss: 6.9636e-05 - accuracy: 1.0000\n",
            "Epoch 00008: val_accuracy did not improve from 0.83422\n",
            "24/24 [==============================] - 2s 80ms/step - loss: 7.0562e-05 - accuracy: 1.0000 - val_loss: 1.0120 - val_accuracy: 0.8289\n",
            "Epoch 9/100\n",
            "23/24 [===========================>..] - ETA: 0s - loss: 6.3034e-05 - accuracy: 1.0000\n",
            "Epoch 00009: val_accuracy did not improve from 0.83422\n",
            "24/24 [==============================] - 2s 80ms/step - loss: 6.2561e-05 - accuracy: 1.0000 - val_loss: 1.0139 - val_accuracy: 0.8289\n",
            "Epoch 10/100\n",
            "23/24 [===========================>..] - ETA: 0s - loss: 5.6638e-05 - accuracy: 1.0000\n",
            "Epoch 00010: val_accuracy did not improve from 0.83422\n",
            "24/24 [==============================] - 2s 80ms/step - loss: 5.6030e-05 - accuracy: 1.0000 - val_loss: 1.0150 - val_accuracy: 0.8289\n",
            "Epoch 11/100\n",
            "23/24 [===========================>..] - ETA: 0s - loss: 5.0925e-05 - accuracy: 1.0000\n",
            "Epoch 00011: val_accuracy did not improve from 0.83422\n",
            "24/24 [==============================] - 2s 79ms/step - loss: 5.0618e-05 - accuracy: 1.0000 - val_loss: 1.0158 - val_accuracy: 0.8289\n",
            "Epoch 12/100\n",
            "23/24 [===========================>..] - ETA: 0s - loss: 4.6229e-05 - accuracy: 1.0000\n",
            "Epoch 00012: val_accuracy did not improve from 0.83422\n",
            "24/24 [==============================] - 2s 80ms/step - loss: 4.6350e-05 - accuracy: 1.0000 - val_loss: 1.0167 - val_accuracy: 0.8289\n",
            "Epoch 13/100\n",
            "23/24 [===========================>..] - ETA: 0s - loss: 4.2795e-05 - accuracy: 1.0000\n",
            "Epoch 00013: val_accuracy did not improve from 0.83422\n",
            "24/24 [==============================] - 2s 79ms/step - loss: 4.2391e-05 - accuracy: 1.0000 - val_loss: 1.0173 - val_accuracy: 0.8289\n",
            "Epoch 14/100\n",
            "23/24 [===========================>..] - ETA: 0s - loss: 3.9693e-05 - accuracy: 1.0000\n",
            "Epoch 00014: val_accuracy did not improve from 0.83422\n",
            "24/24 [==============================] - 2s 80ms/step - loss: 3.9308e-05 - accuracy: 1.0000 - val_loss: 1.0179 - val_accuracy: 0.8289\n",
            "Epoch 15/100\n",
            "23/24 [===========================>..] - ETA: 0s - loss: 3.5830e-05 - accuracy: 1.0000\n",
            "Epoch 00015: val_accuracy did not improve from 0.83422\n",
            "24/24 [==============================] - 2s 80ms/step - loss: 3.6384e-05 - accuracy: 1.0000 - val_loss: 1.0182 - val_accuracy: 0.8289\n",
            "Epoch 16/100\n",
            "23/24 [===========================>..] - ETA: 0s - loss: 3.4175e-05 - accuracy: 1.0000\n",
            "Epoch 00016: val_accuracy did not improve from 0.83422\n",
            "24/24 [==============================] - 2s 92ms/step - loss: 3.3813e-05 - accuracy: 1.0000 - val_loss: 1.0189 - val_accuracy: 0.8289\n",
            "Epoch 17/100\n",
            "23/24 [===========================>..] - ETA: 0s - loss: 3.1592e-05 - accuracy: 1.0000\n",
            "Epoch 00017: val_accuracy did not improve from 0.83422\n",
            "24/24 [==============================] - 2s 79ms/step - loss: 3.1585e-05 - accuracy: 1.0000 - val_loss: 1.0192 - val_accuracy: 0.8289\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "#### 모델로딩"
      ],
      "metadata": {
        "id": "5ho9jhIFyDaE"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from tensorflow.keras.models import load_model"
      ],
      "metadata": {
        "id": "MKSMPK0Nx_6s"
      },
      "execution_count": 19,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "m = load_model('./model/animal/model2_007_0.8342.hdf5')"
      ],
      "metadata": {
        "id": "_NoSrHrAyKW5"
      },
      "execution_count": 20,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "m.trainable=True"
      ],
      "metadata": {
        "id": "kNDr7NI1ySmj"
      },
      "execution_count": 23,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "m.summary()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Jkzpvpb1yUaE",
        "outputId": "22feb1bc-487e-4fc4-d7bb-1ef416a42a53"
      },
      "execution_count": 24,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model: \"sequential_1\"\n",
            "_________________________________________________________________\n",
            " Layer (type)                Output Shape              Param #   \n",
            "=================================================================\n",
            " vgg16 (Functional)          (None, 7, 7, 512)         14714688  \n",
            "                                                                 \n",
            " flatten (Flatten)           (None, 25088)             0         \n",
            "                                                                 \n",
            " dense_3 (Dense)             (None, 128)               3211392   \n",
            "                                                                 \n",
            " dense_4 (Dense)             (None, 64)                8256      \n",
            "                                                                 \n",
            " dense_5 (Dense)             (None, 3)                 195       \n",
            "                                                                 \n",
            "=================================================================\n",
            "Total params: 17,934,531\n",
            "Trainable params: 17,934,531\n",
            "Non-trainable params: 0\n",
            "_________________________________________________________________\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        ""
      ],
      "metadata": {
        "id": "8OMXDUYMyhh8"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}
