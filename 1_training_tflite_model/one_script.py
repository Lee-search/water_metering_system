# .ipynb 환경을 수행하지 못하는 경우를 위한 Python Script
# e.g. Jupyter notebook 설치불가

########### Basic Parameters for Running: ################################

TFliteName = "digit_v3"   # Used for tflite Filename
Epoch = 500

##########################################################################

import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
from sklearn.utils import shuffle
from tensorflow.python import keras
from tensorflow.python.keras import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Conv2D, MaxPool2D, Flatten, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import History
from tensorflow.keras.utils import to_categorical
from PIL import Image
from pathlib import Path

loss_ges = np.array([])
acc_ges = np.array([])

#%matplotlib inline
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

Input_dir='image_resized'

files = glob.glob(Input_dir + '/*.jpg')
x_data = []
y_data = []

for file in files:
    base = os.path.basename(file)
    target = base[0:1]
    if target == "N":       # 첫 글자가 N 인 경우
        category = 10       # NaN -> 10 으로 변경
    else:
        category = int(target)

    test_image = Image.open(file)
    test_image = np.array(test_image, dtype="float32")
    x_data.append(test_image)
    y_data.append(np.array([category]))

#print(f'Image Size: {len(x_data[0])}')
#print(f'Total X: {len(x_data)}, Y: {len(y_data)}')

x_data = np.array(x_data)
y_data = np.array(y_data)
y_data = to_categorical(y_data, 11)

#print(f'Shape of Input Layer: {x_data.shape}')
#print(f'# of Layer: {y_data.shape}')

x_data, y_data = shuffle(x_data, y_data)

# 해당 예제에선 모든 이미지를 Train Set 으로 사용
X_train = x_data
y_train = y_data

inputs = tf.keras.Input(shape=(32, 20, 3))
inputs2 = tf.keras.layers.BatchNormalization()(inputs)
inputs3 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation="relu")(inputs2)
inputs4 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(inputs3)
inputs5 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation="relu")(inputs4)
inputs6 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(inputs5)
inputs7 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation="relu")(inputs6)
inputs8 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(inputs7)
inputs9 = tf.keras.layers.Flatten()(inputs8)
inputs10 = tf.keras.layers.Dense(128,activation="relu")(inputs9)
output = tf.keras.layers.Dense(11, activation='softmax')(inputs10)

model = tf.keras.Model(inputs=inputs, outputs=output)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95),
              metrics = ["accuracy"])

#model.summary()

# ImageDataGenerator 를 통해 테스트 데이터셋 확장
Batch_Size = 4
Shift_Range = 1
Brightness_Range = 0.3
Rotation_Angle = 10
ZoomRange = 0.4

datagen = ImageDataGenerator(width_shift_range=[-Shift_Range,Shift_Range],
                             height_shift_range=[-Shift_Range,Shift_Range],
                             brightness_range=[1-Brightness_Range,1+Brightness_Range],
                             zoom_range=[1-ZoomRange, 1+ZoomRange],
                             rotation_range=Rotation_Angle)

train_iterator = datagen.flow(x_data, y_data, batch_size=Batch_Size)
history = model.fit(train_iterator, epochs = Epoch)

FileName = TFliteName

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(FileName + ".tflite", "wb").write(tflite_model)

FileName = TFliteName + "q.tflite"

import tensorflow as tf

def representative_dataset():
    for n in range(x_data[0].size):
        data = np.expand_dims(x_data[5], axis=0)
        yield [data.astype(np.float32)]

converter2 = tf.lite.TFLiteConverter.from_keras_model(model)
converter2.representative_dataset = representative_dataset
converter2.optimizations = [tf.lite.Optimize.DEFAULT]
converter2.representative_dataset = representative_dataset
tflite_quant_model = converter2.convert()

open(FileName, "wb").write(tflite_quant_model)
#print(FileName)
Path(FileName).stat().st_size