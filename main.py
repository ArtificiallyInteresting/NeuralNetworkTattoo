import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
from util import *

import keras as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from round import Round

x_train = []
y_train = []

df_train = pd.read_csv('input/train.csv')
df_train = df_train.sample(frac=1) #shuffle
x_train, y_train = lettersToNumbers(df_train)

model = Sequential()
model.add(Dense(8, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Round())


def custom_loss(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred))

model.compile(loss=custom_loss,
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train.values, y_train.values,
          batch_size=1,
          epochs=500,
          verbose=1)

predictions = model.predict(x_train.values, batch_size=1)
print(predictions)
print(numbersToLetters(predictions))