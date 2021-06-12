import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
from util import *
import kerastuner as kt
import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

from round import Round
from model import build_model
from myTuner import MyTuner

x_train = []
y_train = []

df_train = pd.read_csv('input/train.csv')
df_train = df_train.sample(frac=1) #shuffle
x_train, y_train = lettersToNumbers(df_train)

# tuner = kt.Hyperband(
#     build_model,
#     objective='loss',
#     max_epochs=3,
#     hyperband_iterations=5)

tuner = MyTuner(
    build_model,
    objective='loss',
    max_trials=400)

tuner.search(x_train,y_train)
             # epochs=10)
             # callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)])
best_model = tuner.get_best_models(1)[0]
sorted_hyperparameters = tuner.get_best_hyperparameters(3)
best_hyperparameters = sorted_hyperparameters[0]
print("Best: ")
print(best_hyperparameters.values)
print("Second: ")
print(sorted_hyperparameters[1].values)
print("Third: ")
print(sorted_hyperparameters[2].values)
# exit()

model = build_model(best_hyperparameters)
model.fit(x_train.values, y_train.values,
          batch_size=best_hyperparameters.values["batch_size"],
          epochs=5000,
          verbose=1)

predictions = model.predict(x_train.values, batch_size=1)
print(predictions)
print(numbersToLetters(predictions))
finalStats(predictions, y_train)