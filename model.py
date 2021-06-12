import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow.keras as k
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf


from round import Round

def custom_loss(y_true, y_pred):
    return tf.reduce_sum(tf.abs(y_true - y_pred))

def build_model(hp):
    model = Sequential()
    model.add(Dense(8, activation='relu'))
    for i in range(hp.Int('dense_layers', 1, 8, default=3)):
        model.add(Dense(hp.Int('nodes_per_layer', 4, 64, default=16), activation='relu'))

    model.add(Dense(8, activation='relu'))
    model.add(Round())

    model.compile(loss=custom_loss,
                  optimizer=tf.keras.optimizers.Adam(
                      hp.Float('learning_rate', 1e-5, 1e-1, sampling='log')),
                  metrics=['accuracy'])

    return model

