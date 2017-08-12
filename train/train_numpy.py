#!/usr/bin/env python3

import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import pickle
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation

H, W, D, L = 40, 80, 3, 2

def model_A():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='elu', padding='same', input_shape=(H, W, D)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (5, 5), activation='elu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, (5, 5), activation='elu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='elu', padding='valid'))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, (3, 3), activation='elu', padding='valid'))
    model.add(Flatten())
    model.add(Dense(L, name='x_out'))
    return model
    
def main():
    train_path = sys.argv[1]
    train_name = os.path.basename(train_path).split(".")[0]

    # Load data
    with open(train_path, 'rb') as f:
        train_x, train_y = pickle.load(f)
    print(train_x.shape, train_y.shape)

    # Prepare model
    model = model_A()
    model.compile(optimizer='adadelta', loss='mse')
    model.summary()

    # train model
    #config = tf.ConfigProto( device_count = {'GPU': 0} )
    #with tf.Session(config=config) as sess:
    with tf.Session() as sess:
        model.fit(train_x, train_y, epochs=64, shuffle=True)

        # Store model and weights
        model_json = model.to_json()
        with open('model-%s.json' % train_name, "w") as f:
            f.write(model_json)
        model.save_weights('weights-%s.h5' % train_name)        

if __name__ == "__main__":
    main()
