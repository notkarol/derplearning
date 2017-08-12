#!/usr/bin/env python3

import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import pickle
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

H, W, D, L = 32, 128, 3, 2

def model_A():
    model = Sequential()
    model.add(BatchNormalization(input_shape=(H, W, D)))
    model.add(Conv2D(16, (5, 5), activation='elu', padding='same'))
    model.add(BatchNormalization(input_shape=(H, W, D)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (5, 5), activation='elu', padding='same'))
    model.add(Conv2D(32, (5, 5), activation='elu', padding='same'))
    model.add(BatchNormalization(input_shape=(H, W, D)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(48, (3, 3), activation='elu', padding='same'))
    model.add(Conv2D(48, (3, 3), activation='elu', padding='same'))
    model.add(BatchNormalization(input_shape=(H, W, D)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='elu', padding='valid'))
    model.add(BatchNormalization(input_shape=(H, W, D)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
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
    opt = keras.optimizers.SGD(lr=0.002, decay=1e-9, momentum=0.9, nesterov=True)
    model.compile(optimizer=opt, loss='mse')
    model.summary()

    # train model
    #config = tf.ConfigProto( device_count = {'GPU': 0} ) # disable GPU
    #with tf.Session(config=config) as sess:
    with tf.Session() as sess:
        model.fit(train_x, train_y, epochs=64, shuffle=True)

        # Store model and weights
        model_json = model.to_json()
        with open('../data/model-%s.json' % train_name, "w") as f:
            f.write(model_json)
        model.save_weights('../data/weights-%s.h5' % train_name)        

if __name__ == "__main__":
    main()
