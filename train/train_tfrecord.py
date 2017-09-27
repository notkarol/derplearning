#!/usr/bin/env python3

import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation

H, W, D, L = 40, 80, 3, 2

def model_A(x_in):
    x = Conv2D(32, (5, 5), activation='elu', padding='same')(x_in)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32, (5, 5), activation='elu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(32, (5, 5), activation='elu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32, (3, 3), activation='elu', padding='valid')(x)
    x = Dropout(0.25)(x)
    x = Conv2D(32, (3, 3), activation='elu', padding='valid')(x)
    x = Flatten()(x)
    x_out = Dense(L, name='x_out')(x)
    return x_out


def read_and_decode(filenames):

    # Prepare data readers
    file_queue = tf.train.string_input_producer(filenames, name='queue')
    reader = tf.TFRecordReader()
    key, entire_example = reader.read(file_queue)
    feature = { 'label': tf.FixedLenFeature([], tf.string),
                'thumb': tf.FixedLenFeature([], tf.string) }
    
    features = tf.parse_single_example(entire_example, features=feature)

    # Prepare image
    label = tf.decode_raw(features['label'], tf.float32)
    thumb = tf.decode_raw(features['thumb'], tf.uint8)
    label = tf.reshape(label, [L])
    thumb = tf.reshape(thumb, [H, W, D])
    thumb = tf.to_float(thumb)
    
    # Return shuffled
    examples, labels = tf.train.shuffle_batch(
        tensors=[thumb, label],
        batch_size=32,
        num_threads=2,
        enqueue_many=False,
        capacity=2**10,
        min_after_dequeue=2**9)
    return examples, labels
    
def main():
    paths = sys.argv[1:]
    
    with tf.Session() as sess:
        x_train_batch, y_train_batch = read_and_decode(paths)

        
        # Prepare variables
        x_train_batch = tf.cast(x_train_batch, tf.float32)
        x_train_batch = tf.reshape(x_train_batch, shape=(32, H, W, D))
        y_train_batch = tf.cast(y_train_batch, tf.float32)
        x_batch_shape = x_train_batch.get_shape().as_list()
        y_batch_shape = y_train_batch.get_shape().as_list()
        x_train_in = keras.layers.Input(tensor=x_train_batch, batch_shape=x_batch_shape)
        y_train_in = keras.layers.Input(tensor=y_train_batch, batch_shape=y_batch_shape)
        x_train_out = model_A(x_train_in)
                
        # Prepare model
        model = keras.models.Model(inputs=x_train_in, outputs=x_train_out)
        model.compile(optimizer='adadelta', loss='mse')
        model.summary()
        
        # Prepare threads for running data
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Train model
        model.fit(y=y_train_in, epochs=5, steps_per_epoch=5)
        model.save_weights('weights.h5')        

        coord.request_stop()
        coord.join(threads)
                            

if __name__ == "__main__":
    main()
