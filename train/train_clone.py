#!/usr/bin/env python3

import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, Dense

import drputil


def model_A(x_in):
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x_in)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(32, activation="relu")(x)
    x_out = Dense(2, name='x_out')(x)
    return x_out


def read_and_decode(filenames, config):

    # Prepare data readers
    file_queue = tf.train.string_input_producer(filenames, name='queue')
    reader = tf.TFRecordReader()
    key, entire_example = reader.read(file_queue)
    feature = { 'front': tf.FixedLenFeature([], tf.string),
                'speed': tf.FixedLenFeature([], tf.float32),
                'steer': tf.FixedLenFeature([], tf.float32) }
    
    features = tf.parse_single_example(entire_example, features=feature)

    # Prepare label
    label = tf.stack([features['speed'], features['steer']], 0)

    # Process image
    camera = 'front'
    front = tf.decode_raw(features['front'], tf.uint8)
    front = tf.reshape(front, [config['patch'][camera]['height'],
                               config['patch'][camera]['width'],
                               config['patch'][camera]['depth']])
    
    # Return shuffled
    examples, labels = tf.train.shuffle_batch(
        tensors=[front, label],
        batch_size=32,
        num_threads=2,
        enqueue_many=False,
        capacity=2**10,
        min_after_dequeue=2**9)
    return examples, labels
    
def main():
    config = drputil.loadConfig(sys.argv[1])
    camera = 'front'
    dataset_path = os.path.join(os.environ["DRP_SCRATCH"], "%s.tfrecords" % config['name'])
    batch_shape = (32, config['patch'][camera]['height'],
                   config['patch'][camera]['width'],
                   config['patch'][camera]['depth'])
    
    with tf.Session() as sess:
        x_train_batch, y_train_batch = read_and_decode([dataset_path], config)

        
        # Prepare variables
        x_train_batch = tf.cast(x_train_batch, tf.float32)
        x_train_batch = tf.reshape(x_train_batch, shape=batch_shape)
        y_train_batch = tf.cast(y_train_batch, tf.float32)
        x_batch_shape = x_train_batch.get_shape().as_list()
        y_batch_shape = y_train_batch.get_shape().as_list()
        print(x_batch_shape, y_batch_shape)
        x_train_in = keras.layers.Input(tensor=x_train_batch, batch_shape=x_batch_shape)
        y_train_in = keras.layers.Input(tensor=y_train_batch, batch_shape=y_batch_shape)
        x_train_out = model_A(x_train_in)
                
        # Prepare model
        model = keras.models.Model(inputs=x_train_in, outputs=x_train_out)
        model.compile(optimizer='adam', loss='mse')
        model.summary()
        
        # Prepare threads for running data
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Train model
        model.fit(x_train_in, y_train_in, batch_size=32, epochs=5, shuffle=True)
        model.save_weights('weights.h5')        

        coord.request_stop()
        coord.join(threads)
                            

if __name__ == "__main__":
    main()
