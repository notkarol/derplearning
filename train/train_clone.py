#!/usr/bin/env python3

import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.utils import (saved_model_export_utils)

import drputil

def model_fn(features, labels, mode):
    is_training = mode != tf.estimator.ModeKeys.PREDICT
    
    x = features['front']
    c1 = tf.layers.conv2d(x, filters=32, kernel_size=5, padding='same', activation=tf.nn.relu)
    c1 = tf.layers.max_pooling2d(c1, 2, 2)
    c2 = tf.layers.conv2d(c1, filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)
    c2 = tf.layers.max_pooling2d(c2, 2, 2)
    c3 = tf.layers.conv2d(c2, filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)
    c3 = tf.layers.max_pooling2d(c3, 2, 2)
    c4 = tf.layers.conv2d(c3, filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)
    c4 = tf.layers.max_pooling2d(c4, 2, 2)
    fc1 = tf.contrib.layers.flatten(c4)
    fc1 = tf.layers.dense(fc1, 64)
    fc1 = tf.layers.dropout(fc1, rate=0.5, training=is_training)
    out = tf.layers.dense(fc1, 2, name='servo_tensor') # make it based on labels['servo']

    if not is_training:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=out)

    # Define loss
    loss_op = tf.losses.mean_squared_error(labels['servo'], out)
    optimizer = tf.train.AdamOptimizer(learning_rate=1E-3)
    train_op = optimizer.minimize(loss=loss_op, global_step=tf.train.get_global_step())
    eval_metric_ops = { 'servo': tf.reduce_mean(labels['servo'] - out, name='servo_mean'),
                        'speed': tf.reduce_mean(labels['servo'][:, 0] - out[:, 0], name='speed_mean'),
                        'steer': tf.reduce_mean(labels['servo'][:, 1] - out[:, 1], name='steer_mean')}
    estim_specs = tf.estimator.EstimatorSpec(mode=mode, loss=loss_op, predictions=out,
                                             train_op=train_op, eval_metric_ops={})
    return estim_specs



def read_and_decode(filenames, config):
    """ Generate examples from all the specified tfrecord files """
    # Prepare file queue
    file_queue = tf.train.string_input_producer(filenames, name='queue')

    # Prepare data readers
    reader = tf.TFRecordReader()
    key, entire_example = reader.read(file_queue)
    feature = { 'front': tf.FixedLenFeature([], tf.string),
                'speed': tf.FixedLenFeature([], tf.float32),
                'steer': tf.FixedLenFeature([], tf.float32) }
    features = tf.parse_single_example(entire_example, features=feature)

    # Prepare label
    label = tf.stack([features['speed'], features['steer']], 0)
    label = tf.cast(label, tf.float32)

    # Process example
    camera = 'front'
    front = tf.decode_raw(features['front'], tf.uint8)
    front = tf.reshape(front, [config['patch'][camera]['height'],
                               config['patch'][camera]['width'],
                               config['patch'][camera]['depth']])
    front = tf.cast(front, tf.float32)
    front /= 256
    
    # Return shuffled
    examples, labels = tf.train.shuffle_batch(
        tensors=[front, label],
        batch_size=config['batch_size'],
        num_threads=config['num_threads'],
        enqueue_many=False,
        capacity=2**10,
        min_after_dequeue=2**9)
    return {'front': examples}, {'servo': labels}

def get_input_fn(filenames, config):
    return lambda: read_and_decode(filenames, config)

def main():

    # Load arguemnts
    config = drputil.loadConfig(sys.argv[1])
    if len(sys.argv) >= 3:
        gpu = sys.argv[2]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    # Make sure we have somewhere to run the experiment
    output_path = os.path.join(os.environ["DRP_SCRATCH"], config['name'])
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # prepare data fetchers
    train_path = os.path.join(os.environ["DRP_SCRATCH"], "%s_train.tfrecords" % config['name'])
    eval_path = os.path.join(os.environ["DRP_SCRATCH"], "%s_eval.tfrecords" % config['name'])
    input_fn = get_input_fn([train_path], config)
    
    # Prepare classifier
    classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir=output_path)

    # setup logging
    tensors_to_log = {"servo": "servo_mean", "speed": "speed_mean", "steer": "steer_mean"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)

    # Train
    classifier.train(input_fn=input_fn,
                     steps=100,
                     hooks=[logging_hook])
    eval_results = classifier.evaluate(input_fn=get_input_fn([eval_path], config))
    print(eval_results)
    

if __name__ == "__main__":
    main()
