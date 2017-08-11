#!/usr/bin/env python3

import cv2
import numpy as np
import os
import sys
import tensorflow as tf


def float_feature(value):
    if type(value) is not list and type(value) is not tuple and type(value) is not np.ndarray:
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    if type(value) is not list and type(value) is not tuple and type(value) is not np.ndarray:
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def int64_feature(value):
    if type(value) is not list and type(value) is not tuple and type(value) is not np.ndarray:
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    

def main():
    source_size = (640, 480)
    crop_size = (640, 320)
    crop_x = (source_size[0] - crop_size[0]) // 2
    crop_y = (source_size[1] - crop_size[1]) // 2
    target_size = (80, 40)

    name = sys.argv[1]
    folders = sys.argv[2:]
    for folder in folders:

        # Prepare timestamps and labels
        timestamps = []
        labels = []
        csv_path = os.path.join(folder, 'video.csv')
        with open(csv_path) as f:
            headers = f.readline()
            for row in f:
                timestamp, speed, nn_speed, steer, nn_steer = row.split(",")
                timestamps.append([float(timestamp)])
                labels.append((float(speed), float(steer)))
        timestamps = np.array(timestamps, dtype=np.double)
        labels = np.array(labels, dtype=np.float)

        # Prepare video frames by extracting the patch and thumbnail for training
        counter = 0
        writer = tf.python_io.TFRecordWriter("%s.tfrecords" % name)
        video_path = os.path.join(folder, 'video.mp4')
        video_cap = cv2.VideoCapture(video_path)
        while video_cap.isOpened() and counter < len(labels):

            # Get the frame
            ret, frame = video_cap.read()
            if not ret: break

            # Prepare patch
            patch = frame[crop_x : crop_x + crop_size[0], crop_y : crop_y + crop_size[1], :]
            thumb = cv2.resize(patch, target_size)

            # Insert into tensorflow
            thumb_raw = thumb.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'width': int64_feature(thumb.shape[0]),
                'height': int64_feature(thumb.shape[1]),
                'depth': int64_feature(thumb.shape[2]),
                'label': float_feature(labels[counter]),
                'image_raw': bytes_feature(thumb_raw)}))
            writer.write(example.SerializeToString())

            # Make sure to increment counter
            counter += 1

        # Clean up video capture
        video_cap.release()

if __name__ == "__main__":
    main()
