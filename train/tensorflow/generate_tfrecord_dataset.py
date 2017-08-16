#!/usr/bin/env python3

import cv2
import numpy as np
import os
import sys
import tensorflow as tf

def _int64_feature(value):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

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
        labels = np.array(labels, dtype=np.float32)

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

            # Prepare label, shape, thumb            
            label = labels[counter].tobytes()
            thumb = thumb.tobytes()

            # Prepare example and write it
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _bytes_feature(label),
                'thumb': _bytes_feature(thumb)}))
            writer.write(example.SerializeToString())

            # Make sure to increment counter
            counter += 1

        # Clean up video capture
        writer.close()
        video_cap.release()

if __name__ == "__main__":
    main()
