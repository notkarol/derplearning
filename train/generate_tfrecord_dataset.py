#!/usr/bin/env python3

import csv
import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import yaml

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def read_state(path):
    """
    Read thhrough the state file and get our timestamps and recorded values.
    Returns the non-timestamp headers, timestamps as a double array, and
    all non-timestamp values in one 2D float32 array.
    """
    timestamps = []
    states = []
    with open(path) as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            states.append([float(x) for x in row])
            
    timestamps_arr = np.array(timestamps, dtype=np.double)
    states_arr = np.array(states, dtype=np.float32)

    return headers[1:], timestamps_arr, states_arr


def process_recording(train_config, folder, writer):

    # Prepare timestamps and labels
    state_path = os.path.join(folder, 'state.csv')
    state_headers, timestamps, state_rows = read_state(state_path)

    # For all video frames we have data, save them to the dataset
    frame_i = 0
    video_path = os.path.join(folder, 'camera_front.mp4')
    video_cap = cv2.VideoCapture(video_path)
    while video_cap.isOpened() and counter < len(timestamps):

        # Get the frame
        ret, frame = video_cap.read()
        if not ret: break

        # Prepare patch
        crop_x = 0
        crop_w = 640
        crop_y = 0
        crop_h = 480
        target_size = (80, 60)
        patch = frame[crop_x : crop_x + crop_w, crop_y : crop_y + crop_h, :]
        thumb = cv2.resize(patch, target_size)

        # Prepare label, shape, thumb            
        label = labels[frame_i].tobytes()
        thumb = thumb.tobytes()

        # Prepare example and write it
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': bytes_feature(label),
            'thumb': bytes_feature(thumb)}))
        writer.write(example.SerializeToString())

        # Make sure to increment counter
        frame_i += 1
    video_cap.release()


def main():

    # Handle arguments
    config_path = sys.argv[1]
    folder_paths = sys.argv[2:]

    # Import config for ho we want to store
    with open(config_path) as f:
        config = yaml.load(f)
    
    # Open TF data writer
    model_path = os.path.join(os.environ["DRP_SCRATCH"], "%s.tfrecords" % config['name'])
    writer = tf.python_io.TFRecordWriter(model_path)

    for folder in folders:
        process_recording(config, writer, folder)

    # Clean up video capture
    writer.close()

if __name__ == "__main__":
    main()
