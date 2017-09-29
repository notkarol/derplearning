#!/usr/bin/env python3

import csv
import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import yaml
import drputil

def processRecording(target_config, writer, path):

    # Prepare timestamps and labels
    state_path = os.path.join(path, 'state.csv')
    state_headers, timestamps, state_rows = readState(state_path)

    # Get our source config
    source_config_path = os.path.join(folder, 'config.yaml')
    with open(source_config_path) as f:
        source_config = yaml.load(f)

    # The same of our example thumbs
    camera = 'front'
    bbox = util.getPatchBbox(source_config, target_config, camera)
    size = util.getPatchSize(target_config, camera)
    
    # For all video frames we have data, save them to the dataset
    frame_i = 0
    video_path = os.path.join(folder, 'camera_%s.mp4' % camera)
    video_cap = cv2.VideoCapture(video_path)
    while video_cap.isOpened() and counter < len(timestamps):

        # Get the frame
        ret, frame = video_cap.read()
        if not ret: break

        # Prepare patch
        patch = util.cropImage(frame, bbox)
        thumb = util.resizeImage(patch, size)

        # Prepare example and write it
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': bytes_feature(label.tobytes()),
            'thumb': bytes_feature(thumb.tobytes())}))
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
        target_config = yaml.load(f)
    
    # Open TF data writer
    model_path = os.path.join(os.environ["DRP_SCRATCH"], "%s.tfrecords" % target_config['name'])
    writer = tf.python_io.TFRecordWriter(model_path)

    # Process each recording indiviudally
    for recording_path in recording_paths:
        process_recording(target_config, writer, recording_path)

    # Clean up video capture
    writer.close()

if __name__ == "__main__":
    main()
