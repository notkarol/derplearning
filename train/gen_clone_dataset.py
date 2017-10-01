#!/usr/bin/env python3

import cv2
import numpy as np
import os
import sys
import tensorflow as tf

import drputil
import srperm as srp

def processRecording(target_config, writer, path):
    print('processRecording [%s]' % path)
    
    # Prepare our variables
    timestamps, states = drputil.readState(path)
    source_config = drputil.loadConfig(path)
    camera = 'front'
    bbox = drputil.getPatchBbox(source_config, target_config, camera)
    size = drputil.getPatchSize(target_config, camera)
    hfov_ratio = target_config['patch'][camera]['hfov'] / source_config['camera'][camera]['hfov']
    frame_i = -1
    video_path = os.path.join(path, 'camera_%s.mp4' % camera)
    video_cap = cv2.VideoCapture(video_path)

    if not video_cap.isOpened():
        print("Unable to open [%s]" % video_path)
        
    # Loop through video and add frames into dataset
    while video_cap.isOpened() and frame_i + 1 < len(timestamps):
        frame_i += 1
        print("%.3f"% (100 * frame_i / len(timestamps)), end='\r')
        
        # Get the frame
        ret, frame = video_cap.read()
        if not ret:
            print("Failed to read frame [%i]" % frame_i)
            break

        # Get the current sensor information
        timestamp = timestamps[frame_i]
        state = states[frame_i]
        
        # Perturb frame
        if target_config['perturb'] and hfov_ratio < 0.9:
            rotation = np.random.uniform(target_config['patch'][camera]['rotate_min'],
                                         target_config['patch'][camera]['rotate_max'])
            shift = np.random.uniform(target_config['patch'][camera]['shift_min'],
                                      target_config['patch'][camera]['shift_max'])
            perturbed_frame = srp.shiftimg(frame, rotation, shift,
                                           source_config['camera'][camera]['hfov'],
                                           source_config['camera'][camera]['vfov'])
            state['steer'] = srp.shiftsteer(state['steer'], rotation, shift)
            patch = drputil.cropImage(perturbed_frame, bbox)
        else:
            patch = drputil.cropImage(frame, bbox)

        # Get the patch into the target final size
        thumb = drputil.resizeImage(patch, size)

        # Prepare states
        features = {'timestamp': drputil.getTfFeature(timestamp),
                    camera: drputil.getTfFeature(thumb)}
        for key in state:
            features[key] = drputil.getTfFeature(state[key])
        
        # Prepare example and write it
        example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(example.SerializeToString())
        
    video_cap.release()

def main():

    # Handle arguments
    config_path = sys.argv[1]

    # Import config for ho we want to store
    target_config = drputil.loadConfig(config_path)
    
    # Open TF train data writer
    train_path = os.path.join(os.environ["DRP_SCRATCH"], "%s_train.tfrecords" % target_config['name'])
    train_writer = tf.python_io.TFRecordWriter(train_path)
    for recording_path in target_config['train_paths']:
        processRecording(target_config, train_writer, recording_path)
    train_writer.close()

    # Open TF eval data writer
    eval_path = os.path.join(os.environ["DRP_SCRATCH"], "%s_eval.tfrecords" % target_config['name'])
    eval_writer = tf.python_io.TFRecordWriter(eval_path)
    for recording_path in target_config['eval_paths']:
        processRecording(target_config, eval_writer, recording_path)
    eval_writer.close()

    # Clean up video capture

if __name__ == "__main__":
    main()
