#!/usr/bin/env python3

import argparse
import cv2
import imageio
import numpy as np
import os
from os.path import basename, exists, isabs, isdir, join
import sys
import derp.util
import derp.inferer

def write_thumb(inferer, state, data_dir, store_name):
    store_path = join(data_dir, store_name)
    imageio.imwrite(store_path, inferer.prepare_thumb(state))


def write_csv(writer, array, data_dir, store_name):
    recording_name = basename(data_dir)
    writer.write(join(recording_name, store_name))
    for val in array:
        writer.write(',%f' % val)
    writer.write("\n")
    writer.flush()


def process_recording(sw_config, target_hw_config, recording_path,
                      folders, predict_fds, status_fds):
    recording_name = basename(recording_path)

    # Prepare our data input
    states_path = join(recording_path, 'state.csv')
    state_timestamps, state_headers, states = derp.util.read_csv(states_path, floats=True)

    # Skip if there are no labels
    labels_path = join(recording_path, 'label.csv')
    if not exists(labels_path):
        print("Unable to open [%s]" % labels_path)
        return False
    label_timestamps, label_headers, labels = derp.util.read_csv(labels_path, floats=False)
    
    # Prepare  configs
    source_hw_config = derp.util.load_config(recording_path)
    inferer = derp.inferer.Inferer(source_hw_config, target_hw_config, sw_config).script

    # load video
    video_path = join(recording_path, '%s.mp4' % sw_config['thumb']['component'])
    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        print("Unable to open [%s]" % video_path)
        return False
    
    # Create directories for this recording
    for name in folders:
        out_path = join(folders[name], recording_name)
        if exists(out_path):
            print("unable to create this recording's dataset as it was already started")
            return False
        derp.util.mkdir(out_path)
    
    # Loop through video and add frames into dataset
    frame_id = 0
    while video_cap.isOpened() and frame_id < len(label_timestamps):
        print("%5.1f%%" % (100 * frame_id / len(label_timestamps)), end='\r')
        
        # Get the frame if we can
        ret, frame = video_cap.read()
        if not ret or frame is None:
            print("Failed to read frame [%i]" % frame_id)
            break

        # Skip if label isn't good
        if labels[frame_id][label_headers.index('status')] != 'good':
            frame_id += 1
            continue

        # Prepare state array
        state = {sw_config['thumb']['component']: frame}
        
        # Prepare predict array
        predict = np.zeros(len(sw_config['predict']), dtype=np.float)
        for pos, pd in enumerate(sw_config['predict']):
            if pd['field'] not in state_headers:
                print("Unable to find field [%s]" % pd['field'])
                return False
            
            state_pos = state_headers.index(pd['field'])
            timestamp = state_timestamps[frame_id] + int(pd['delay'] * 1E6)
            predict[pos] = derp.util.find_value(state_timestamps, timestamp,
                                                states[:, state_pos]) * pd['scale']

        # Predict status array
        status = np.zeros(len(sw_config['status']), dtype=np.float)
        for pos, sd in enumerate(sw_config['status']):
            if sd['field'] not in state_headers:
                print("Unable to find field [%s]" % sd['field'])
                return False
            
            state_pos = state_headers.index(sd['field'])
            timestamp = state_timestamps[frame_id]
            status[pos] = derp.util.find_value(state_timestamps, timestamp,
                                               states[:, state_pos]) * sd['scale']
        
        # Figure out if this frame is going to be train or validation
        name = 'train' if np.random.rand() < sw_config['create']['train_chance'] else 'val'
        store_name = "%06i.png" % (frame_id)
        data_dir = join(folders[name], recording_name)
        write_thumb(inferer, state, data_dir, store_name)
        write_csv(predict_fds[name], predict, data_dir, store_name)
        write_csv(status_fds[name], status, data_dir, store_name)
        frame_id += 1

    video_cap.release()
    return True


def main(args):
    
    # Import configs that we wish to train for
    sw_config = derp.util.load_config(args.sw)
    target_hw_config = derp.util.load_config(args.hw)
    
    # Create folders
    experiment_path = join(os.environ['DERP_SCRATCH'], sw_config['name'])
    derp.util.mkdir(experiment_path)

    # Prepare folders and file descriptors
    folders = {}
    predict_fds = {}
    status_fds = {}
    for name in ['train', 'val']:
        folders[name] = join(experiment_path, name)
        derp.util.mkdir(folders[name])
        predict_path = join(folders[name], 'predict.csv')
        predict_fds[name] = open(predict_path, 'a')
        status_path = join(folders[name], 'status.csv')
        status_fds[name] = open(status_path, 'a')

    # Run through each folder and include it in dataset
    for data_folder in sw_config['create']['data_folders']:

        # If we don't have an absolute path, prepend derp_data folder
        if not isabs(data_folder):
            data_folder = join(os.environ['DERP_DATA'], data_folder)

        # If we have the recording path, process what's in it
        for filename in os.listdir(data_folder):
            recording_path = join(data_folder, filename)
            if isdir(recording_path):
                print("Processing", recording_path)
                process_recording(sw_config, target_hw_config, recording_path,
                                  folders, predict_fds, status_fds)
                
    # Close our states file descriptors
    print("Completed", experiment_path)
                
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--sw', type=str, required=True,
                        help="software config we wish to make dataset for")
    parser.add_argument('--hw', type=str, required=True,
                        help="target hardware config our car will be at")
    args = parser.parse_args()
    main(args)
