#!/usr/bin/env python3

import argparse
import cv2
import numpy as np
import os
from os.path import join, isdir, exists, basename
import sys

import derp.util
import derp.inferer

def write(inferer, state, predict, data_dir, writer, store_name):
    recording_name = data_dir.split('/')[-1]

    # Crop, resize, and write out image
    thumb = inferer.prepare_x(state)
    store_path = join(data_dir, store_name + '.png')
    cv2.imwrite(store_path, thumb)            

    # Write out state
    writer.write(join(recording_name, store_name))
    for val in predict:
        writer.write(',%f' % val)
    writer.write("\n")
    writer.flush()

def process_recording(sw_config, exp, recording_dir,
                      train_states_fd, val_states_fd,
                      train_dir, val_dir):
    print('process_recording [%s]' % recording_dir)
    recording_name = basename(recording_dir)

    # Prepare our data input
    states_path = join(recording_dir, 'state.csv')
    state_timestamps, state_headers, states = derp.util.read_csv(states_path, floats=True)

    # Skip if there are no labels
    labels_path = join(recording_dir, 'label.csv')
    if not exists(labels_path):
        print("Unable to open [%s]" % labels_path)
        return False
    label_timestamps, label_headers, labels = derp.util.read_csv(labels_path, floats=False)
    
    # Prepare  configs
    hw_config = derp.util.load_config(recording_dir)
    inferer = derp.inferer.Inferer(hw_config, sw_config, exp).script

    # load video
    video_path = join(recording_dir, '%s.mp4' % sw_config[exp]['patch']['component'])
    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        print("Unable to open [%s]" % video_path)
        return False
    
    # Create directories for this recording
    recording_train_dir = join(train_dir, recording_name)
    recording_val_dir = join(val_dir, recording_name)
    if exists(recording_train_dir) or exists(recording_val_dir):
        print("This recording has already been processed")
        return False
    derp.util.mkdir(recording_train_dir)
    derp.util.mkdir(recording_val_dir)    
    
    # Loop through video and add frames into dataset
    frame_id = 0
    while video_cap.isOpened() and frame_id < len(label_timestamps):
        print("%.3f%%" % (100 * frame_id / len(label_timestamps)), end='\r')
        
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
        state = {sw_config[exp]['patch']['component']: frame}
        
        # Prepare predict array
        predict = np.zeros(len(sw_config[exp]['predict']), dtype=np.float)
        for pos, pd in enumerate(sw_config[exp]['predict']):
            if pd['field'] not in state_headers:
                print("Unable to find field [%s]" % pd['field'])
                return False
            
            state_pos = state_headers.index(pd['field'])
            timestamp = state_timestamps[frame_id] + int(pd['delay'] * 1E6)
            predict[pos] = derp.util.find_value(state_timestamps, timestamp,
                                                states[:, state_pos])

        
        # Figure out if this frame is going to be train or validation
        if np.random.rand() < sw_config[exp]['train_chance']:
            data_dir = join(train_dir, recording_name)
            writer = train_states_fd
        else:
            data_dir = join(val_dir, recording_name)
            writer = val_states_fd

        # Write out unperturbed
        store_name = "%06i.png" % (frame_id)
        write(inferer, state, predict, data_dir, writer, store_name)
                
        frame_id += 1
        
    video_cap.release()
    return True


def main(args):
    
    # Import config
    sw_config = derp.util.load_config(args.sw)
    
    # Create folders
    experiment_dir = join(os.environ['DERP_SCRATCH'], sw_config['name'])
    train_dir = join(experiment_dir, 'train')
    val_dir = join(experiment_dir, 'val')
    derp.util.mkdir(experiment_dir)
    derp.util.mkdir(train_dir)
    derp.util.mkdir(val_dir)

    # Create train and val states 
    train_states_path = join(train_dir, 'states.csv')
    val_states_path = join(val_dir, 'states.csv')
    train_states_fd = open(train_states_path, 'a')
    val_states_fd = open(val_states_path, 'a')

    # If we haven't yet written the heading 
    if not exists(train_states_path):
        headers = (['key'] +
                   [sd['name'] for sd in sw_config[args.name]['predict']])
        train_states_fd.write(",".join(headers) + "\n")
        val_states_fd.write(",".join(headers) + "\n")

    # Run through each folder and include it in dataset
    for data_folder in sw_config[args.name]['data_folders']:

        # If we don't have an absolute path, prepend derp_data
        if data_folder[0] != '/':
            data_folder = join(os.environ['DERP_DATA'], data_folder)

        # If we have the recording path, process what's in it
        for filename in os.listdir(data_folder):
            recording_dir = join(data_folder, filename)
            if isdir(recording_dir):
                process_recording(sw_config, args.name, recording_dir,
                                  train_states_fd, val_states_fd,
                                  train_dir, val_dir)

    # Close our states file descriptors
    train_states_fd.close()
    val_states_fd.close()
    print("Wrote to", experiment_dir)
                
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--sw', type=str, required=True, help="software yaml file")
    parser.add_argument('--name', type=str, required=True, help="name of which experiment to train")
    args = parser.parse_args()
    main(args)
