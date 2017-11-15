#!/usr/bin/env python3

import argparse
import cv2
import numpy as np
import os
from os.path import join, isdir
import sys

import derp.util

def write(frame, bbox, size, state, data_dir, writer, store_name):
    recording_name = data_dir.split('/')[-1]

    # Crop, resize, and write out image
    patch = frame[bbox.y : bbox.y + bbox.h, bbox.x : bbox.x + bbox.w]
    thumb = cv2.resize(patch, size, interpolation=cv2.INTER_AREA)
    store_path = join(data_dir, store_name + '.png')
    cv2.imwrite(store_path, thumb)            

    # Write out state
    writer.write(join(recording_name, store_name))
    for val in state:
        writer.write(',%f' % val)
    writer.write("\n")
    writer.flush()

def process_recording(settings, recording_dir,
                      train_states_fd, val_states_fd,
                      train_dir, val_dir):
    print('process_recording [%s]' % recording_dir)
    recording_name = basename(recording_dir)
    
    # Prepare our variables
    states_path = join(recording_dir, 'state.csv')
    labels_path = join(recording_dir, 'label.csv')
    state_timestamps, state_headers, states = derp.util.read_csv(states_path)
    label_timestamps, label_headers, labels = derp.util.read_csv(labels_path, floats=False)

    
    component = settings['patch']['component']
    hw_config = derp.util.load_config(recording_dir)

    bbox = derp.util.get_patch_bbox(hw_config[component], settings)
    qbbox = derp.util.Bbox(bbox.x // 4, bbox.y // 4, bbox.w // 4, bbox.h // 2)
    size = (settings['patch']['width'],  settings['patch']['height'])
    
    hfov_ratio = settings['patch']['hfov'] / hw_config[component]['hfov']
    n_perts = 0 if hfov_ratio > 0.75 else settings['n_perts']
    frame_id = 0
    video_path = join(recording_dir, 'camera_front.mp4')
    video_cap = cv2.VideoCapture(video_path)
    rot_i = settings['predict'].index('rotation')
    
    # Create directories for this recording
    recording_train_dir = join(train_dir, recording_name)
    recording_val_dir = join(val_dir, recording_name)
    if exists(recording_train_dir) or exists(recording_val_dir):
        print("This recording has already been processed")
        return
    derp.util.mkdir(recording_train_dir)
    derp.util.mkdir(recording_val_dir)
    
    if not video_cap.isOpened():
        print("Unable to open [%s]" % video_path)
        return

    # Loop through video and add frames into dataset
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
        state = {sd['name'] : None for sd in  settings['predict']}
        target_frame_id = frame_id + int(hw_config[component]['fps'] * settings['delay'] + 0.5)
        target_frame_id = max(target_frame_id, len(states) - 1)
        for target_pos, name in enumerate(settings['predict']):
            if name in state_headers:
                source_pos = state_headers.index(name)
                state[target_pos] = states[frame_id, source_pos]
        
        # Figure out if this frame is going to be train or validation
        if np.random.rand() < settings['train_chance']:
            data_dir = join(train_dir, recording_name)
            writer = train_states_fd
        else:
            data_dir = join(val_dir, recording_name)
            writer = val_states_fd

        # Write out unperturbed
        store_name = "%05i" % (frame_id)
        write(frame, bbox, size, state, data_dir, writer, store_name)
        
        # Generate perturbations and store perturbed
        if n_perts:
            qframe = cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        for pert_id in range(n_perts):
            pstate = state.copy()
            
            # Generate perturbation variable
            pstate[rot_i] = np.random.uniform(settings['perturbations']['rotate']['min'],
                                              settings['perturbations']['rotate']['max'])
            pstate[shift_i] = np.random.uniform(settings['perturbations']['shift']['min'],
                                                settings['perturbations']['shift']['max'])

            # Generate perturbation
            pframe = derp.util.shiftimg(qframe, pstate[rot_i], pstate[shift_i],
                                        hw_config[component]['hfov'],
                                        hw_config[component]['vfov'])
            pstate[steer_i]= derp.util.shiftsteer(pstate[steer_i], pstate[rot_i],
                                                  pstate[shift_i])
            
            # Write out
            store_name = "%05i_%i" % (frame_id, pert_id)
            write(pframe, qbbox, size, pstate, data_dir, writer, store_name)
            cv2.waitKey(1000)
        
        frame_id += 1
        
    video_cap.release()

def main(args):
    
    # Import config
    config = derp.util.load_config(args.sw)
    sw_config = config[args.name]
    
    # Create folders
    experiment_dir = join(os.environ['DERP_SCRATCH'], config['name'])
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
    if getsize(train_states_path) == 0:
        headers = (['key'] +
                   [sd['name'] for sd in settings['predict']] +
                   [key for key in settings['perturbations']])
        train_states_fd.write(",".join(headers) + "\n")
        val_states_fd.write(",".join(headers) + "\n")

    # Run through each folder and include it in dataset
    for data_folder in settings['data_folders']:

        # If we don't have an absolute path, prepend derp_data
        if data_folder[0] != '/':
            data_folder = join(os.environ['DERP_DATA'], data_folder)

        # If we have the recording path, process what's in it
        for filename in os.listdir(data_folder):
            recording_path = join(data_folder, filename)
            if isdir(recording_path):
                process_recording(settings, recording_path,
                                  train_states_fd, val_states_fd,
                                  train_dir, val_dir)

    # Close our states file descriptors
    train_states_fd.close()
    val_states_fd.close()
    print("Wrote to", experiment_dir)
                
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--sw', type=str, required=True, help="software yaml file")
    parser.add_argument('--name', type=str, required=True, help="name of settings dict")
    args = parser.parse_args()
    main(args)
