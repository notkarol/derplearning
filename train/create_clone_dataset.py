#!/usr/bin/env python3

import cv2
import numpy as np
import os
import sys

import derputil
import srperm as srp

def write(frame, bbox, state, data_dir, writer, store_name):
    recording_name = data_dir.split('/')[-1]

    # Crop, resize, and write out image
    patch = image[bbox.y : bbox.y + bbox.h, bbox.x : bbox.x + bbox.w]
    thumb = csv.resize(patch, size)
    store_path = os.path.join(data_dir, store_name + '.png')
    cv2.imwrite(store_path, thumb)            

    # Write out state
    writer.write(os.path.join(recording_name, store_name))
    for val in state:
        writer.write(',%f' % val)
    writer.write("\n")
    writer.flush()

def process_recording(target_config, recording_dir, train_states_fd, val_states_fd,
                     train_dir, val_dir):
    print('process_recording [%s]' % recording_dir)
    
    # Prepare our variables
    states_path = os.path.join(recording_dir, 'state.csv')
    labels_path = os.path.join(recording_dir, 'label.csv')
    state_timestamps, state_headers, states = derputil.read_csv(states_path)
    label_timestamps, label_headers, labels = derputil.read_csv(labels_path, floats=False)

    recording_name = os.path.basename(recording_dir)
    source_config = derputil.loadConfig(recording_dir)
    cam = 'front'
    bbox = derputil.getPatchBbox(source_config, target_config, cam)
    size = derputil.getPatchSize(target_config, cam)
    hfov_ratio = (target_config['patch'][cam]['hfov']
                  / source_config['camera'][cam]['hfov'])
    frame_id = 0
    video_path = os.path.join(path, 'camera_%s.mp4' % cam)
    video_cap = cv2.VideoCapture(video_path)
    n_perts = target_config['n_perts'] if hfov_ratio > 0.9 else 0
    rot_i = target_config['states'].index('rotation')
    shift_i = target_config['states'].index('shift')
    steer_i = target_config['states'].index('steer')

    # Create directories for this recording
    derputil.mkdir(os.path.join(train_dir, recording_name))
    derputil.mkdir(os.path.join(val_dir, recording_name))
    
    if not video_cap.isOpened():
        print("Unable to open [%s]" % video_path)
        return
    
    # Loop through video and add frames into dataset
    while video_cap.isOpened() and frame_id < len(timestamps):
        print("%.3f%%" % (100 * frame_id / len(timestamps)), end='\r')
        
        # Get the frame if we can
        ret, frame = video_cap.read()
        if not ret or frame is None:
            print("Failed to read frame [%i]" % frame_id)
            break

        # Skip if label isn't good
        if labels[frame_id] != 'good':
            frame_id += 1
            continue

        # Prepare state array
        state = np.zeros(len(target_config['states']), dtype=np.float)
        for target_pos, name in enumerate(target_config['states']):
            if name in state_headers:
                source_pos = state_headers.index(name)
                state[target_pos] = states[frame_id, source_pos]
        
        # Figure out if this frame is going to be train or validation
        if np.random.rand() < target_config['train_chance']:
            data_dir = os.path.join(train_dir, recording_name)
            writer = train_states_fd
        else:
            data_dir = os.path.join(val_dir, recording_name)
            writer = val_states_fd

        # Write out unperturbed
        store_name = "%03i" % (frame_id)
        write(frame, bbox, state, data_dir, writer, store_name)
        
        # Generate perturbations and store perturbed
        for pert_id in range(n_perts):
            pstate = state.copy()
            
            # Generate perturbation variable
            pstate[rot_i] = np.random.uniform(target_config['patch'][cam]['rotate_min'],
                                              target_config['patch'][cam]['rotate_max'])
            pstate[shift_i] = np.random.uniform(target_config['patch'][cam]['shift_min'],
                                                target_config['patch'][cam]['shift_max'])

            # Generate perturbation
            pframe = srp.shiftimg(frame, pstate[rot_i], pstate[shift_i],
                                  source_config['camera'][cam]['hfov'],
                                  source_config['camera'][cam]['vfov'])
            pstate[steer_i]= srp.shiftsteer(pstate[steer_i], pstate[rot_i],
                                           pstate[shift_i])

            # Write out
            store_name = "%03i_%i" % (frame_id, pert_id)
            write(pframe, bbox, pstate, data_dir, writer, store_name)
        
        frame_id += 1
        
    video_cap.release()

def main():

    # Handle arguments
    config_path = sys.argv[1]

    # Import config for ho we want to store
    target_config = derputil.loadConfig(config_path)

    # Create folder and sates files
    experiment_dir = os.path.join(os.environ['DERP_SCRATCH'], target_config['name'])
    train_dir = os.path.join(experiment_dir, 'train')
    val_dir = os.path.join(experiment_dir, 'val')
    train_states_path = os.path.join(train_dir, 'states.csv')
    val_states_path = os.path.join(val_dir, 'states.csv')
    derputil.mkdir(experiment_dir)
    derputil.mkdir(train_dir)
    derputil.mkdir(val_dir)
    train_states_fd = open(train_states_path, 'w')
    val_states_fd = open(val_states_path, 'w')
    train_states_fd.write(",".join(['key'] + target_config['states']) + "\n")
    val_states_fd.write(",".join(['key'] + target_config['states']) + "\n")

    # Run through each folder and include it in dataset
    for data_folder in target_config['data_folders']:
        for filename in os.listdir(data_folder):
            recording_path = os.path.join(data_folder, filename)
            if os.path.isdir(recording_path):
                process_recording(target_config, recording_path,
                                  train_states_fd, val_states_fd,
                                  train_states_path, val_states_path)
    train_states_fd.close()
    val_states_fd.close()
                
if __name__ == "__main__":
    main()
