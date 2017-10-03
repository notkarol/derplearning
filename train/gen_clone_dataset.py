#!/usr/bin/env python3

import cv2
import numpy as np
import os
import sys

import drputil
import srperm as srp

def processRecording(target_config, writer, path, target_dir):
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
        pert_id = 0
        
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

        # Store image
        store_name = os.path.join(target_dir, "%i_%02i.png" % (timestamp, pert_id))
        cv2.imwrite(store_name, thumb)
        
        # Write out states
        writer.write("%i,%02i" % (timestamp, pert_id))
        for state in target_config['states']:
            writer.write(',%f' % states[frame_i][state])
        writer.write("\n")
        writer.flush()
        
    video_cap.release()

def main():

    # Handle arguments
    config_path = sys.argv[1]

    # Import config for ho we want to store
    target_config = drputil.loadConfig(config_path)

    # Create folder for experiment
    experiment_dir = os.path.join(os.environ['DRP_SCRATCH'], target_config['name'])
    drputil.mkdir(experiment_dir)
    
    # Process each recording for training and evaluation
    for mode in ['train', 'eval']:
        dataset_dir = os.path.join(experiment_dir, mode)
        drputil.mkdir(dataset_dir)
        class_dir = os.path.join(dataset_dir, 'default')
        drputil.mkdir(class_dir)
        
        metadata_path = os.path.join(dataset_dir, 'metadata.csv')
        with open(metadata_path, 'w') as metadata_fd:
            for recording_dir in target_config['%s_paths' % mode]: 
                processRecording(target_config, metadata_fd, recording_dir, class_dir)


if __name__ == "__main__":
    main()
