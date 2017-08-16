#!/usr/bin/env python3

import cv2
import numpy as np
import os
import pickle
import sys
import yaml
import tables
import srperm as srp

def get_crop_size(height, width):
    if height / width == 0.5625: # 16x9 screen
        return width * 3 // 4, height * 1 // 3
    elif height / width  == 0.75: # 4x3 screen 
        return width, height * 1 // 3
    return None, None

def process(train_config, folder, train_x, train_y, val_x, val_y):

    config_path = os.path.join(folder, 'config.yaml')
    with open(config_path) as f:
        data_config = yaml.load(f)

    target_size = (train_config['patch']['width'], train_config['patch']['height'])
    train_shape = (1, train_config['patch']['depth'],
                   train_config['patch']['height'],
                   train_config['patch']['width'])

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
    video_path = os.path.join(folder, 'front.mp4')
    video_cap = cv2.VideoCapture(video_path)
    counter = 0
    while video_cap.isOpened() and counter < len(labels):

        # Get the frame
        ret, frame = video_cap.read()
        if not ret: break

        # Get crop from frame TODO use config
        crop_width, crop_height = get_crop_size(*frame.shape[0:2])
        if crop_width is None or crop_height is None:
            print("Unable to get crop height or width")
            continue
        crop_x = (data_config['camera']['width'] - crop_width) // 2
        crop_y = data_config['camera']['height'] - crop_height

        # Perturb if we got a wide enough image
        if train_config['perturb'] and data_config['camera']['hfov'] > train_config['patch']['hfov']:
            drot = max(min(np.random.normal(0, 3), 10), -10)
            mshift = max(min(np.random.normal(0, 0.1), 1), -1)
            pframe = srp.shiftimg(frame, drot, mshift, 100, 60)

            patch = pframe[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width, :]
            thumb = cv2.resize(patch, (train_config['patch']['width'],
                                       train_config['patch']['height']))
            speed = labels[counter][0]
            steer = srp.shiftsteer(labels[counter][1], drot, mshift)
            y = np.array([[speed, steer]])
            x = np.reshape(thumb, train_shape)

        # Store original too
        patch = frame[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width, :]
        thumb = cv2.resize(patch, target_size)
        speed = labels[counter][0]
        steer = labels[counter][1]
        y = np.array([[speed, steer]])
        x = np.reshape(thumb, train_shape)

        counter += 1
        print("%.2f\r" % (100.0 * counter / len(labels)), end='')
    print("Done [%i] %s"  % (counter, folder))

    # Clean up
    video_cap.release()
    
def main():

    # Handle arguments
    config_path = sys.argv[1]
    folder_paths = sys.argv[2:]

    # Load Config
    with open(config_path) as f:
        config = yaml.load(f)

    # Update folder paths 
    config['folders'].extend(folder_paths)

    # Create a file for writing
    config['data_path'] = os.path.join(config['data_folder'], "%s.hdf5" % config['name'])
    config['data_fp'] = tables.open_file(config['data_path'], mode='w')
    train = config['data_fp'].create_group("/", 'train', 'training examples and labels')
    val = config['data_fp'].create_group("/", 'val', 'validation examples and labels')
    atom = tables.Float32Atom()
    shape_x = (0, config['patch']['depth'], config['patch']['height'], config['patch']['width'])
    shape_y = (0, len(config['supervisors']))
    train_x = config['data_fp'].create_earray('/train', 'X', atom, shape_x)
    train_y = config['data_fp'].create_earray('/train', 'Y', atom, shape_y)
    val_x = config['data_fp'].create_earray('/val', 'X', atom, shape_x)
    val_y = config['data_fp'].create_earray('/val', 'Y', atom, shape_y)
    
    # Whether to apply perturbs
    for folder in config['folders']:
        process(config, folder, train_x, train_y, val_x, val_y)

    # Clean up after ourselves
    config['data_fp'].close()
    config['data_fp'] = None

    # Save this updated config
    path = os.path.join(config['data_folder'], "%s.yaml" % config['name'])
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
        
if __name__ == "__main__":
    main()
