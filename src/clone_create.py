#!/usr/bin/env python3

import cv2
import argparse
import imageio
import numpy as np
import multiprocessing
from numpy.random import rand
import os
import sys
import derp.util


def prepare_state(config, frame_id, state_headers, states, frame):
    state = {}
    for header, value in zip(state_headers, states[frame_id]):
        state[header] = value
    state = {config['thumb']['component']: frame}
    return state


def prepare_predict(config, frame_id, state_headers, state_ts, states):
    predict = np.zeros(len(config['predict']), dtype=np.float)
    for pos, pd in enumerate(config['predict']):
        if pd['field'] not in state_headers:
            print("Unable to find field [%s]" % pd['field'])
            return False

        state_pos = state_headers.index(pd['field'])
        timestamp = state_ts[frame_id] + int(pd['delay'] * 1E6)
        predict[pos] = derp.util.find_value(state_ts, timestamp,
                                            states[:, state_pos]) * pd['scale']
    return predict


def prepare_status(config, frame_id, state_headers, state_ts, states):
    status = np.zeros(len(config['status']), dtype=np.float)
    for pos, sd in enumerate(config['status']):
        if sd['field'] not in state_headers:
            print("Unable to find field [%s]" % sd['field'])
            return False

        state_pos = state_headers.index(sd['field'])
        timestamp = state_ts[frame_id]
        status[pos] = derp.util.find_value(state_ts, timestamp,
                                           states[:, state_pos]) * sd['scale']
    return status


def prepare_pert_magnitudes(config, zero):
    perts = {}
    for pert in sorted(config['perts']):
        perts[pert] = 0.0 if zero else rand(-pert_config[pert], pert_config[pert])
    return perts


def prepare_store_name(frame_id, pert_id, perts):
    store_name = "%06i_%02i" % (frame_id, pert_id)
    for pert in sorted(perts):
        store_name += "_%s-%06.2f" % (pert, perts[pert])
    store_name += ".png"
    return store_name


def write_thumb(inferer, state, data_dir, store_name):
    store_path = os.path.join(data_dir, store_name)
    thumb = inferer.prepare_thumb(state)
    imageio.imwrite(store_path, thumb)


def write_csv(writer, array, data_dir, store_name):
    recording_name = os.path.basename(data_dir)
    writer.write(os.path.join(recording_name, store_name))
    for val in array:
        writer.write(',%f' % val)
    writer.write("\n")
    writer.flush()


def process_recording(args):
    component_config, recording_path, folders = args
    print("Processing", recording_path)
    recording_name = os.path.basename(recording_path)
    
    # Prepare our data input
    states_path = os.path.join(recording_path, 'state.csv')
    state_ts, state_headers, states = derp.util.read_csv(states_path, floats=True)

    # Skip if there are no labels
    labels_path = os.path.join(recording_path, 'label.csv')
    if not os.path.exists(labels_path):
        print("Unable to open [%s]" % labels_path)
        return False
    label_ts, label_headers, labels = derp.util.read_csv(labels_path, floats=False)
    
    # Prepare configs
    source_config = derp.util.load_config(recording_path)
    inferer = derp.util.load_component(component_config, source_config).script

    # Prepare directories and writers
    predict_fds = {}
    status_fds = {}
    for part in folders:
        out_path = os.path.join(folders[part], recording_name)
        if os.path.exists(out_path):
            print("unable to create this recording's dataset as it was already started")
            return False
        os.mkdir(out_path)
        predict_fds[part] = open(os.path.join(out_path, 'predict.csv'), 'a')
        status_fds[part] = open(os.path.join(out_path, 'status.csv'), 'a')
    
    # load video
    video_path = os.path.join(recording_path, '%s.mp4' % component_config['thumb']['component'])
    reader = imageio.get_reader(video_path)

    # Loop through video and add frames into dataset
    frame_id = 0
    for frame_id in range(len(label_ts)):

        # Skip if label isn't good
        if labels[frame_id][label_headers.index('status')] != 'good':
            continue

        # Prepare attributes regardless of perturbation
        part = 'train' if rand() < component_config['create']['train_chance'] else 'val'
        data_dir = os.path.join(folders[part], recording_name)
        frame = reader.get_data(frame_id)

        # Perturb our arrays
        for pert_id in range(component_config['create']['n_perts']):

            # Prepare pert names
            state = prepare_state(component_config, frame_id, state_headers, states, frame)
            predict = prepare_predict(component_config, frame_id, state_headers, state_ts, states)
            status = prepare_status(component_config, frame_id, state_headers, state_ts, states)
            perts = prepare_pert_magnitudes(component_config['create'], pert_id == 0)
            
            # Prepare store name
            store_name = prepare_store_name(frame_id, pert_id, perts)

            write_thumb(inferer, state, data_dir, store_name)
            write_csv(predict_fds[part], predict, data_dir, store_name)
            write_csv(status_fds[part], status, data_dir, store_name)
            frame_id += 1

    # Cleanup and return
    reader.close()
    return True


def main(args):
    
    # Import configs that we wish to train for
    full_config = derp.util.load_config(args.car)
    component_config = derp.util.find_component_config(full_config, 'inference', args.script)
    
    # Create folders
    name = "%s-%s" % (full_config['name'], component_config['name'])
    experiment_path = os.path.join(os.environ['DERP_SCRATCH'], name)
    if not os.path.exists(experiment_path):
        print("Created", experiment_path)
        os.mkdir(experiment_path)

    # Prepare folders and file descriptors
    parts = ['train', 'val']
    folders = {}
    for part in parts:
        folders[part] = os.path.join(experiment_path, part)
        if not os.path.exists(folders[part]):
            print("Created", folders[part])
            os.mkdir(folders[part])

    # Run through each folder and include it in dataset
    process_args = []
    for data_folder in component_config['create']['data_folders']:

        # If we don't have an absolute path, prepend derp_data folder
        if not os.path.isabs(data_folder):
            data_folder = os.path.join(os.environ['DERP_DATA'], data_folder)

        # If we have the recording path, process what's in it
        for filename in os.listdir(data_folder):
            recording_path = os.path.join(data_folder, filename)
            if os.path.isdir(recording_path):
                process_args.append([component_config, recording_path, folders])
                
    # Prepare pool of workers
    if args.count <= 0:
        for arg in process_args:
            process_recording(arg)
    else:
        pool = multiprocessing.Pool(args.count)
        pool.map(process_recording, process_args)
    print("Completed", experiment_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--car', type=str, required=True,
                        help="car components and setup we wish to train for")
    parser.add_argument('--script', type=str, required=True,
                        help="name of script we wish to target in car's config")
    parser.add_argument('--count', type=int, default=4,
                        help='Number of processes to run in parallel')
    args = parser.parse_args()
    main(args)
