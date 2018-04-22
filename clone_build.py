#!/usr/bin/env python3

import argparse
import imageio
import numpy as np
import multiprocessing
import os
import sys
import derp.util

def prepare_state(config, frame_id, state_headers, states, frame):
    state = {}
    for header, value in zip(state_headers, states[frame_id]):
        state[header] = value
    state[config['thumb']['component']] = frame
    return state


def prepare_predict(config, frame_id, state_headers, state_ts, states, perts):
    predict = np.zeros(len(config['predict']), dtype=np.float)
    for pos, pd in enumerate(config['predict']):
        if pd['field'] in state_headers:
            state_pos = state_headers.index(pd['field'])
            timestamp = state_ts[frame_id] + pd['delay']
            predict[pos] = derp.util.find_value(state_ts, timestamp, states[:, state_pos])
        elif pd['field'] in perts:
            predict[pos] = perts[pd['field']]
            if 'delay' in pd:
                print("Delay is not implemented for fields in perts")
                return False
        else:
            print("Unable to find field [%s] in perts or states" % pd['field'])
            return False
        predict[pos] *= pd['scale']
    return predict


def prepare_status(config, frame_id, state_headers, state_ts, states):
    status = np.zeros(len(config['status']), dtype=np.float)
    for pos, sd in enumerate(config['status']):
        if sd['field'] not in state_headers:
            print("Unable to find field [%s]" % sd['field'])
            return False

        state_pos = state_headers.index(sd['field'])
        timestamp = state_ts[frame_id]
        status[pos] = derp.util.find_value(state_ts, timestamp, states[:, state_pos])
        status[pos] *= sd['scale']
    return status


def prepare_pert_magnitudes(config, zero):
    perts = {}
    for pert in sorted(config):
        if zero:
            perts[pert] = 0.0
        else:
            perts[pert] = np.random.uniform(-config[pert]['max'], config[pert]['max'])
    return perts


def prepare_store_name(frame_id, pert_id, perts, predict):
    store_name = "%06i_%02i" % (frame_id, pert_id)
    for pert in sorted(perts):
        store_name += "_%s%05.2f" % (pert[0], perts[pert])
    for pred in predict:
        store_name += "_%06.3f" % (pred)
    store_name += ".png"
    return store_name


def write_thumb(thumb, data_dir, store_name):
    store_path = os.path.join(data_dir, store_name)
    imageio.imwrite(store_path, thumb)


def write_csv(writer, array, data_dir, store_name):
    recording_name = os.path.basename(data_dir)
    writer.write(os.path.join(recording_name, store_name))
    for val in array:
        writer.write(',%f' % val)
    writer.write("\n")
    writer.flush()


def perturb(config, frame_config, frame, predict, status, perts):

    # figure out steer correction based on perturbations
    steer_correction = 0
    for pert in perts:
        pd = config['create']['perts'][pert]
        steer_correction += pd['fudge'] * perts[pert]

    # skip if we have nothing to correct
    if steer_correction == 0:
        return
        
    # apply steer corrections
    for i, d in enumerate(config['status']):
        if d['field'] == 'steer':
            status[i] += steer_correction * (1 - min(d['delay'], 1))
            status[i] = min(1, status[i])
            status[i] = max(-1, status[i])
    for i, d in enumerate(config['predict']):
        if d['field'] == 'steer':
            predict[i] += steer_correction * (1 - min(d['delay'], 1))
            predict[i] = min(1, predict[i])
            predict[i] = max(-1, predict[i])

    # Manipulate image
    derp.util.perturb(frame, frame_config, perts)


def process_recording(args):
    config, recording_path, folders = args
    component_name = config['thumb']['component']
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
    source_config_path = os.path.join(recording_path, 'car.yaml')
    source_config = derp.util.load_config(source_config_path)
    frame_config = derp.util.find_component_config(source_config, component_name)

    # Load controller off of the first state entry
    state = prepare_state(config, 0, state_headers, states, None)
    controller = derp.util.load_controller(config, source_config, state)

    # Perturb our arrays
    n_perts = 1
    if frame_config['hfov'] > config['thumb']['hfov']:
        n_perts = config['create']['n_perts']
    print("Processing", recording_path, n_perts)

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
    video_path = os.path.join(recording_path, '%s.mp4' % config['thumb']['component'])
    reader = imageio.get_reader(video_path)

    # Loop through video and add frames into dataset
    frame_id = 0
    for frame_id in range(len(label_ts)):

        # Skip if label isn't good
        if labels[frame_id][label_headers.index('status')] != 'good':
            continue

        # Prepare attributes regardless of perturbation
        if np.random.rand() < config['create']['train_chance']:
            part = 'train'
        else:
            part = 'val'
        data_dir = os.path.join(folders[part], recording_name)
        frame = reader.get_data(frame_id)

        # Create each perturbation for dataset
        for pert_id in range(n_perts if part == 'train' else 1):

            # Prepare variables to store for this example
            controller.state = prepare_state(config, frame_id, state_headers, states, frame)
            perts = prepare_pert_magnitudes(config['create']['perts'], pert_id == 0)
            predict = prepare_predict(config, frame_id, state_headers, state_ts, states,
                                      perts)
            status = prepare_status(config, frame_id, state_headers, state_ts, states)

            # Perturb the image and status/predictions
            frame = controller.state[component_name]            
            perturb(config, frame_config, frame, predict, status, perts)

            # Get thumbnail
            thumb = controller.prepare_thumb()
            
            # Prepare store name
            store_name = prepare_store_name(frame_id, pert_id, perts, predict)
            write_thumb(thumb, data_dir, store_name)
            write_csv(predict_fds[part], predict, data_dir, store_name)
            write_csv(status_fds[part], status, data_dir, store_name)

    # Cleanup and return
    reader.close()
    return True


def main(args):
    
    # Import configs that we wish to train for
    config_path = derp.util.get_controller_config_path(args.controller)
    controller_config = derp.util.load_config(config_path)
    experiment_path = derp.util.get_experiment_path(controller_config['name'])

    # Create folders
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
    for data_folder in controller_config['create']['data_folders']:

        # If we don't have an absolute path, prepend derp_data folder
        if not os.path.isabs(data_folder):
            data_folder = os.path.join(os.environ['DERP_ROOT'], 'data', data_folder)

        # If we have the recording path, process what's in it
        for filename in os.listdir(data_folder):
            recording_path = os.path.join(data_folder, filename)
            if os.path.isdir(recording_path):
                process_args.append([controller_config, recording_path, folders])
                
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
    parser.add_argument('--controller', type=str, required=True,
                        help="car controller we wish to train for")
    parser.add_argument('--count', type=int, default=4,
                        help='Number of processes to run in parallel')
    args = parser.parse_args()
    main(args)
