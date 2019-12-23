#!/usr/bin/env python3
"""
Builds cloning datasets for use in the clone_train.py script.
"""
import argparse
import multiprocessing
from pathlib import Path
import cv2
import numpy as np
import derp.util


def prepare_predict(config, frame_id, state_headers, state_timestamps, states, perts):
    """
    Prepares the output that the model needs to predict at the current frame, depending on
    perturbations and the delay.
    """
    predict = np.zeros(len(config["predict"]), dtype=np.float)
    for pos, predict_dict in enumerate(config["predict"]):
        if predict_dict["field"] in state_headers:
            state_pos = state_headers.index(predict_dict["field"])
            timestamp = state_timestamps[frame_id] + predict_dict["time_offset"]
            predict[pos] = derp.util.find_value(state_timestamps, timestamp, states[:, state_pos])
        elif predict_dict["field"] in perts:
            predict[pos] = perts[predict_dict["field"]]
            if "time_offset" in predict_dict:
                print("Time offset is not implemented for fields in perts")
                return False
        else:
            print("Unable to find field [%s] in perts or states" % predict_dict["field"])
            return False
        predict[pos] *= predict_dict["scale"]
    return predict


def prepare_status(config, frame_id, state_headers, state_timestamps, states):
    """
    Prepares the status array, which is a vector of inputs that gets attached to the fully
    connected layer of a model.
    """
    status = np.zeros(len(config["status"]), dtype=np.float)
    for pos, status_dict in enumerate(config["status"]):
        if status_dict["field"] not in state_headers:
            print("Unable to find field [%s]" % status_dict["field"])
            return False

        state_pos = state_headers.index(status_dict["field"])
        timestamp = state_timestamps[frame_id]
        status[pos] = derp.util.find_value(state_timestamps, timestamp, states[:, state_pos])
        status[pos] *= status_dict["scale"]
    return status


def prepare_pert_magnitudes(config, zero=False):
    """
    For each perturbation we uniformly sample its range to generate a perturbation.
    """
    perts = {}
    for pert in sorted(config):
        if zero:
            perts[pert] = 0.0
        else:
            perts[pert] = np.random.uniform(-config[pert]["max"], config[pert]["max"])
    return perts


def prepare_store_name(frame_id, pert_id, perts, predict):
    """
    Build the name of the image by conjoining the perturbations and predicted values.
    """
    store_name = "%06i_%02i" % (frame_id, pert_id)
    for pert in sorted(perts):
        store_name += "_%s%05.2f" % (pert[0], perts[pert])
    for pred in predict:
        store_name += "_%06.3f" % (pred)
    store_name += ".png"
    return store_name


def perturb(config, frame_config, frame, predict, status, perts):
    """
    Apply perturbations to the frame so that the frame appears to be located
    - shift: at a parallel offset with respect to its current heading
    - rotate: at heading in a different direction with respect to its current position
    """

    # figure out steer correction based on perturbations
    steer_correction = 0
    for pert in perts:
        pertd = config["build"]["perts"][pert]
        steer_correction += pertd["fudge"] * perts[pert]

    # skip if we have nothing to correct
    if steer_correction == 0:
        return

    # apply steer corrections TODO is this doing what we think it is?
    for i, status_dict in enumerate(config["status"]):
        if status_dict["field"] == "steer":
            status[i] += steer_correction * (1 - min(status_dict["time_offset"], 1))
            status[i] = min(1, status[i])
            status[i] = max(-1, status[i])
    for i, predict_dict in enumerate(config["predict"]):
        if predict_dict["field"] == "steer":
            predict[i] += steer_correction * (1 - min(predict_dict["time_offset"], 1))
            predict[i] = min(1, predict[i])
            predict[i] = max(-1, predict[i])

    # Apply perturbation to pixel values
    derp.util.perturb(frame, frame_config, perts)


def process_recording(args):
    """
    For each frame in the video generate frames and perturbations to save into a dataset.
    """
    config, folder, folders = args
    camera_config = derp.util.load_config(folder / "config.yaml")['camera']
    n_perts = config["build"]["n_perts"] if camera_config["hfov"] > config["thumb"]["hfov"] else 1

    # Prepare folders and status files
    metadata_fds = {}
    for part in folders:
        out_path = folders[part] / folder.name
        if out_path.exists():
            return False
        out_path.mkdir(parents=True)
        metadata_fds[part] = open(str(out_path / "metadata.csv"), "a")

    # Load data, if there are no labels the don't run ourselves
    topics = derp.util.load_topics(folder)
    if 'label' not in topics:
        print(folder, "has no labels")
        return False

    print("Processing", folder.name)
    state = {topic: derp.util.TOPICS[topic].new_message() for topic in topics}
    for timestamp, topic, msg in derp.util.replay(topics):
        state[topic] = msg
        if topic != 'camera' or state['label'].quality != "good":
            continue

        partition = 'train' if np.random.rand() < config["build"]["train_chance"] else 'val'
        data_dir = folders[partition] / folder.name

        # Create each perturbation for dataset
        for pert_id in range(n_perts):

            # Prepare variables to store for this example
            perts = prepare_pert_magnitudes(config["build"]["perts"], pert_id == 0)
            predict = prepare_predict(
                config, frame_id, state_headers, state_timestamps, states, perts
            )
            status = prepare_status(config, frame_id, state_headers, state_timestamps, states)

            # Perturb the image and status/predictions
            frame = brain.state[component_name]
            perturb(config, frame_config, frame, predict, status, perts)

            # Get thumbnail
            thumb = brain.prepare_thumb(frame)

            # Prepare store name
            store_name = prepare_store_name(frame_id, pert_id, perts, predict)
            write_thumb(thumb, data_dir, store_name)
            write_csv(predict_fds[part], predict, data_dir, store_name)
            write_csv(status_fds[part], status, data_dir, store_name)

    return True


def main():
    """
    Builds cloning datasets for use in the clone_train.py script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--brain", type=Path, required=True, help="brain we wish to train")
    parser.add_argument("--car", type=Path, required=True, help="physical car we wish to train for")
    parser.add_argument("--count", type=int, default=4, help="number of parallel processes")
    args = parser.parse_args()

    # Import configs that we wish to train for
    brain_config = derp.util.load_config(args.brain)
    car_config = derp.util.load_config(args.car)

    experiment_path = derp.util.ROOT / 'scratch' / brain_config['name']
    print(experiment_path)
    if not experiment_path.exists():
        experiment_path.mkdir(parents=True, exist_ok=True)

    # Prepare folders and file descriptors
    parts = ["train", "val"]
    folders = {}
    for part in parts:
        folders[part] = experiment_path / part
        if not folders[part].exists():
            folders[part].mkdir(parents=True, exist_ok=True)

    # Run through each folder and include it in dataset
    process_args = []
    for data_folder in brain_config["build"]["data_folders"]:
        data_folder = Path(data_folder)
        if not data_folder.is_absolute():
            data_folder = derp.util.ROOT / "data" / data_folder
        for filename in data_folder.glob("*"):
            folder = data_folder / filename
            if folder.is_dir():
                process_args.append([brain_config, folder, folders])

    # Prepare pool of workers
    if args.count <= 0:
        for arg in process_args:
            process_recording(arg)
    else:
        pool = multiprocessing.Pool(args.count)
        pool.map(process_recording, process_args)


if __name__ == "__main__":
    main()
