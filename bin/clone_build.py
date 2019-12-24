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


def sample_perturbs(config):
    """ Sample each value of the perurbs we do """
    sample = {}
    for perturb in config:
        sample[perturb] = np.random.uniform(*config[perturb]['range'])
    return sample

METADATA_HEADER = ['timestamp', 'perturb_i', 'shift', 'rotate',
                   'speed-1.0', 'steer-1.0', 'speed-0.9', 'steer-0.9',
                   'speed-0.8', 'steer-0.8', 'speed-0.7', 'steer-0.7',
                   'speed-0.6', 'steer-0.6', 'speed-0.5', 'steer-0.5', 
                   'speed-0.4', 'steer-0.4', 'speed-0.3', 'steer-0.3',
                   'speed-0.2', 'steer-0.2', 'speed-0.1', 'steer-0.1',
                   'speed', 'steer',
                   'speed+0.1', 'steer+0.1', 'speed+0.2', 'steer+0.2',
                   'speed+0.3', 'steer+0.3', 'speed+0.4', 'steer+0.4',
                   'speed+0.5', 'steer+0.5', 'speed+0.6', 'steer+0.6',
                   'speed+0.7', 'steer+0.7', 'speed+0.8', 'steer+0.8',
                   'speed+0.9', 'steer+0.9', 'speed+1.0', 'steer+1.0']
                    

def prepare_metadata(timestamp, perturb_i, camera_times, camera_speeds, camera_steers, perturbs):
    out = [str(timestamp), '%.3f' % perturb_i,
           '%.3f' % perturbs['shift'], '%.3f' % perturbs['rotate']]
    for offset in np.linspace(-1, 1, 21):
        index = np.searchsorted(camera_times, timestamp + offset * 1E6)
        out.append('%.3f' % camera_speeds[index])
        out.append('%.3f' % camera_steers[index])
    return out


def process_recording(args):
    """
    For each frame in the video generate frames and perturbations to save into a dataset.
    """
    config, recording_folder, out_folder = args
    print("Processing %s into %s" % (recording_folder, out_folder))
    camera_config = derp.util.load_config(recording_folder / 'config.yaml')['camera']
    state_fd = open(str(out_folder / 'state.csv'), 'w')
    state_fd.write(','.join(METADATA_HEADER) + '\n')
    
    topics = derp.util.load_topics(recording_folder)
    assert 'label' in topics and topics['label']

    camera_times = [msg.timePublished for msg in topics["camera"]]
    controls = derp.util.extract_car_controls(topics)                                       
    camera_speeds = derp.util.extract_latest(camera_times, controls[:, 0], controls[:, 1])
    camera_steers = derp.util.extract_latest(camera_times, controls[:, 0], controls[:, 2])
        
    for camera_i, timestamp in enumerate(camera_times):
        if topics['label'][camera_i].quality != 'good':
            continue
        # Skip the first/last 2 seconds of video no matter how it's labeled
        if timestamp < camera_times[0] + 2E6 or timestamp > camera_times[-1] - 2E6:
            continue
        for perturb_i in range(config['build']['n_samples']):
            store_name = '%i_%03i.png' % (timestamp, perturb_i)
            perturbs = sample_perturbs(config['build']['perturbs'])
            metadata = prepare_metadata(
                timestamp, perturb_i, camera_times, camera_speeds, camera_steers, perturbs
            )
            frame = derp.util.decode_jpg(topics['camera'][camera_i].jpg)
            derp.util.perturb(frame, camera_config, perturbs)
            bbox = derp.util.get_patch_bbox(config['thumb'], camera_config)
            patch = derp.util.crop(frame, bbox)
            thumb = derp.util.resize(patch, (config['thumb']['width'], config['thumb']['height']))
            derp.util.save_image(out_folder / store_name, thumb)
            state_fd.write(','.join(metadata) + '\n')
    state_fd.close()
    return True


def main():
    """
    Builds cloning datasets for use in the clone_train.py script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('brain', type=Path, help='brain we wish to train')
    parser.add_argument('--count', type=int, default=4, help='number of parallel processes')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    args = parser.parse_args()

    np.random.seed(args.seed)
    config = derp.util.load_config(args.brain)
    experiment_path = derp.util.ROOT / 'scratch' / config['name']
    experiment_path.mkdir(parents=True, exist_ok=True)

    process_args = []
    for folder in [derp.util.ROOT / 'data' / x for x in config['build']['folders']]:
        for recording_folder in folder.glob('recording-*-*-*'):
            partition = 'train' if np.random.rand() < config['build']['train_chance'] else 'val'
            out_folder = experiment_path / partition / recording_folder.stem
            if not out_folder.exists():
                out_folder.mkdir(parents=True)
                process_args.append([config, recording_folder, out_folder])

    if args.count <= 0:
        for arg in process_args:
            process_recording(arg)
    else:
        pool = multiprocessing.Pool(args.count)
        pool.map(process_recording, process_args)


if __name__ == '__main__':
    main()
