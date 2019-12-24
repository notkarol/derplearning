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


def process_recording(args):
    """
    For each frame in the video generate frames and perturbations to save into a dataset.
    """
    config, recording_folder, out_folder = args
    print("Processing %s into %s" % (recording_folder, out_folder))
    camera_config = derp.util.load_config(recording_folder / 'config.yaml')['camera']
    predict_fd = open(str(out_folder / 'predict.csv'), 'w')
    status_fd = open(str(out_folder / 'status.csv'), 'w')
    
    topics = derp.util.load_topics(recording_folder)
    assert 'label' in topics and topics['label']

    controls = derp.util.extract_car_controls(topics)
    camera = {'times': [msg.timePublished for msg in topics["camera"]]}
    camera['speed'] = derp.util.extract_latest(camera['times'], controls[:, 0], controls[:, 1])
    camera['steer'] = derp.util.extract_latest(camera['times'], controls[:, 0], controls[:, 2])
        
    for camera_i, timestamp in enumerate(camera['times']):
        if topics['label'][camera_i].quality != 'good':
            continue
        # Skip the first/last 2 seconds of video no matter how it's labeled
        if timestamp < camera['times'][0] + 2E6 or timestamp > camera['times'][-1] - 2E6:
            continue
        for perturb_i in range(config['build']['n_samples']):
            store_name = '%i_%03i.png' % (timestamp, perturb_i)
            perturbs = sample_perturbs(config['build']['perturbs'])

            predict = [store_name]
            for predictor_config in config['predict']:
                predictor_timestamp = int(timestamp + predictor_config['time_offset'] * 1E6)
                index = np.searchsorted(camera['times'], predictor_timestamp)
                predict.append(camera[predictor_config['field']][index])

            status = [store_name]
            for status_config in config['status']:
                status.append(camera[status_config['field']][camera_index - 1])

            frame = derp.util.decode_jpg(topics['camera'][camera_i].jpg)
            derp.util.perturb(frame, camera_config, perturbs)
            bbox = derp.util.get_patch_bbox(config['thumb'], camera_config)
            patch = derp.util.crop(frame, bbox)
            thumb = derp.util.resize(patch, (config['thumb']['width'], config['thumb']['height']))
            derp.util.save_image(out_folder / store_name, thumb)

            predict_row = ['%.6f' % x if isinstance(x, float) else x for x in  predict]
            predict_fd.write(','.join(predict_row) + '\n')
            status_row = ['%.6f' % x if isinstance(x, float) else x for x in  status]
            status_fd.write(','.join(status_row) + '\n')
    predict_fd.close()
    status_fd.close()
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
