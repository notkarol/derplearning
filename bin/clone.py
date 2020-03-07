#!/usr/bin/env python3
"""
Clone builds a cloning dataset and runs a training run over it. We combine these two steps as the
datasets tend to be pretty small so it's simpler to just have all the code together.
"""
import argparse
from pathlib import Path
import multiprocessing
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from derp.fetcher import Fetcher
import derp.util
import derp.model


def build_recording(config, recording_folder, out_folder, do_perturb):
    """
    For each frame in the video generate frames and perturbations to save into a dataset.
    """
    camera_config = derp.util.load_config(recording_folder / 'config.yaml')['camera']
    predict_fd = open(str(out_folder / 'predict.csv'), 'w')
    status_fd = open(str(out_folder / 'status.csv'), 'w')

    topics = derp.util.load_topics(recording_folder)
    assert 'quality' in topics and topics['quality']

    actions = derp.util.extract_car_actions(topics)
    camera = {'times': [msg.publishNS for msg in topics['camera']]}
    camera['speed'] = derp.util.extract_latest(camera['times'], actions[:, 0], actions[:, 1])
    camera['steer'] = derp.util.extract_latest(camera['times'], actions[:, 0], actions[:, 2])

    bbox = derp.util.get_patch_bbox(config['thumb'], camera_config)
    size = (config['thumb']['width'], config['thumb']['height'])
    n_frames_processed = 0
    for camera_i, timestamp in enumerate(camera['times']):
        if topics['quality'][camera_i].quality != 'good':
            continue
        for perturb_i in range(config['build']['n_samples'] if do_perturb else 1):
            store_name = '%i_%03i.png' % (timestamp, perturb_i)

            shift = np.random.uniform(*config['build']['perturbs']['shift']['range'])
            rotate = np.random.uniform(*config['build']['perturbs']['rotate']['range'])

            predict = [store_name]
            skip = False
            for predictor_config in config['predict']:
                predictor_timestamp = int(timestamp + predictor_config['time_offset'] * 1e9)
                index = np.searchsorted(camera['times'], predictor_timestamp)
                value = float(camera[predictor_config['field']][index])
                if predictor_config['field'] == 'steer':
                    value += config['build']['perturbs']['shift']['fudge'] * shift
                    value += config['build']['perturbs']['rotate']['fudge'] * rotate
                predict.append(value)
            if skip:
                continue

            status = [store_name]
            for status_config in config['status']:
                status.append(float(camera[status_config['field']][camera['index']]))

            frame = derp.util.decode_jpg(topics['camera'][camera_i].jpg)
            derp.util.perturb(frame, camera_config, shift, rotate)
            patch = derp.util.crop(frame, bbox)
            thumb = derp.util.resize(patch, size)
            derp.util.save_image(out_folder / store_name, thumb)
                
            predict_row = ['%.6f' % x if isinstance(x, float) else x for x in predict]
            predict_fd.write(','.join(predict_row) + '\n')
            status_row = ['%.6f' % x if isinstance(x, float) else x for x in status]
            status_fd.write(','.join(status_row) + '\n')
        n_frames_processed += 1
    print('Build %5i %s' % (n_frames_processed, out_folder))
    predict_fd.close()
    status_fd.close()
    return True


def build_recording_fn(args):
    return build_recording(*args)


def build(config, experiment_path, count):
    """ Build the dataset """
    np.random.seed(config['seed'])
    process_args = []
    for i, recording_folder in enumerate(derp.util.RECORDING_ROOT.glob('recording-*-*-*')):
        partition = 'train' if i % config['build']['train_mod'] else 'test'
        out_folder = experiment_path / partition / recording_folder.stem
        if not out_folder.exists():
            out_folder.mkdir(parents=True)
            process_args.append([config, recording_folder, out_folder, partition == 'train'])
    pool = multiprocessing.Pool(count)
    pool.map(build_recording_fn, process_args)


def train(config, experiment_path, gpu):
    device = torch.device('cuda:' + gpu if torch.cuda.is_available() else 'cpu')
    model_fn = eval('derp.model.' + config['train']['model'])
    criterion = eval('torch.nn.' + config['train']['criterion'])().to(device)
    optimizer_fn = eval('torch.optim.' + config['train']['optimizer'])
    scheduler_fn = torch.optim.lr_scheduler.ReduceLROnPlateau
    dim_in = np.array([config['thumb'][x] for x in ['depth', 'height', 'width']])

    # Prepare transforms
    transformer = derp.model.compose_transforms(config['train']['transforms'])
    train_fetcher = Fetcher(experiment_path / 'train', transformer, config['predict'])
    assert len(train_fetcher)
    test_fetcher = Fetcher(experiment_path / 'test', transformer, config['predict'])
    assert len(test_fetcher)
    train_loader = DataLoader(
        train_fetcher, config['train']['batch_size'], shuffle=True, num_workers=3,
    )
    test_loader = DataLoader(test_fetcher, config['train']['batch_size'], num_workers=3)
    print('Train Loader: %6i' % len(train_loader.dataset))
    print('Test  Loader: %6i' % len(test_loader.dataset))
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    n_status = len(config['status'])
    n_predict = len(config['predict'])
    model = model_fn(dim_in, n_status, n_predict).to(device)
    optimizer = optimizer_fn(model.parameters(), config['train']['learning_rate'])
    scheduler = scheduler_fn(optimizer, factor=0.1, verbose=True, patience=8)
    loss_threshold = derp.model.test_epoch(device, model, criterion, test_loader)
    print('initial loss: %.6f' % loss_threshold)
    for epoch in range(config['train']['epochs']):
        start_time = time.time()
        train_loss = derp.model.train_epoch(device, model, optimizer, criterion, train_loader)
        test_loss = derp.model.test_epoch(device, model, criterion, test_loader)
        scheduler.step(test_loss)
        note = ''
        if test_loss < loss_threshold:
            loss_threshold = test_loss
            torch.save(model, str(experiment_path / 'model.pt'))
            note = 'saved'
        duration = time.time() - start_time
        print('Epoch %5i %.6f %.6f %.1fs %s' % (epoch, train_loss, test_loss, duration, note))


def main():
    """
    Run a training instance over the supplied controller dataset. Stores a torch model in
    the controller dataset folder every time the validation loss decreases.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('brain', type=Path, help='Controller we wish to train')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--count', type=int, default=4, help='parallel processes to build with')

    args = parser.parse_args()

    config = derp.util.load_config(args.brain)
    experiment_path = derp.util.MODEL_ROOT / config['name']
    experiment_path.mkdir(parents=True, exist_ok=True)

    build(config, experiment_path, args.count)
    train(config, experiment_path, args.gpu)


if __name__ == "__main__":
    main()
