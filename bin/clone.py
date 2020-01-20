#!/usr/bin/env python3
"""
Clone builds a cloning dataset and runs a training run over it. We combine these two steps as the
datasets tend to be pretty small so it's simpler to just have all the code together. 
"""
import argparse
from pathlib import Path
import multiprocessing
import time
from subprocess import call
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from derp.fetcher import Fetcher
import derp.util
import derp.model


def sample_perturbs(config):
    """ Sample each value of the perurbs we do """
    sample = {}
    for perturb in config:
        sample[perturb] = np.random.uniform(*config[perturb]['range'])
    return sample


def build_recording(args):
    """
    For each frame in the video generate frames and perturbations to save into a dataset.
    """
    config, recording_folder, out_folder = args
    camera_config = derp.util.load_config(recording_folder / 'config.yaml')['camera']
    predict_fd = open(str(out_folder / 'predict.csv'), 'w')
    status_fd = open(str(out_folder / 'status.csv'), 'w')
    
    topics = derp.util.load_topics(recording_folder)
    assert 'quality' in topics and topics['quality']

    actions = derp.util.extract_car_actions(topics)
    camera = {'times': [msg.publishNS for msg in topics["camera"]]}
    camera['speed'] = derp.util.extract_latest(camera['times'], actions[:, 0], actions[:, 1])
    camera['steer'] = derp.util.extract_latest(camera['times'], actions[:, 0], actions[:, 2])

    bbox = derp.util.get_patch_bbox(config['thumb'], camera_config)
    size = (config['thumb']['width'], config['thumb']['height'])
    n_frames_processed = 0
    for camera_i, timestamp in enumerate(camera['times']):
        if topics['quality'][camera_i].quality != 'good':
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
                predict.append(float(camera[predictor_config['field']][index]))

            status = [store_name]
            for status_config in config['status']:
                status.append(float(camera[status_config['field']][camera_index - 1]))

            frame = derp.util.decode_jpg(topics['camera'][camera_i].jpg)
            derp.util.perturb(frame, camera_config, perturbs)
            patch = derp.util.crop(frame, bbox)
            thumb = derp.util.resize(patch, size)
            derp.util.save_image(out_folder / store_name, thumb)

            predict_row = ['%.6f' % x if isinstance(x, float) else x for x in  predict]
            predict_fd.write(','.join(predict_row) + '\n')
            status_row = ['%.6f' % x if isinstance(x, float) else x for x in  status]
            status_fd.write(','.join(status_row) + '\n')
        n_frames_processed += 1
    print("Build %s %5i %s" % (recording_folder, n_frames_processed, out_folder))
    predict_fd.close()
    status_fd.close()
    return True


def build(config, experiment_path, count):
    np.random.seed(config['seed'])
    process_args = []
    root_folder = derp.util.ROOT / 'recordings'
    for recording_folder in root_folder.glob('recording-*-*-*'):
        partition = 'train' if np.random.rand() < config['build']['train_chance'] else 'test'
        out_folder = experiment_path / partition / recording_folder.stem
        if not out_folder.exists():
            out_folder.mkdir(parents=True)
            process_args.append([config, recording_folder, out_folder])
    pool = multiprocessing.Pool(count)
    pool.map(build_recording, process_args)


def train_epoch(device, model, optimizer, criterion, loader):
    model.train()
    epoch_loss = 0
    for batch_index, (examples, statuses, labels) in enumerate(loader):
        optimizer.zero_grad()
        guesses = model(examples.to(device), statuses.to(device))
        loss = criterion(guesses, labels.to(device))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / (batch_index + 1)


def test_epoch(device, model, optimizer, criterion, loader):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch_index, (examples, statuses, labels) in enumerate(loader):
            guesses = model(examples.to(device), statuses.to(device))
            loss = criterion(guesses, labels.to(device))
            epoch_loss += loss.item()
    return epoch_loss / (batch_index + 1)

def compose_transforms(transform_config):
    transform_list = []
    for perturb_config in transform_config:
        if perturb_config["name"] == "colorjitter":
            transform = transforms.ColorJitter(
                brightness=perturb_config["brightness"],
                contrast=perturb_config["contrast"],
                saturation=perturb_config["saturation"],
                hue=perturb_config["hue"],
            )
            transform_list.append(transform)
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)


class CloneTrainer:
    def __init__(self, config, experiment_path, gpu, count):
        self.config = config
        self.experiment_path = experiment_path
        self.device = torch.device("cuda:" + gpu if torch.cuda.is_available() else "cpu")
        self.count = count
        self.model_fn = eval('derp.model.' + self.config['train']['model'])
        self.criterion = eval('torch.nn.' + self.config['train']['criterion'])().to(self.device)
        self.optimizer_fn = eval('torch.optim.' + self.config['train']['optimizer'])
        self.scheduler_fn = torch.optim.lr_scheduler.ReduceLROnPlateau
        self.dim_in = np.array([config["thumb"][x] for x in ['depth', 'height', 'width']])

        # Prepare transforms
        transforms = compose_transforms(self.config['train']['transforms'])
        train_fetcher = Fetcher(experiment_path / 'train', transforms, config['predict'])
        assert len(train_fetcher)
        test_fetcher = Fetcher(experiment_path / 'test', transforms, config['predict'])
        assert len(test_fetcher)
        self.train_loader = DataLoader(train_fetcher, self.config['train']['batch_size'],
                                       shuffle=True, num_workers=3)
        self.test_loader = DataLoader(test_fetcher, self.config['train']['batch_size'],
                                      num_workers=3)
        print('Train Loader: %6i' % len(self.train_loader.dataset))
        print('Test  Loader: %6i' % len(self.test_loader.dataset))

    def train(self):
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        n_status = len(self.config["status"])
        n_predict = len(self.config["predict"])
        model = self.model_fn(self.dim_in, n_status, n_predict).to(self.device)
        optimizer = self.optimizer_fn(model.parameters(),
                                      self.config['train']['learning_rate'])
        scheduler = self.scheduler_fn(optimizer, factor=0.25, verbose=True, patience=8)
        loss_threshold = test_epoch(self.device, model, optimizer, self.criterion,
                                    self.test_loader)
        print('initial loss: %.6f' % loss_threshold)
        for epoch in range(self.config['train']['epochs']):
            start_time = time.time()
            train_loss = train_epoch(self.device, model, optimizer, self.criterion,
                                     self.train_loader)
            test_loss = test_epoch(self.device, model, optimizer, self.criterion,
                                   self.test_loader)
            scheduler.step(test_loss)
            note = ''
            if test_loss < loss_threshold:
                loss_threshold = test_loss
                torch.save(model, str(self.experiment_path / 'model.pt'))
                note = 'saved'
            duration = time.time() - start_time
            print("Epoch %5i %.6f %.6f %.1fs %s" %
                  (epoch, train_loss, test_loss, duration, note))

def main():
    """
    Run a training instance over the supplied controller dataset. Stores a torch model in
    the controller dataset folder every time the validation loss decreases.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("brain", type=Path, help="Controller we wish to train")
    parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
    parser.add_argument('--count', type=int, default=4, help="parallel processes to build with")

    args = parser.parse_args()

    config = derp.util.load_config(args.brain)
    experiment_path = derp.util.ROOT / 'models' / config['name']
    experiment_path.mkdir(parents=True, exist_ok=True)    

    build(config, experiment_path, args.count)
    trainer = CloneTrainer(config, experiment_path, args.gpu, args.count)
    trainer.train()


if __name__ == "__main__":
    main()
