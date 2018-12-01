#!/usr/bin/env python3
"""
Run a training instance over the supplied controller dataset. Stores a torch model in
the controller dataset folder every time the validation loss decreases.
"""
import argparse
import time
from subprocess import call

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from derp.fetcher import Fetcher
import derp.util
import derp.model

def step(epoch, model, loader, optimizer, criterion, is_train, device, experiment_path):
    """
    Run through dataset to complete a single epoch
    """
    if is_train:
        model.train()
    else:
        model.eval()

    # Store the average loss for this epoch
    losses = []
    for batch_index, batch in enumerate(loader):

        example_batch = batch[0].to(device)
        status_batch = batch[1].to(device)
        label_batch = batch[2].to(device)
        guess_batch = model(example_batch, status_batch)
        loss = criterion(guess_batch, label_batch)
        losses.append(loss.item())

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        elif batch_index == 0 and epoch % 10 == 0:
            path = experiment_path / ("batch_%02i_%04i" % (epoch, batch_index))
            derp.util.plot_batch(path, batch[0].numpy(), batch[1].numpy(), batch[2].numpy(),
                                 guess_batch.detach().cpu().numpy())

    return np.mean(losses), batch_index + 1

def main():
    """
    Run a training instance over the supplied controller dataset. Stores a torch model in
    the controller dataset folder every time the validation loss decreases.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--controller', type=str, required=True,
                        help="Controller we wish to train")
    parser.add_argument('--model', type=str, default="StarTree",
                        help="Model to run. Default to Medium Sized")
    parser.add_argument('--gpu', type=str, default='0', help="GPU to use")
    parser.add_argument('--bs', type=int, default=32, help="Batch Size")
    parser.add_argument('--lr', type=float, default=1E-3, help="Learning Rate")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs to run for")
    args = parser.parse_args()

    # Prepare config and paths
    config_path = derp.util.get_controller_config_path(args.controller)
    controller_config = derp.util.load_config(config_path)
    experiment_path = derp.util.get_experiment_path(controller_config['name'])

    # If we don't have the experiment file created, make sure we try creating it
    if not experiment_path.exists():
        call(["python3", "clone_build.py", '--controller', args.controller])
    
    # Prepare device we will train on
    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")

    # Prepare model
    thumb_config = controller_config['thumb']
    dim_in = np.array((thumb_config['depth'], thumb_config['height'], thumb_config['width']))
    n_status = len(controller_config['status'])
    n_out = len(controller_config['predict'])
    model_class = derp.util.load_class('derp.model', args.model)
    model = model_class(dim_in, n_status, n_out).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25,
                                                     verbose=True,  patience=8)

    # Prepare perturbation of example
    tlist = []
    for traind in controller_config['train']['prepare']:
        if traind['name'] == 'colorjitter':
            transform = transforms.ColorJitter(brightness=traind['brightness'],
                                               contrast=traind['contrast'],
                                               saturation=traind['saturation'],
                                               hue=traind['hue'])
            tlist.append(transform)
        tlist.append(transforms.ToTensor())
    transform = transforms.Compose(tlist)

    # prepare data loaders
    parts = ['train', 'val']
    loaders = {}
    fetchers = {}
    for part in parts:
        fetchers[part] = Fetcher(experiment_path / part, transform)
        loaders[part] = DataLoader(fetchers[part], batch_size=args.bs, shuffle=True, num_workers=2)
        print("Part %6s #examples: %6i  #batches: %4i"
              % (part, len(fetchers[part]), len(fetchers[part]) / args.bs + 0.9999))

    # Train
    min_loss = 1
    for epoch in range(args.epochs + 1):
        durations = {}
        batch_durations = {}
        losses = {}
        for part in parts:
            start_time = time.time()
            is_train = epoch if 'train' in part else False
            loss, count = step(epoch, model, loaders[part], optimizer, criterion,
                               is_train, device, experiment_path)
            durations[part] = time.time() - start_time
            batch_durations[part] = 1000 * (time.time() - start_time) / count
            losses[part] = loss
        scheduler.step(loss)

        # Only save models that have a lower loss than ever seen before
        note = ''
        if losses[parts[-1]] < min_loss:
            min_loss = losses[parts[-1]]
            name = "%s_%03i_%.6f.pt" % (args.model, epoch, min_loss)
            torch.save(model, str(experiment_path / name))
            note = '*'

        # Prepare
        print("Epoch %5i" % epoch, end=" ")
        total_duration = 0
        for part in parts:
            total_duration += durations[part]
            print("%s (%.5f %2ims)" % (part, losses[part], batch_durations[part]), end=' ')
        print("%4.1fs %s" % (total_duration, note))


if __name__ == "__main__":
    main()
