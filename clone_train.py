#!/usr/bin/env python3

import argparse
import numpy as np
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from derp.fetcher import Fetcher
import torchvision.transforms as transforms
import derp.util

def step(epoch, config, model, loader, optimizer, criterion,
         is_train, nocuda, plot_batch=False):

    # prepare model for either training or evaluation
    if is_train:
        model.train()
    else:
        model.eval()

    # Store the average loss for this epoch
    step_loss = []

    # Go throgh each epoch
    for batch_idx, (example, status, label) in enumerate(loader):

        # Plot this batch if desired
        if plot_batch:
            name = "batch_%02i_%i_%04i" % (epoch, is_train, batch_idx)
            derp.util.plot_batch(example, label, name)

        # Run training or evaluation to get loss, and then use loss if in training
        if not nocuda:
            example = example.cuda()
            status = status.cuda()
            label = label.cuda()
        example = Variable(example)
        status = Variable(status)
        label = Variable(label)
        
        if is_train:
            optimizer.zero_grad()
        out = model(example, status)
        loss = criterion(out, label)
        step_loss.append(loss.data.mean())
        if is_train:
            loss.backward()
            optimizer.step()
        letter = 'T' if is_train else 'V'
        print("%s %03i %.6f" % (letter, batch_idx, np.mean(step_loss)), end='\r')
        
    return np.mean(step_loss), batch_idx

def main(args):

    # Make sure we have somewhere to run the experiment
    config_path = os.path.join(os.environ['DERP_ROOT'], 'config', args.config + '.yaml')
    full_config = derp.util.load_config(config_path)
    target_config = derp.util.find_component_config(full_config, 'clone')
    experiment_path = os.path.join(os.environ["DERP_ROOT"], 'scratch', full_config['name'])

    # Prepare model
    tc = target_config['thumb']
    dim_in = np.array((tc['depth'], tc['height'], tc['width']))
    n_status = len(target_config['status'])
    n_out = len(target_config['predict'])
    model_class = derp.util.load_class('derp.models.' + args.model.lower(), args.model)
    model = model_class(dim_in, n_status, n_out)
    criterion = nn.MSELoss()
    if not args.nocuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        model = model.cuda()
        criterion = criterion.cuda()
    optimizer = optim.Adam(model.parameters(), args.lr)

    # Prepare perturbation of example
    tlist = []
    for td in target_config['train']['prepare']:
        if td['name'] == 'colorjitter':
            t = transforms.ColorJitter(brightness=td['brightness'],
                                       contrast=td['contrast'],
                                       saturation=td['saturation'],
                                       hue=td['hue'])
            tlist.append(t)
    tlist.append(transforms.ToTensor())
    transform = transforms.Compose(tlist)

    # prepare data loaders
    parts = ['train', 'val']
    loaders = {}
    for part in parts:
        path = os.path.join(experiment_path, part)
        fetcher = Fetcher(path, transform)
        loaders[part] = DataLoader(fetcher, batch_size=args.bs,
                                   num_workers=args.threads, shuffle=True)

    # Train
    min_loss = 1
    for epoch in range(args.epochs + 1):        
        durations = {}
        batch_durations = {}
        losses = {}
        for part in parts:
            start_time = time.time()
            is_train = epoch if 'train' in part else False
            loss, count = step(epoch, target_config, model, loaders[part],
                               optimizer, criterion, is_train,
                               args.nocuda, args.plot)
            durations[part] = time.time() - start_time
            batch_durations[part] = 1000 * (time.time() - start_time) / count
            losses[part] = loss

        # Only save models that have a lower loss than ever seen before
        note = ''
        if losses[parts[-1]] < min_loss:
            min_loss = losses[parts[-1]]
            name = "%s_%03i_%.6f.pt" % (args.model, epoch, min_loss)
            torch.save(model, os.path.join(experiment_path, name))
            note = '*'

        # Prepare
        print("epoch %03i" % epoch, end=" ")
        total_duration = 0
        for part in parts:
            total_duration += durations[part]
            print("%s (%.5f %ims)" % (part, losses[part], batch_durations[part]), end=' ')
        print("%.1fs %s" % (total_duration, note))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help="Car components and setup we wish to train form")
    parser.add_argument('--model', type=str, default="BModel",
                        help="Model to run. Default to Medium Sized")
    parser.add_argument('--gpu', type=int, default=0, help="GPU to use")
    parser.add_argument('--bs', type=int, default=32, help="Batch Size")
    parser.add_argument('--lr', type=float, default=1E-3, help="Learning Rate")
    parser.add_argument('--threads', type=int, default=4, help="Number of threads to fetch data")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs to run for")
    parser.add_argument('--nocuda', default=False, action='store_true', help='do not use cuda')
    parser.add_argument('--plot', default=False, action='store_true',
                        help='save a plot of each batch for verification purposes')
    args = parser.parse_args()    
    main(args)
