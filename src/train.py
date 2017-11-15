#!/usr/bin/env python3

import argparse
import numpy as np
from os.path import join
from os import environ
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from derp.fetcher import Fetcher
import torchvision.transforms as transforms
import derp.util as util

def step(epoch, config, model, loader, optimizer, criterion, is_train, nocuda, plot_batch=False):

    # prepare model for either training or evaluation
    if is_train:
        model.train()
    else:
        model.eval()

    # Store the average loss for this epoch
    step_loss = []

    # Go throgh each epoch
    for batch_idx, (example, state) in enumerate(loader):

        # Prepare label based on what we want to predict
        label = torch.stack([state[x] for x in config['predict']], dim=1).float()

        # Plot this batch if desired
        if plot_batch:
            name = "batch_%02i_%i_%04i" % (epoch, is_train, batch_idx)
            util.plot_batch(example, label, name)

        # Run training or evaluation to get loss, and then use loss if in training
        example, label = Variable(example.cuda()), Variable(label.cuda())
        if is_train:
            optimizer.zero_grad()
        out = model(example)
        loss = criterion(out, label)
        step_loss.append(loss.data.mean())
        if is_train:
            loss.backward()
            optimizer.step()
        letter = 'T' if is_train else 'V'
        print("%s %03i %.6f" % (letter, batch_idx, np.mean(step_loss)), end='\r')
        
    return np.mean(step_loss), batch_idx


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--sw', type=str, required=True, help="Software configuration to use")
    parser.add_argument('--exp', type=str, required=True, help="Experiment in configruation")
    parser.add_argument('--model', type=str, default="BasicModel", help="Model to run")
    parser.add_argument('--gpu', type=int, default=0, help="GPU to use")
    parser.add_argument('--bs', type=int, default=64, help="Batch Size")
    parser.add_argument('--lr', type=float, default=1E-3, help="Learning Rate")
    parser.add_argument('--threads', type=int, default=4, help="Number of threads to fetch data")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs to run for")
    parser.add_argument('--nocuda', default=False, action='store_true', help='do not use cuda')
    parser.add_argument('--plot', default=False, action='store_true',
                        help='save a plot of each batch for verification purposes')
    args = parser.parse_args()    

    # Make sure we have somewhere to run the experiment
    config = util.load_config(args.sw)
    experiment_path = join(environ["DERP_SCRATCH"], config['name'])

    # Prepare model
    pc = config[args.exp]['patch']
    dim_in = np.array((pc['depth'], pc['height'], pc['width']))
    dim_out = len(config[args.exp]['predict'])
    model = util.load_class('derp.models.' + args.model.lower(), args.model)(dim_in, dim_out)
    criterion = nn.MSELoss()
    if not args.nocuda:
        environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        model = model.cuda()
        criterion = criterion.cuda()
    optimizer = optim.Adam(model.parameters(), args.lr)

    # prepare data loaders
    cj = config[args.exp]['transforms']['colorjitter']
    transform_x = transforms.Compose([
        transforms.ColorJitter(brightness=cj['brightness'], contrast=cj['contrast'],
                               saturation=cj['saturation'], hue=cj['hue']),
        transforms.ToTensor()])
    transform_xy = transforms.Compose([])

    train_dir = join(experiment_path, 'train')
    val_dir = join(experiment_path, 'val')
    train_set = Fetcher(train_dir, transform_x, transform_xy)
    val_set = Fetcher(val_dir, transform_x, transform_xy)
    train_loader = DataLoader(train_set, batch_size=args.bs, num_workers=args.threads,
                              shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.bs, num_workers=args.threads)

    # Train
    min_loss = 1
    for epoch in range(args.epochs + 1):
        start_time = time.time()
        tloss, tcount = step(epoch, config[args.exp], model, train_loader,
                             optimizer, criterion, epoch, args.nocuda, args.plot)
        train_time = time.time()
        vloss, vcount = step(epoch, config[args.exp], model, val_loader,
                             optimizer, criterion, False, args.nocuda, args.plot)
        trainval_time = time.time()
        
        # Only save models that have a lower loss than ever seen before
        if vloss < min_loss:
            min_loss = vloss
            name = "%s_%03i_%.6f.pt" % (args.model, epoch, vloss)
            torch.save(model, join(experiment_path, name))

        # Prepare
        dur = trainval_time - start_time
        tms = 1000 * (train_time - start_time) / tcount
        vms = 1000 * (trainval_time - train_time) / vcount
        note = '*' if vloss == min_loss else ''
        print("epoch %03i  tloss: %.6f  vloss: %.6f  dur: %6.1f  tms: %3i, vms: %3i  %s" %
              (epoch, tloss, vloss, dur, tms, vms, note))
               
    
if __name__ == "__main__":
    main()
