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
import derp.util as util

def step(epoch, config, model, loader, optimizer, criterion, is_train, plot_batch=False):

    # prepare model for either training or evaluation
    if is_train:
        model.train()
    else:
        model.eval()

    # Store the average loss for this epoch
    step_loss = []

    # Go throgh each epoch
    for batch_idx, (example, state) in enumerate(loader):

        # If we're given a dictionary, then figure out the keys, otherwise just return the state
        if type(state) == dict:
            label = torch.stack([state[x] for x in config['predict']], dim=1).float()
        else:
            label = state.float()

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
        print("%03i %.6f" % (batch_idx, np.mean(step_loss)), end='\r')
    return np.mean(step_loss)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Configuration to use")
    parser.add_argument('--name', type=str, required=True, help="Experiment in configruation")
    parser.add_argument('--model', type=str, default="Pyramid", help="Model to run")
    parser.add_argument('--gpu', type=int, default=0, help="GPU to use")
    parser.add_argument('--bs', type=int, default=32, help="Batch Size")
    parser.add_argument('--lr', type=float, default=1E-3, help="Learning Rate")
    parser.add_argument('--threads', type=int, default=3, help="Number of threads to collect data")
    parser.add_argument('--epochs', type=int, default=128, help="Number of epochs to run for")
    
    args = parser.parse_args()    
    
    # Make sure we have somewhere to run the experiment
    config = util.load_config(args.config)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    experiment_path = os.path.join(os.environ["DERP_SCRATCH"], config['name'])

    # Prepare model
    pc = config[args.name]['patch']
    dim_in = np.array((pc['depth'], pc['height'], pc['width']))
    dim_out = len(config[args.name]['predict'])
    model = util.load_class('derp.models.' + args.model.lower(), args.model)(dim_in, dim_out).cuda()
    criterion = nn.MSELoss().cuda()
    optimizer = optim.Adam(model.parameters(), args.lr)

    # prepare data loaders
    cj = config[args.name]['transforms']['colorjitter']
    transform = transforms.Compose([
        transforms.ColorJitter(brightness=cj['brightness'] if 'brightness' in cj else 0.0,
                               contrast=cj['contrast'] if 'contrast' in cj else 0.0,
                               saturation=cj['saturation'] if 'saturation' in cj else 0.0,
                               hue=cj['hue'] if 'hue' in cj else 0.0),
        transforms.ToTensor()]) 
    train_dir = os.path.join(experiment_path, 'train')
    val_dir = os.path.join(experiment_path, 'val')
    train_set = Fetcher(train_dir, transform)
    val_set = Fetcher(val_dir, transform)
    train_loader = DataLoader(train_set, batch_size=args.bs, num_workers=args.threads, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.bs, num_workers=args.threads)

    # Train
    min_loss = 1
    for epoch in range(args.epochs + 1):
        tloss = step(epoch, config[args.name], model, train_loader,
                     optimizer, criterion, is_train=epoch)
        vloss = step(epoch, config[args.name], model, val_loader,
                     optimizer, criterion, is_train=False)

        # Only save models that have a lower loss
        if vloss < min_loss:
            min_loss = vloss
            torch.save(model, os.path.join(experiment_path, "%s_%03i_%.6f.pt" %
                                           (args.model, epoch, vloss)))
        print("epoch %03i  tloss:%.6f  vloss:%.6f %s" %
              (epoch, tloss, vloss, '*' if vloss == min_loss else ''))
    
if __name__ == "__main__":
    main()
