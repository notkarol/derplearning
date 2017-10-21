#!/usr/bin/env python3

import cv2
import numpy as np
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms

import derputil
from derpfetcher import DerpFetcher
import derpmodels
    
def step(epoch, config, model, loader, optimizer, criterion, is_train, plot_batch=False):
    if is_train:
        model.train()
    else:
        model.eval()
    step_loss = []
    for batch_idx, (example, state) in enumerate(loader):
        label = torch.stack([state[x] for x in config['states']], dim=1).float()
        if plot_batch:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(8,4, figsize=(16,12))
            for i in range(len(example)):
                img = np.transpose(example[i].numpy(), (1, 2, 0))
                axs[i // 4, i % 4].imshow(img)
                axs[i // 4, i % 4].set_title(" ".join(["%.2f" % x for x in label[i]]))
            plt.savefig("batch_%i.png" % batch_idx, bbox_inches='tight', dpi=160)
            print("Saved batch")
            
        example, label = Variable(example.cuda()), Variable(label.cuda())
        if is_train:
            optimizer.zero_grad()
        out = model(example)
        loss = criterion(out, label)
        step_loss.append(loss.data.mean())
        if is_train:
            loss.backward()
            optimizer.step()
    return np.mean(step_loss), batch_idx


def main():

    # Load arguemnts
    config = derputil.loadConfig(sys.argv[1])
    if len(sys.argv) >= 3:
        gpu = sys.argv[2]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    # Make sure we have somewhere to run the experiment
    experiment_path = os.path.join(os.environ["DERP_SCRATCH"], config['name'])

    # prepare data fetchers
    transform = transforms.Compose([transforms.ColorJitter(brightness=0.8,
                                                           contrast=0.8,
                                                           saturation=0.8,
                                                           hue=0.0),
                                    transforms.ToTensor()])
    train_dir = os.path.join(experiment_path, 'train')
    val_dir = os.path.join(experiment_path, 'val')
    train_set = DerpFetcher(train_dir, transform)
    val_set = DerpFetcher(val_dir, transform)

    # Parameters
    batch_size = 64
    n_threads = 3
    learning_rate = 1E-3
    n_epochs = 128
    
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=n_threads, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=n_threads)

    model = derpmodels.ModelB(config).cuda()
    criterion = nn.MSELoss().cuda()
    optimizer = optim.Adam(model.parameters(), learning_rate)

    lowest_loss = 1
    for epoch in range(n_epochs + 1):
        train_time = time.time()
        tloss, t = step(epoch, config, model, train_loader, optimizer, criterion, is_train=epoch)
        val_time = time.time()
        vloss, v = step(epoch, config, model, val_loader, optimizer, criterion, is_train=False)
        end_time = time.time()
        print("epoch %03i  tloss:%.6f  vloss:%.6f  ttime:%3i  vtime:%3i" %
              (epoch, tloss, vloss,
               (val_time - train_time) * 1000 / t,
               (end_time - val_time) * 1000 / v))
        if vloss < lowest_loss:
            lowest_loss = vloss
            torch.save(model, os.path.join(experiment_path, "model_%03i.pt" % epoch))
    
if __name__ == "__main__":
    main()
