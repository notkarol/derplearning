#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn
import sys
import derp.util

def populate(mcl, path, depth):
    if depth == 0:
        return
    for root, dirs, files in os.walk(sys.argv[1]):
        for filename in files:
            if filename[-3:] == '.pt':
                model_path = os.path.join(root, filename)
                config = os.path.basename(os.path.dirname(model_path))
                model, epoch, loss = derp.util.get_name(model_path).split('_')
                epoch = int(epoch)
                loss = float(loss)
                if model not in mcl:
                    mcl[model] = {}
                if config not in mcl[model] or loss < mcl[model][config]:
                    mcl[model][config] = loss
        for folder in dirs:
            folder_path = os.path.join(root, folder)
            populate(mcl, folder_path, depth - 1)

def main():
    mcl = {}
    root_path = sys.argv[1]
    populate(mcl, root_path, depth=1)


    for model in sorted(mcl):
        configs = []
        losses = []
        for config in sorted(mcl[model]):
            configs.append(config)
            losses.append(mcl[model][config])
        plt.semilogy(losses, label=model)

    configs = ('128 x 32 pixels\n75° x 18.75°',
               '128 x 48 pixels\n75° x 28.125°',
               '128 x 64 pixels\n75° x 37.5°',
               '128 x 80 pixels\n75° x 46.875°',
               '128 x 96 pixels\n75° x 56.25°')
    plt.title("Comparing clone models and patch field of views\nLowest validation set loss after 128 epochs on 22 minutes of driving data")
    plt.legend()
    plt.xticks(np.arange(len(configs)), configs)
    plt.xlabel('Dataset')
    plt.ylabel('Validation Loss')
    plt.margins(0.18)
    plt.savefig("2017-10_clone_models.png", bbox_inches='tight', dpi=100)

if __name__ == '__main__':
    main()
