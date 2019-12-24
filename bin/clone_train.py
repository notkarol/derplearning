#!/usr/bin/env python3
"""
Run a training instance over the supplied controller dataset. Stores a torch model in
the controller dataset folder every time the validation loss decreases.
"""
import argparse
from pathlib import Path
import time
from subprocess import call
import numpy as np
import torch
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
            derp.util.plot_batch(
                path,
                batch[0].numpy(),
                batch[1].numpy(),
                batch[2].numpy(),
                guess_batch.detach().cpu().numpy(),
            )

    return np.mean(losses), batch_index + 1


def main():
    """
    Run a training instance over the supplied controller dataset. Stores a torch model in
    the controller dataset folder every time the validation loss decreases.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("brain", type=Path, help="Controller we wish to train")
    parser.add_argument("--model", type=str, default="StarTree", help="Model class to train")
    parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
    parser.add_argument("--bs", type=int, default=32, help="Batch Size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to run for")
    parser.add_argument('--seed', type=int, default=0, help='seed')
    args = parser.parse_args()

    np.random.seed(args.seed)
    config = derp.util.load_config(args.brain)
    experiment_path = derp.util.ROOT / 'scratch' / config['name']
    if not experiment_path.exists():
        call(["python3", "bin/clone_build.py", args.brain])
    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")

    # Prepare model
    dim_in = np.array([config["thumb"]["depth"],
                       config["thumb"]["height"],
                       config["thumb"]["width"]])
    model_fn = eval('derp.model.' + args.model)
    model = model_fn(dim_in, len(config["status"]), len(config["predict"])).to(device)
    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.25, verbose=True, patience=8
    )

    # Prepare perturbation of example
    transform_list = []
    for perturb_config in config["train"]["perturbs"]:
        if perturb_config["name"] == "colorjitter":
            transform = transforms.ColorJitter(
                brightness=perturb_config["brightness"],
                contrast=perturb_config["contrast"],
                saturation=perturb_config["saturation"],
                hue=perturb_config["hue"],
            )
            transform_list.append(transform)
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)

    # prepare data loaders
    parts = ["train", "val"]
    loaders = {}
    fetchers = {}
    for part in parts:
        fetchers[part] = Fetcher(experiment_path / part, transform, config['predict'])
        loaders[part] = DataLoader(fetchers[part], batch_size=args.bs, shuffle=True, num_workers=2)
        print("Part %6s #examples: %6i  #batches: %4i"
              % (part, len(fetchers[part]), len(fetchers[part]) / args.bs + 0.9999))

    # Train
    min_loss = 1E6
    for epoch in range(args.epochs + 1):
        durations = {}
        batch_durations = {}
        losses = {}
        for part in parts:
            start_time = time.time()
            is_train = epoch if "train" in part else False
            loss, count = step(
                epoch, model, loaders[part], optimizer, criterion, is_train, device, experiment_path
            )
            durations[part] = time.time() - start_time
            batch_durations[part] = 1000 * (time.time() - start_time) / count
            losses[part] = loss
        scheduler.step(loss)

        # Only save models that have a lower loss than ever seen before
        note = ""
        if losses[parts[-1]] < min_loss:
            min_loss = losses[parts[-1]]
            name = "%s_%03i_%.6f.pt" % (args.model, epoch, min_loss)
            torch.save(model, str(experiment_path / name))
            note = "*"

        # Prepare
        print("Epoch %5i" % epoch, end=" ")
        total_duration = 0
        for part in parts:
            total_duration += durations[part]
            print("%s (%.5f %2ims)" % (part, losses[part], batch_durations[part]), end=" ")
        print("%4.1fs %s" % (total_duration, note))


if __name__ == "__main__":
    main()
