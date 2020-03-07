import pytest
import numpy as np
import PIL.Image
import torch
import torchvision.transforms as transforms
import derp.util
import derp.model


@pytest.fixture
def frame():
    return derp.util.load_image("test/100deg.jpg")


@pytest.fixture
def source_config():
    return {'hfov': 50, 'vfov': 50, 'yaw': 0, 'pitch': 0, 'width': 100, 'height': 100,
            'x': 0, 'y': 0, 'z': 1}


@pytest.fixture
def target_config():
    return {'hfov': 32, 'vfov': 32, 'yaw': 0, 'pitch': -4, 'x': 0, 'y': 0, 'z': 1}


class Fetcher(torch.utils.data.Dataset):
    def __init__(self, table):
        self.table = table
        self.transform = transforms.Compose([transforms.ToTensor()])
    def __getitem__(self, index):
        image = PIL.Image.fromarray(self.table[index][0])
        return self.transform(image), self.table[index][1], self.table[index][2]
    def __len__(self):
        return len(self.table)


def test_perturb(frame, source_config, target_config):
    """ verify that a zero perturb does nothing to the pixels """
    bbox = derp.util.get_patch_bbox(target_config, source_config)
    zero_frame = derp.util.perturb(frame.copy(), source_config)
    assert (zero_frame - frame).sum() == 0


def test_perturb_learnability(frame, source_config, target_config):
    bbox = derp.util.get_patch_bbox(target_config, source_config)
    train_table, test_table = [], []
    for shift in np.linspace(-0.4, 0.4, 51):
        for rotate in np.linspace(-4, 4, 51):
            p_frame = derp.util.perturb(frame.copy(), source_config, shift, rotate)
            p_patch = derp.util.crop(p_frame, bbox)
            table = test_table if shift == 0 or rotate == 0 else train_table
            table.append([p_patch, torch.FloatTensor(), torch.FloatTensor([shift * 2.5,
                                                                           rotate * 0.25])])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    train_fetcher, test_fetcher = Fetcher(train_table), Fetcher(test_table)
    train_loader = torch.utils.data.DataLoader(train_fetcher, 32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_fetcher, len(test_fetcher))
    model = derp.model.Tiny(np.roll(train_table[0][0].shape, 1), 0, 2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), 1E-3)
    criterion = torch.nn.MSELoss().to(device)
    test_losses = []
    for epoch in range(5):
        train_loss = derp.model.train_epoch(device, model, optimizer, criterion, train_loader)
        test_loss = derp.model.test_epoch(device, model, criterion, test_loader)
        test_losses.append(test_loss)
    assert min(test_losses) < 2E-3
