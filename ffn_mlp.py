import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset

import numpy as np
from tqdm import tqdm
import cv2


def create_grid(h, w, device="cpu"):
    grid_y, grid_x = torch.meshgrid([torch.linspace(0, 1, steps=h),
                                     torch.linspace(0, 1, steps=w)])
    grid = torch.stack([grid_y, grid_x], dim=-1)
    return grid.to(device)



class ImageDataset(Dataset):

    def __init__(self, image_path, img_dim):
        self.img_dim = (img_dim, img_dim) if type(img_dim) == int else img_dim
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        h, w, c = image.shape
        left_w = int((w - h) / 2)

        image = image[:, left_w:left_w + h]
        image = cv2.resize(image, self.img_dim, interpolation=cv2.INTER_LINEAR)
        self.img = image

    def __getitem__(self, idx):
        image = self.img / 255

        grid = create_grid(*self.img_dim[::-1])

        return grid, torch.tensor(image, dtype=torch.float32)

    def __len__(self):
        return 1



class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class NetLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()

    def init_weights(self):
        b = 1 / \
            self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.w0 * x)


def input_mapping(x, B):
    if B is None:
        return x
    else:
        x_proj = (2. * np.pi * x) @ B.t()
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


def make_network(num_layers, input_dim, hidden_dim):
    layers = [nn.Linear(input_dim, hidden_dim), Swish()]
    for i in range(1, num_layers - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(Swish())

    layers.append(nn.Linear(hidden_dim, 3))
    layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


def rff_model(num_layers, input_dim, hidden_dim):
    layers = [NetLayer(input_dim, hidden_dim, is_first=True)]
    for i in range(1, num_layers - 1):
        layers.append(NetLayer(hidden_dim, hidden_dim))
    layers.append(NetLayer(hidden_dim, 3, is_last=True))

    return nn.Sequential(*layers)


def train_model(network_size, learning_rate, iters, B, train_data, test_data, device="cpu"):
    model = rff_model(*network_size).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    train_psnrs = []
    test_psnrs = []
    xs = []
    for i in tqdm(range(iters), desc='train iter', leave=False):
        model.train()
        optim.zero_grad()

        t_o = model(input_mapping(train_data[0], B))
        t_loss = .5 * loss_fn(t_o, train_data[1])

        t_loss.backward()
        optim.step()

        # print(f"---[steps: {i}]: train loss: {t_loss.item():.6f}")

        train_psnrs.append(- 10 * torch.log10(2 * t_loss).item())

        if i % 25 == 0:
            model.eval()
            with torch.no_grad():
                v_o = model(input_mapping(test_data[0], B))
                v_loss = loss_fn(v_o, test_data[1])
                v_psnrs = - 10 * torch.log10(2 * v_loss).item()
                test_psnrs.append(v_psnrs)
                xs.append(i)
                torchvision.utils.save_image(v_o.permute(0, 3, 1, 2), f"imgs/{i}_{v_loss.item():.6f}.jpeg")
            # print(f"---[steps: {i}]: valid loss: {v_loss.item():.6f}")

    return {
        'state': model.state_dict(),
        'train_psnrs': train_psnrs,
        'test_psnrs': test_psnrs,
    }


if __name__ == '__main__':
    device = "cuda:0"

    network_size = (4, 512, 256)
    learning_rate = 1e-4
    iters = 500
    mapping_size = 256

    B_gauss = torch.randn((mapping_size, 2)).to(device) * 10

    ds = ImageDataset("models.jpg", 512)

    grid, image = ds[0]
    grid = grid.unsqueeze(0).to(device)
    image = image.unsqueeze(0).to(device)

    test_data = (grid, image)
    train_data = (grid[:, ::2, ::2], image[:, ::2, :: 2])

    output = train_model(network_size, learning_rate, iters, B_gauss,
                         train_data=train_data, test_data=(grid, image), device=device)


