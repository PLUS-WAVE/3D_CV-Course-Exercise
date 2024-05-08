import os
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from livelossplot import PlotLosses
from skimage.measure import \
    marching_cubes  # Lewiner et al. algorithm is faster, resolves ambiguities, and guarantees topologically correct results
import matplotlib.pyplot as plt

# set random seed for reproducibility
np.random.seed(42)

data_dir = 'data'
out_dir = 'output'

for d in [data_dir, out_dir]:
    os.makedirs(d, exist_ok=True)


###########################
##### Helper Function #####
###########################
def load_data(file_path):
    ''' Load points and occupancy values from file.

    Args:
    file_path (string): path to file
    '''
    data_dict = np.load(file_path)
    points = data_dict['points']
    occupancies = data_dict['occupancies']

    # Unpack data format of occupancies
    occupancies = np.unpackbits(occupancies)[:points.shape[0]]
    occupancies = occupancies.astype(np.float32)

    # Align z-axis with top of object
    points = np.stack([points[:, 0], -points[:, 2], points[:, 1]], 1)

    return points, occupancies


def visualize_occupancy(points, occupancies, n=10000000):
    ''' Visualize points and occupancy values.

    Args:
    points (torch.Tensor or np.ndarray): 3D coordinates of the points
    occupancies (torch.Tensor or np.ndarray): occupancy values for the points
    n (int): maximum number of points to visualize
    '''
    # if needed to convert torch.tensor to np.ndarray
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if isinstance(occupancies, torch.Tensor):
        occupancies = occupancies.detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    n = min(len(points), n)

    # visualize a random subset of n points
    idcs = np.random.randint(0, len(points), n)
    points = points[idcs]
    occupancies = occupancies[idcs]

    # define colors
    red = np.array([1, 0, 0, 0.5]).reshape(1, 4).repeat(n, 0)  # plot occupied points in red with alpha=0.5
    blue = np.array([0, 0, 1, 0.01]).reshape(1, 4).repeat(n, 0)  # plot free points in blue with alpha=0.01
    occ = occupancies.reshape(n, 1).repeat(4, 1)  # reshape to RGBA format to determine color

    color = np.where(occ == 1, red, blue)  # occ=1 -> red, occ=0 -> blue

    # plot the points
    ax.scatter(*points.transpose(), color=color)

    # make it pretty
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)

    plt.show()


def make_grid(xmin, xmax, resolution):
    """ Create equidistant points on 3D grid (cube shaped).

    Args:
    xmin (float): minimum for x,y,z
    xmax (float): number of hidden layers
    """
    grid_1d = torch.linspace(xmin, xmax, resolution)
    grid_3d = torch.stack(torch.meshgrid(grid_1d, grid_1d, grid_1d, indexing='xy'), -1)
    return grid_3d.flatten(0, 2)  # return as flattened tensor: RxRxRx3 -> (R^3)x3


###########################
#### Exercise Function ####
###########################
def get_train_val_split(points, occupancies):
    ''' Split data into train and validation set.

    Args:
    points (torch.Tensor or np.ndarray): 3D coordinates of the points
    occupancies (torch.Tensor or np.ndarray): occupancy values for the points
    '''

    # 随机分割数据，使用 80% 作为训练集，20% 作为验证集
    n = len(points)
    idx = np.arange(n)
    np.random.shuffle(idx)
    train_idx = idx[:int(n * 0.8)]
    val_idx = idx[int(n * 0.8):]

    train_points = points[train_idx]
    train_occs = occupancies[train_idx]
    val_points = points[val_idx]
    val_occs = occupancies[val_idx]

    # this converts the points and labels from numpy.ndarray to a pytorch dataset
    train_set = torch.utils.data.TensorDataset(torch.from_numpy(train_points).float(),
                                               torch.from_numpy(train_occs).float())
    val_set = torch.utils.data.TensorDataset(torch.from_numpy(val_points).float(), torch.from_numpy(val_occs).float())
    return train_set, val_set


class OccNet(nn.Module):
    """ Network to predict an occupancy value for every 3D location.

    Args:
    size_h (int): hidden dimension
    n_hidden (int): number of hidden layers
    """

    def __init__(self, size_h=64, n_hidden=4):
        super().__init__()

        # 4 个隐藏层、隐藏维度为 64（总共 6 层，4 个隐藏层 + 1 个输入 + 1 个输出）
        layers = [nn.Linear(3, size_h), nn.ReLU()]
        for _ in range(n_hidden):
            layers += [nn.Linear(size_h, size_h), nn.ReLU()]
        layers += [nn.Linear(size_h, 1)]

        self.main = nn.Sequential(*layers)

    def forward(self, pts):
        return self.main(pts).squeeze(-1)  # squeeze dimension of the single output value


def train_model(model, train_loader, val_loader, optimizer, criterion, nepochs=15, eval_every=100, out_dir='output'):
    liveloss = PlotLosses()  # to plot training progress
    losses = {'loss': None,
              'val_loss': None}

    best = float('inf')
    it = 0
    for epoch in range(nepochs):
        stime = time.time()
        losses['loss'] = []  # initialize emtpy container for training losses
        for pts, occ in train_loader:
            it += 1

            pts = pts.cuda()
            occ = occ.cuda()

            optimizer.zero_grad()  # 清零梯度，为计算新的梯度做准备
            output = model(pts)  # 前向传播，通过模型计算预测值
            loss = criterion(output, occ).mean()  # 计算损失
            loss.backward()  # 反向传播，计算每个参数的梯度
            optimizer.step()  # 更新权重，使用梯度进行一步优化算法

            losses['loss'].append(loss.item())

            if (it == 1) or (it % eval_every == 0):

                with torch.no_grad():
                    val_loss = []
                    for val_pts, val_occ in val_loader:
                        val_pts = val_pts.cuda()
                        val_occ = val_occ.cuda()
                        val_output = model(val_pts)
                        val_loss_i = criterion(val_output, val_occ)
                        val_loss.extend(val_loss_i)
                    val_loss = torch.stack(val_loss).mean().item()

                    if val_loss < best:  # keep track of best model
                        best = val_loss
                        torch.save(model.state_dict(), os.path.join(out_dir, 'model_best.pt'))
        etime = time.time()
        print(f'Epoch {epoch + 1}/{nepochs} ({etime - stime:.2f}s)')
        # update liveplot with latest values
        losses['val_loss'] = val_loss
        losses['loss'] = np.mean(losses['loss'])  # average over all training losses
        liveloss.update(losses)
        liveloss.send()


if __name__ == '__main__':
    points, occupancies = load_data('data/points.npz')
    visualize_occupancy(points, occupancies)

    train_set, val_set = get_train_val_split(points, occupancies)

    batch_size = 1024

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, num_workers=1, shuffle=True, drop_last=True
        # randomly shuffle the training data and potentially drop last batch
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, num_workers=1, shuffle=False, drop_last=False
        # do not shuffle validation set and do not potentially drop last batch
    )

    model = OccNet(size_h=64, n_hidden=4)

    # put the model on the GPU to accelerate training
    if torch.cuda.is_available():
        model = model.cuda()
    else:
        print('Fall back to CPU - GPU usage is recommended, e.g. using Google Colab.')

    criterion = nn.BCEWithLogitsLoss(reduction='none')  # binary cross entropy loss + softmax
    optimizer = torch.optim.Adam(model.parameters())

    train_model(model, train_loader, val_loader, optimizer, criterion, nepochs=45, eval_every=100, out_dir=out_dir)

    resolution = 128  # use 128 grid points in each of the three dimensions -> 128^3 query points
    grid = make_grid(-0.5, 0.5, resolution)

    # wrap query points in data loader
    test_loader = torch.utils.data.DataLoader(
        grid, batch_size=1024, num_workers=1, shuffle=False, drop_last=False
    )
    weights_best = torch.load(
        os.path.join(out_dir, 'model_best.pt'))  # we saved the best model there in the training loop
    model.load_state_dict(weights_best)

    grid_values = []
    with torch.no_grad():
        for pts in tqdm(test_loader):
            pts = pts.cuda()
            output = model(pts)
            grid_values.append(output)

    grid_values = torch.cat(grid_values, 0).cpu()
    grid_occupancies = grid_values > 0.  # convert model scores to classification score
    visualize_occupancy(grid, grid_occupancies)

    # extract mesh with Marching Cubes
    threshold = 0.  # because grid values are model scores
    assert (grid_values.min() <= threshold) and (
                grid_values.max() >= threshold), "Threshold is not in range of predicted values"

    vertices, faces, _, _ = marching_cubes(grid_values.reshape(resolution, resolution, resolution).numpy(),
                                           threshold,
                                           spacing=(1 / (resolution - 1), 1 / (resolution - 1), 1 / (resolution - 1)),
                                           allow_degenerate=False)

    # plot mesh
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(vertices[:, 1], vertices[:, 0], triangles=faces, Z=vertices[:, 2])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    plt.show()
