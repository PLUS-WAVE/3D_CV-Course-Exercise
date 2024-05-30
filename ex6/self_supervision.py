import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets


###########################
##### Helper Function #####
###########################
def get_transform():
    '''
    returns a transform that randomly crops, flips, jitters color or drops color from the input
    '''
    return transforms.Compose([
        transforms.RandomResizedCrop(32, scale=[0.75, 1.0],
                                     interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                    saturation=0.2, hue=0.1)],
            p=0.9
        ),
        transforms.RandomGrayscale(p=0.3),
    ])


class BarlowTwins(nn.Module):
    '''
    Full Barlow Twins model with encoder, projector and loss
    '''

    def __init__(self, encoder, projector, lambd):
        '''
        :param encoder: encoder network
        :param projector: projector network
        :param lambd: tradeoff function (hyper-parameter)
        '''
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.lambd = lambd

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(128, affine=False)

    def forward(self, y1, y2):
        z1 = self.encoder(y1)
        z2 = self.encoder(y2)
        z1 = self.projector(z1)
        z2 = self.projector(z2)

        return barlow_loss(z1, z2, self.bn, self.lambd)


cifar_train_mean = [125.30691805, 122.95039414, 113.86538318]
cifar_train_std = [62.99321928, 62.08870764, 66.70489964]


class Transform:
    def __init__(self, t1, t2):
        '''
        :param t1: Transforms to be applied to first input
        :param t2: Transforms to be applied to second input
        '''
        cifar_train_mean = [125.30691805, 122.95039414, 113.86538318]
        cifar_train_std = [62.99321928, 62.08870764, 66.70489964]
        self.t1 = transforms.Compose([
            t1,
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar_train_mean, std=cifar_train_std)
        ])
        self.t2 = transforms.Compose([
            t2,
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar_train_mean, std=cifar_train_std)
        ])

    def __call__(self, x):
        y1 = self.t1(x)
        y2 = self.t2(x)
        return y1, y2


###########################
#### Exercise Function ####
###########################
def predict_knn(sample, train_data, train_labels, k):
    '''
    returns the predicted label for a specific validation sample

    :param sample: single example from validation set
    :param train_data: full training set as a single array
    :param train_labels: full set of training labels and a single array
    :param k: number of nearest neighbors used for k-NN voting
    '''

    # 对于每个验证样本，获取到每个训练样本的 L1 距离
    data = train_data.reshape(NUM_SAMPLES, -1)
    distances = np.sum(np.abs(data - sample.flatten()), axis=1)

    # 过滤与每个验证样本具有最小 L1 距离的 k 个训练样本
    k_indices = np.argsort(distances)[:k]

    # 检查这k个过滤后的训练样本的标签，并使用多数票进行分类
    k_labels = train_labels[k_indices]
    unique, counts = np.unique(k_labels, return_counts=True)
    return unique[np.argmax(counts)]


def off_diagonal(x):
    '''
    returns a flattened view of the off-diagonal elements of a square matrix x
    '''
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, m + 1)[:, 1:].flatten()


def barlow_loss(z1, z2, bn, lambd):
    '''
    return the barlow twins loss function for a pair of features. Makes use of the off_diagonal function.

    :param z1: first input feature
    :param z2: second input feature
    :param bn: nn.BatchNorm1d layer applied to z1 and z2
    :param lambd: trade-off hyper-parameter lambda
    '''

    z1_norm = bn(z1)
    z2_norm = bn(z2)

    batch_size = z1.size(0)

    # 计算 z1 和 z2 的协方差矩阵
    c = torch.mm(z1_norm, z2_norm.t()) / batch_size

    # loss
    c_diff = (c - torch.eye(c.size(0), device=c.device)).pow(2)
    c_diff = off_diagonal(c_diff).mul_(lambd)
    loss = c_diff.sum()

    return loss


class Projector(nn.Module):
    '''
    2-layer neural network (512 -> 256), (256 -> 128), ReLU non-linearity
    '''

    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        return self.main(x)


if __name__ == '__main__':
    train_set = datasets.CIFAR10(root='./data', train=True, download=True)
    val_set = datasets.CIFAR10(root='./data', train=False)
    print(f"Total training examples: {len(train_set)}")
    print(f"Total validation examples: {len(val_set)}")

    NUM_SAMPLES = 1000

    # 1000 random (image, label) pairs from train set
    train_indices = random.sample(range(len(train_set)), k=NUM_SAMPLES)
    train_subset = train_set.data[train_indices]
    train_subset_labels = np.array(train_set.targets)[train_indices]

    # 1000 random (image, label) pairs from validation set
    val_indices = random.sample(range(len(val_set)), k=NUM_SAMPLES)
    val_subset = val_set.data[val_indices]
    val_subset_labels = np.array(val_set.targets)[val_indices]

    predictions_7 = []
    predictions_13 = []
    predictions_19 = []
    for sample in val_subset:
        predictions_7.append(predict_knn(sample, train_subset, train_subset_labels, k=7))
        predictions_13.append(predict_knn(sample, train_subset, train_subset_labels, k=13))
        predictions_19.append(predict_knn(sample, train_subset, train_subset_labels, k=19))

    matches_7 = (np.array(predictions_7) == val_subset_labels)
    accuracy_7 = np.sum(matches_7) / NUM_SAMPLES * 100
    print(f"k-NN accuracy (k=7): {accuracy_7}%")

    matches_13 = (np.array(predictions_13) == val_subset_labels)
    accuracy_13 = np.sum(matches_13) / NUM_SAMPLES * 100
    print(f"k-NN accuracy (k=13): {accuracy_13}%")

    matches_19 = (np.array(predictions_19) == val_subset_labels)
    accuracy_19 = np.sum(matches_19) / NUM_SAMPLES * 100
    print(f"k-NN accuracy (k=19): {accuracy_19}%")

    t1 = get_transform()
    t2 = get_transform()

    # Hyper-parameters
    EPOCHS = 10
    LR = 0.001
    BATCH = 1024
    LAMBDA = 5e-3

    # Initialize encoder, projector and full model
    encoder = models.resnet18(weights=None)
    encoder.fc = nn.Identity()  # removes the 1000-dimensional classification layer
    projector = Projector()
    twins = BarlowTwins(encoder, projector, LAMBDA).cuda()

    # Dataset and optimizer
    dataset = datasets.CIFAR10(root='./data', train=True, transform=Transform(t1, t2))
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=BATCH,
                                         num_workers=4,
                                         shuffle=True)
    optimizer = torch.optim.Adam(twins.parameters(), lr=LR)

    # Training loop
    for epoch in range(EPOCHS):
        for batch_idx, ((x1, x2), _) in enumerate(loader):
            loss = twins(x1.cuda(), x2.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch + 1}, Loss: {float(loss)}")

    # Dataloaders for extracting self-supervised features
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar_train_mean, std=cifar_train_std)
    ])

    train_set = datasets.CIFAR10(root='./data', train=True, transform=test_transform)
    val_set = datasets.CIFAR10(root='./data', train=False, transform=test_transform)

    train_subset_torch = torch.utils.data.Subset(train_set, train_indices)
    val_subset_torch = torch.utils.data.Subset(val_set, val_indices)

    train_loader = torch.utils.data.DataLoader(train_subset_torch,
                                               batch_size=NUM_SAMPLES,
                                               shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_subset_torch,
                                             batch_size=NUM_SAMPLES,
                                             shuffle=False)

    # Extract features with the trained encoder
    # We use a single batch of size 1000
    for batch in train_loader:
        train_features = encoder(batch[0].cuda()).data.cpu().numpy()

    for batch in val_loader:
        val_features = encoder(batch[0].cuda()).data.cpu().numpy()

    predictions_7 = []
    predictions_13 = []
    predictions_19 = []
    for sample in val_features:
        predictions_7.append(predict_knn(sample, train_features, train_subset_labels, k=7))
        predictions_13.append(predict_knn(sample, train_features, train_subset_labels, k=13))
        predictions_19.append(predict_knn(sample, train_features, train_subset_labels, k=19))

    matches_7 = (np.array(predictions_7) == val_subset_labels)
    accuracy_7 = np.sum(matches_7) / NUM_SAMPLES * 100
    print(f"k-NN accuracy (k=7): {accuracy_7}%")

    matches_13 = (np.array(predictions_13) == val_subset_labels)
    accuracy_13 = np.sum(matches_13) / NUM_SAMPLES * 100
    print(f"k-NN accuracy (k=13): {accuracy_13}%")

    matches_19 = (np.array(predictions_19) == val_subset_labels)
    accuracy_19 = np.sum(matches_19) / NUM_SAMPLES * 100
    print(f"k-NN accuracy (k=19): {accuracy_19}%")
