# Uncomment in google colab to extract dataset
# !unzip KITTI_2015_subset.zip

import os
import sys
import argparse
import torch

import numpy as np

from matplotlib import pyplot as plt
from assets.stereo_batch_provider import KITTIDataset, PatchProvider
from scipy.signal import convolve

# Shortcuts
input_dir = './assets/KITTI_2015_subset'
window_size = 3
max_disparity = 50
out_dir = os.path.join(
    './output/handcrafted_stereo', 'window_size_%d' % window_size
)


###########################
##### Helper Function #####
###########################
def add_padding(I, padding):
    """
    Adds zero padding to an RGB or grayscale image.

    Args:
        I (np.ndarray): HxWx? numpy array containing RGB or grayscale image

    Returns:
        P (np.ndarray): (H+2*padding)x(W+2*padding)x? numpy array containing zero padded image
    """
    if len(I.shape) == 2:
        H, W = I.shape
        padded = np.zeros((H + 2 * padding, W + 2 * padding), dtype=np.float32)
        padded[padding:-padding, padding:-padding] = I
    else:
        H, W, C = I.shape
        padded = np.zeros((H + 2 * padding, W + 2 * padding, C), dtype=I.dtype)
        padded[padding:-padding, padding:-padding] = I

    return padded


def hinge_loss(score_pos, score_neg, label):
    """
    Computes the hinge loss for the similarity of a positive and a negative example.

    Args:
        score_pos (torch.Tensor): similarity score of the positive example
        score_neg (torch.Tensor): similarity score of the negative example
        label (torch.Tensor): the true labels

    Returns:
        avg_loss (torch.Tensor): the mean loss over the patch and the mini batch
        acc (torch.Tensor): the accuracy of the prediction
    """
    # Calculate the hinge loss max(0, margin + s_neg - s_pos)
    loss = torch.max(0.2 + score_neg - score_pos, torch.zeros_like(score_pos))

    # Obtain the mean over the patch and the mini batch
    avg_loss = torch.mean(loss)

    # Calculate the accuracy
    similarity = torch.stack([score_pos, score_neg], dim=1)
    labels = torch.argmax(label, dim=1)
    predictions = torch.argmax(similarity, dim=1)
    acc = torch.mean((labels == predictions).float())

    return avg_loss, acc


def training_loop(infer_similarity_metric, patches, optimizer, iterations=1000, batch_size=128):
    '''
    Runs the training loop of the siamese network.

    Args:
        infer_similarity_metric (obj): pytorch module
        patches (obj): patch provider object
        optimizer (obj): optimizer object
        iterations (int): number of iterations to perform
        batch_size (int): batch size
    '''

    loss_list = []
    try:
        print("Starting training loop.")
        for idx, batch in zip(range(iterations), patches.iterate_batches(batch_size)):
            # Extract the batches and labels
            Xl, Xr_pos, Xr_neg = batch
            # uncomment if you don't have a gpu
            # Xl, Xr_pos, Xr_neg = Xl.cpu(), Xr_pos.cpu(), Xr_neg.cpu()

            # use this line if you have a gpu
            label = torch.eye(2).cuda()[[0] * len(Xl)]  # label is always [1, 0]
            # use this line if you don't have a gpu
            # label = torch.eye(2)[[0]*len(Xl)]  # label is always [1, 0]

            # calculate the similarity score
            score_pos = calculate_similarity_score(infer_similarity_metric, Xl, Xr_pos)
            score_neg = calculate_similarity_score(infer_similarity_metric, Xl, Xr_neg)
            # compute the loss and accuracy
            loss, acc = hinge_loss(score_pos, score_neg, label)

            # compute the gradients
            optimizer.zero_grad()
            loss.backward()
            # let the optimizer perform one step and update the weights
            optimizer.step()

            # Append loss to list
            loss_list.append(loss.item())

            if idx % 50 == 0:
                print("Loss (%04d it):%.04f \tAccuracy: %0.3f" % (idx, loss, acc))
    finally:
        patches.stop()
        print("Finished training!")


def compute_disparity_CNN(infer_similarity_metric, img_l, img_r, max_disparity=50):
    """
    Computes the disparity of the stereo image pair.

    Args:
        infer_similarity_metric:  pytorch module object
        img_l: tensor holding the left image
        img_r: tensor holding the right image
        max_disparity (int): maximum disparity

    Returns:
        D: tensor holding the disparity
    """
    # get the image features by applying the similarity metric
    Fl = infer_similarity_metric(img_l[None])
    Fr = infer_similarity_metric(img_r[None])

    # images of shape B x H x W x C
    B, H, W, C = Fl.shape
    # Initialize the disparity
    disparity = torch.zeros((B, H, W)).int()
    # Initialize current similarity to -infimum
    current_similarity = torch.ones((B, H, W)) * -np.inf

    # Loop over all possible disparity values
    Fr_shifted = Fr
    for d in range(max_disparity + 1):
        if d > 0:
            # initialize shifted right image
            Fr_shifted = torch.zeros_like(Fr)
            # insert values which are shifted to the right by d
            Fr_shifted[:, :, d:] = Fr[:, :, :-d]

        # Calculate similarities
        sim_d = torch.sum(Fl * Fr_shifted, dim=3)
        # Check where similarity for disparity d is better than current one
        indices_pos = sim_d > current_similarity
        # Enter new similarity values
        current_similarity[indices_pos] = sim_d[indices_pos]
        # Enter new disparity values
        disparity[indices_pos] = d

    return disparity


###########################
#### Exercise Function ####
###########################
def sad(image_left, image_right, window_size=3, max_disparity=50):
    """
    Compute the sum of absolute differences between image_left and image_right.

    Args:
        image_left (np.ndarray): HxW numpy array containing grayscale right image
        image_right (np.ndarray): HxW numpy array containing grayscale left image
        window_size: window size (default 3)
        max_disparity: maximal disparity to reduce search range (default 50)

    Returns:
        D (np.ndarray): HxW numpy array containing the disparity for each pixel
    """

    D = np.zeros_like(image_left)
    height = image_left.shape[0]
    width = image_left.shape[1]
    # add zero padding
    padding = window_size // 2
    image_left = add_padding(image_left, padding).astype(np.float32)
    image_right = add_padding(image_right, padding).astype(np.float32)

    for y in range(height):
        for x in range(width):
            # 左边图像的窗口
            window_left = image_left[y:y + window_size, x:x + window_size]
            best_disparity = 0
            min_sad = float('inf')
            for d in range(max_disparity):
                if x - d < 0:
                    continue
                # 一定是减去d，因为右边图像是左边图像向右平移d个像素
                window_right = image_right[y:y + window_size, x - d:x - d + window_size]
                now_sad = np.sum(np.abs(window_left - window_right))

                if now_sad < min_sad:
                    min_sad = now_sad
                    best_disparity = d

            # 保存SAD
            D[y, x] = best_disparity

    return D


###########################
##### Bonus Function #####
###########################
def sad_convolve(image_left, image_right, window_size=3, max_disparity=50):
    """
    Compute the sum of absolute differences between image_left and image_right
    by using a mean filter.

    Args:
        image_left (np.nfarray): HxW numpy array containing grayscale right image
        image_right (np.nfarray): HxW numpy array containing grayscale left image
        window_size: window size (default 3)
        max_disparity: maximal disparity to reduce search range (default 50)

    Returns:
        D (np.ndarray): HxW numpy array containing the disparity for each pixel
    """

    padding = window_size // 2
    image_left = add_padding(image_left, padding).astype(np.float32)
    image_right = add_padding(image_right, padding).astype(np.float32)
    SAD = np.zeros((image_left.shape[0], image_left.shape[1], max_disparity + 1))

    kernel = np.ones((window_size, window_size)) / (window_size ** 2)

    for d in range(0, max_disparity + 1):
        if d == 0:
            right_shifted = image_right
        else:
            right_shifted = np.zeros_like(image_right)
            right_shifted[:, d:] = image_right[:, :-d]

        img_diff = np.abs(image_left - right_shifted)
        img_sad = convolve(img_diff, kernel, mode='same')  # 通过卷积运算，可以计算出每个像素邻域的总差异，也就是SAD值
        SAD[:, :, d] = img_sad

    D = np.argmin(SAD, axis=2)
    return D


def visualize_disparity(disparity, im_left, im_right, title='Disparity Map', max_disparity=50):
    """
    Generates a visualization for the disparity map.

    Args:
        disparity (np.array): disparity map
        title: plot title
        out_file: output file path
        max_disparity: maximum disparity
    """
    # 显示在一个窗口只显示一张原图和一张视差图
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(121)
    ax1.imshow(im_left.squeeze(), cmap='gray')
    ax1.title.set_text('Left Image')
    ax2 = fig.add_subplot(122)
    ax2.imshow(disparity.squeeze(), cmap='jet', vmax=max_disparity)
    ax2.title.set_text(title)
    plt.show()


class StereoMatchingNetwork(torch.nn.Module):
    def __init__(self):
        """
        Implementation of the network layers.
        Layer output tensor size: (batch_size, n_features, height - 8, width - 8)
        """
        super().__init__()
        gpu = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.conv1 = torch.nn.Conv2d(1, 64, 3)
        self.conv2 = torch.nn.Conv2d(64, 64, 3)
        self.conv3 = torch.nn.Conv2d(64, 64, 3)
        self.conv4 = torch.nn.Conv2d(64, 64, 3)

        # Hint: Don't forget to move the modules to the gpu
        self.to(gpu)

    def forward(self, X):
        """
        The forward pass of the network. Returns the features for a given image patch.

        Args:
            X (torch.Tensor): image patch of shape (batch_size, height, width, n_channels)

        Returns:
            features (torch.Tensor): predicted normalized features of the input image patch X,
                               shape (batch_size, height - 8, width - 8, n_features)
        """
        import torch.nn.functional as f
        X = X.permute(0, 3, 1, 2)

        X = self.conv1(X)
        X = f.relu(X)
        X = self.conv2(X)
        X = f.relu(X)
        X = self.conv3(X)
        X = f.relu(X)
        X = self.conv4(X)
        X = f.relu(X)

        X = f.normalize(X)
        X = X.permute(0, 2, 3, 1)
        features = X

        return features


def calculate_similarity_score(infer_similarity_metric, Xl, Xr):
    """
    Computes the similarity score for two stereo image patches.

    Args:
        infer_similarity_metric (torch.nn.Module):  pytorch module object
        Xl (torch.Tensor): tensor holding the left image patch
        Xr (torch.Tensor): tensor holding the right image patch

    Returns:
        score (torch.Tensor): the similarity score of both image patches which is the dot product of their features
    """
    l_features = infer_similarity_metric(Xl)
    r_features = infer_similarity_metric(Xr)
    # features shape (batch_size, height - 8, width - 8, n_features)

    # 按照n_features维度进行点乘
    score = torch.sum(l_features * r_features, dim=-1).squeeze()

    return score


if __name__ == '__main__':
    # Create output directory
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Load dataset
    dset = KITTIDataset(os.path.join(input_dir, "data_scene_flow/testing/"))

    # fig = plt.figure(figsize=(15, 10))
    # ax1 = fig.add_subplot(121)
    # ax1.imshow(dset[0][0].squeeze(), cmap='gray')
    # ax1.title.set_text('Left Image')
    # ax2 = fig.add_subplot(122)
    # ax2.imshow(dset[0][1].squeeze(), cmap='gray')
    # ax2.title.set_text('Right Image')
    # plt.show()

    # for i in range(1):
    #     # Load left and right images
    #     im_left, im_right = dset[i]
    #     im_left, im_right = im_left.squeeze(-1), im_right.squeeze(-1)
    #
    #     # Calculate disparity
    #     D = sad_convolve(im_left, im_right, window_size=window_size, max_disparity=max_disparity)
    #
    #     # Define title for the plot
    #     title = 'Disparity map for image %04d with block matching (window size %d)' % (i, window_size)
    #     # Define output file name and patch
    #     file_name = '%04d_w%03d.png' % (i, window_size)
    #     out_file_path = os.path.join(out_dir, file_name)
    #
    #     # Visualize the disparty and save it to a file
    #     visualize_disparity(D, im_left, im_right, title=title, max_disparity=max_disparity)

    # Fix random seed for reproducibility
    np.random.seed(7)
    torch.manual_seed(7)

    # Shortcuts for directories
    model_out_dir = os.path.join(out_dir, 'model')
    model_file = os.path.join(model_out_dir, "model.t7")

    # Hyperparameters
    training_iterations = 1000
    batch_size = 128
    learning_rate = 3e-4
    patch_size = 9
    padding = patch_size // 2
    max_disparity = 50

    # Check if output directory exists and if not create it
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)

    # Create dataloader for KITTI training set
    dataset = KITTIDataset(
        os.path.join(input_dir, "data_scene_flow/training/"),
        os.path.join(input_dir, "data_scene_flow/training/disp_noc_0"),
    )
    # Load patch provider
    patches = PatchProvider(dataset, patch_size=(patch_size, patch_size))

    # Initialize the network
    infer_similarity_metric = StereoMatchingNetwork()
    # Set to train
    infer_similarity_metric.train()
    # uncomment if you don't have a gpu
    # infer_similarity_metric.to('cpu')
    optimizer = torch.optim.SGD(infer_similarity_metric.parameters(), lr=learning_rate, momentum=0.9)

    # Start training loop
    training_loop(infer_similarity_metric, patches, optimizer,
                  iterations=10000, batch_size=128)

    # Set network to eval mode
    infer_similarity_metric.eval()
    infer_similarity_metric.to('cpu')

    # Load KITTI test split
    dataset = KITTIDataset(os.path.join(input_dir, "data_scene_flow/testing/"))
    # Loop over test images
    for i in range(len(dataset)):
        print('Processing %d image' % i)
        # Load images and add padding
        img_left, img_right = dataset[i]
        img_left_padded, img_right_padded = add_padding(img_left, padding), add_padding(img_right, padding)
        img_left_padded, img_right_padded = torch.Tensor(img_left_padded), torch.Tensor(img_right_padded)

        disparity_map = compute_disparity_CNN(
            infer_similarity_metric, img_left_padded, img_right_padded, max_disparity=max_disparity
        )
        # Define title for the plot
        title = 'Disparity map for image %04d with SNN \
            (training iterations %d, batch size %d, patch_size %d)' % (i, training_iterations, batch_size, patch_size)
        visualize_disparity(disparity_map.squeeze(), img_left.squeeze(), img_right.squeeze(), title,
                            max_disparity=max_disparity)
