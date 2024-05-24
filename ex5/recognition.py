import os
import numpy as np
from PIL import Image
import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from skimage.feature import hog
from sklearn.svm import SVC
import faiss  # fast kNN search

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# set random seed for reproducability
np.random.seed(42)


###########################
##### Helper Function #####
###########################
class PennFudanDataset(torch.utils.data.Dataset):
    ''' Penn-Fudan Pedestrian dataset.

    Args:
    root (str): path to data directory
    split (str): dataset split, "train" or "val"
    '''

    def __init__(self, root, split='train'):
        self.root = root

        # load all image files, sorting them to
        # ensure that they are aligned
        imgs = list(sorted(os.listdir(os.path.join(root, "PennFudanPed", "PNGImages"))))
        masks = list(sorted(os.listdir(os.path.join(root, "PennFudanPed", "PedMasks"))))

        # split into train and validation set
        ntrain = int(0.8 * len(imgs))
        if split == 'train':
            self.imgs = imgs[:ntrain]
            self.masks = masks[:ntrain]
        elif split == 'val':
            self.imgs = imgs[ntrain:]
            self.masks = masks[ntrain:]
        else:
            raise AttributeError('split must be "train" or "val".')

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PennFudanPed", "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PennFudanPed", "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background

        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        return img, {'boxes': boxes}

    def __len__(self):
        return len(self.imgs)


###########################
##### Helper Function #####
###########################
def get_width_and_height(box):
    """ Helper function to determine size of bounding boxes.

    Args:
    box (iterable): (left, upper, right, lower) pixel coordinate
    """
    return box[2] - box[0], box[3] - box[1]


def draw_box(ax, box, color='r'):
    ''' Plot box on axes.

    Args:
    ax (matplotlib.axes.Axes): axes to add box to
    box (iterable): (left, upper, right, lower) pixel coordinate
    color (str or list): edgecolor of the box
    '''
    anchor = box[:2]
    W, H = get_width_and_height(box)
    patch = Rectangle(anchor, width=W, height=H, edgecolor=color, facecolor='none')
    ax.add_patch(patch)


def get_resized_patch(img, box, patch_size):
    ''' Crop patch from image and resize it to given size.

    Args:
    img (PIL.Image.Image): image
    box (iterable): (left, upper, right, lower) pixel coordinate
    patch_size (tuple): width, height of resized patch
    '''
    assert isinstance(img, Image.Image), 'img needs to be PIL.Image.Image'
    crop = img.crop(box)
    patch = crop.resize(patch_size)
    return patch


def imgs_to_patches(train_set, patch_size, n_random_negatives=3, n_hard_negatives=3):
    positives = []
    negatives = []
    orig_sizes = []
    for img, annotation in tqdm(train_set):
        for box in annotation['boxes']:

            # keep track of original box sizes for sliding window approach later
            orig_sizes.append(np.array(get_width_and_height(box)))

            # extract positives
            positives.append(get_resized_patch(img, box, patch_size))

            # add some random negatives
            for _ in range(n_random_negatives):
                boxsize = get_width_and_height(box)
                mod_box = get_random_box(boxsize, img.size)
                negatives.append(get_resized_patch(img, mod_box, patch_size))

            # add some hard negatives by adding noise on box
            for _ in range(n_hard_negatives):
                mod_box = add_offset_to_box(box, img.size)
                negatives.append(get_resized_patch(img, mod_box, patch_size))

    return positives, negatives, orig_sizes


def get_hog(img, **kwargs):
    """ Extract HOG features with predefined settings.

    Args:
    img (np.ndarray): image with channel dimension last (HxWx3)
    """
    return hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), channel_axis=-1, **kwargs)


def init_nn_search(X):
    """ Initialize the index for a nearest neighbor search.

    Args:
    X (np.ndarray): training data
    """
    d = X.shape[1]
    quantizer = faiss.IndexFlatL2(d)  # measure L2 distance
    index = faiss.IndexIVFFlat(quantizer, d, 100, faiss.METRIC_L2)  # build the index

    index.train(X.astype(np.float32))
    index.add(X.astype(np.float32))  # add vectors to the index
    return index


def train_svm(X, y):
    """ Train a support vector machine.

    Args:
    X (np.ndarray): training data
    y (np.ndarray): training labels
    """
    clf = SVC(class_weight='balanced')  # use balanced weight since we have more negatives than positives
    clf.fit(X, y)
    return clf


###########################
#### Exercise Function ####
###########################
def get_random_box(boxsize, imsize):
    """ Returns randomly located box with same size as box.

    Args:
    boxsize (tuple): width, height of box
    imsize (tuple): width, height of image / image boundaries
    """
    W, H = imsize

    x = np.random.randint(0, W - boxsize[0])
    y = np.random.randint(0, H - boxsize[1])

    # （左、上、右、下）像素坐标
    box = [x, y, x + boxsize[0], y + boxsize[1]]

    assert all(b >= 0 for b in box) and (box[2] < W) and (box[3] < H), f'Box {box} out of image bounds {W, H}.'
    return box


def add_offset_to_box(box, imsize):
    """ Add a small random integer offset to the box.

    Args:
    box (iterable): (left, upper, right, lower) pixel coordinate
    imsize (tuple): width, height of image / image boundaries
    """
    W, H = imsize

    # 高度/宽度的最小偏移应为 1 像素，最大高度/宽度偏移不应超过框高度/宽度的 20%
    min_offset = 1
    max_offset_w = int(0.2 * (box[2] - box[0]))
    max_offset_h = int(0.2 * (box[3] - box[1]))

    off_x = np.random.randint(min_offset, max(2, max_offset_w))
    off_y = np.random.randint(min_offset, max(2, max_offset_h))

    off_box = [box[0] + off_x, box[1] + off_y, box[2] + off_x, box[3] + off_y]

    # ensure to stay within image boundaries / do not mind changing size slightly
    off_box[0] = max(0, off_box[0])
    off_box[1] = max(0, off_box[1])
    off_box[2] = min(W - 1, off_box[2])
    off_box[3] = min(H - 1, off_box[3])

    assert all(b >= 0 for b in off_box) and (off_box[2] < W) and (
            off_box[3] < H), f'Box {off_box} out of image bounds {W, H}.'

    return off_box


def img_to_hog_patches(img, window_size, patch_size, step_size=1):
    """ Extract hog feature patches from an image using a sliding window approach.

    Args:
    img (PIL.Image.Image): image
    window_size (tuple): width, height of window
    patch_size (tuple): width, height of resized patch
    step_size (int): step size of window (for faster evaluation)
    """

    W, H = img.size
    fds = []
    anchors = []

    # sliding window approach
    for y in range(0, H - window_size[1], step_size):
        for x in range(0, W - window_size[0], step_size):
            window = (x, y, x + window_size[0], y + window_size[1])
            patch = get_resized_patch(img, window, patch_size)
            fds.append(get_hog(np.array(patch)))
            anchor = (x, y)
            anchors.append(anchor)

    # convert to numpy arrays
    fds = np.stack(fds)
    anchors = np.stack(anchors)
    return fds, anchors


if __name__ == '__main__':
    train_set = PennFudanDataset('data', split='train')
    # for i in range(2):
    #     img, annotation = train_set[i]
    #     fig, ax = plt.subplots(1)
    #     ax.set_axis_off()
    #     ax.imshow(img)
    #     for box in annotation['boxes']:
    #         draw_box(ax, box)
    #     plt.show()  # 显示图像

    patch_size = (50, 150)  # hyperparameter
    positives, negatives, orig_sizes = imgs_to_patches(train_set, patch_size)
    # fig, ax = plt.subplots(1, 2)
    # ax[0].set_axis_off()
    # ax[0].imshow(np.concatenate([np.array(p) for p in positives[:5]], 1))
    #
    # ax[1].set_axis_off()
    # ax[1].imshow(np.concatenate([np.array(n) for n in negatives[:5]], 1))
    # plt.show()

    # visualize HOG for one image

    # img = np.array(positives[0])
    # fd, hog_image = get_hog(img, visualize=True)
    # if visualize=True this function returns the descriptors + the image, otherwise it only returns the descriptors

    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(img)
    # ax[1].imshow(hog_image, cmap=plt.cm.gray)
    # plt.show()

    fds_pos = []
    for p in tqdm(positives, position=0, leave=True, desc='Extract HOG features for positive samples'):
        fds_pos.append(get_hog(p))

    fds_neg = []
    for n in tqdm(negatives, position=0, leave=True, desc='Extract HOG features for negative samples'):
        fds_neg.append(get_hog(n))

    # 特征矩阵
    X = np.stack(fds_pos + fds_neg)
    # 目标向量
    y = np.concatenate([np.ones(len(positives), dtype=np.bool), np.zeros(len(negatives), dtype=np.bool)])

    index = init_nn_search(X)
    svm = train_svm(X, y)

    val_set = PennFudanDataset('data', split='val')

    N_imgs = 3
    window_size = np.stack(orig_sizes).mean(0).astype(np.uint)  # use average size of training boxes as window size

    for i in range(N_imgs):
        img, ann = val_set[i]
        fds, anchors = img_to_hog_patches(img, window_size, patch_size, step_size=10)

        # evaluate NN search
        _, I = index.search(fds.astype(np.float32), 1)  # search nearest neighbor for each grid point
        is_positive = y[I]  # assign labels of training points

        mask = is_positive.reshape(-1, 1).repeat(2, 1)  # convert to mask for anchors
        anchors_nn = anchors[mask].reshape(-1, 2)

        # evaluate SVM
        k = 5
        scores = svm.predict(fds)
        idcs_sorted = np.argsort(scores)[::-1][:k]  # sort get top k predictions
        anchors_svm = anchors[idcs_sorted]

        # visualize the results
        fig, ax = plt.subplots(2)

        ax[0].imshow(img)
        for a in anchors_nn:
            box = (a[0], a[1], a[0] + window_size[0], a[1] + window_size[1])
            draw_box(ax[0], box, color='r')

        ax[1].imshow(img)
        for j, a in enumerate(anchors_svm):
            box = (a[0], a[1], a[0] + window_size[0], a[1] + window_size[1])
            color = 'r' if j == 0 else 'orange'
            draw_box(ax[1], box, color=color)

        plt.show()
