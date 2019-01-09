import os
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.filters import gaussian

from skimage import color
from skimage import io

print("gigel")

df = pd.read_csv('../ShipDetection/5percent/data.csv')
path_train = '../input/train/'
path_test = '../input/test/'
train_ids = df.ImageId.values
df = df.set_index('ImageId')

tile_size = (500, 500)


def get_filename(image_id, image_type):
    check_dir = False
    if "Train" == image_type:
        data_path = "../ShipDetection/5percent"
    else:
        raise Exception("Image type '%s' is not recognized" % image_type)

    if check_dir and not os.path.exists(data_path):
        os.makedirs(data_path)

    return os.path.join(data_path, "{}".format(image_id))


def get_image_data(image_id, image_type, **kwargs):
    img = _get_image_data_opencv(image_id, image_type, **kwargs)
    img = img.astype('uint8')
    return img


def _get_image_data_opencv(image_id, image_type, **kwargs):
    fname = get_filename(image_id, image_type)
    img = cv2.imread(fname)
    assert img is not None, "Failed to read image : %s, %s" % (image_id, image_type)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


# https://github.com/ternaus/TernausNet/blob/master/Example.ipynb
def mask_overlay(image, mask):
    """
    Helper function to visualize mask
    """
    mask = mask.astype(np.uint8)
    weighted_sum = cv2.addWeighted(mask, 0.75, image, 0.5, 0.)
    img = image.copy()
    ind = mask[:, :, 1] > 0
    img[ind] = weighted_sum[ind]
    return img


_train_ids = list(train_ids)
np.random.seed(time.gmtime())
fig = plt.figure(1, figsize=(15, 15))
x = 10
y = 4
for i in range(x):
    image_id = _train_ids[np.random.randint(0, len(_train_ids))]

    ax = fig.add_subplot(x, y, i * y + 1)
    img = get_image_data(image_id, 'Train')
    img = cv2.resize(img, dsize=tile_size)
    ax.imshow(img)
    ax.set_title('Original')

    ax = fig.add_subplot(x, y, i * y + 2)
    img = get_image_data(image_id, 'Train')
    im = gaussian(img)
    img = cv2.resize(img, dsize=tile_size)
    ax.imshow(img)
    ax.set_title('Smoothed Image')

    ax = fig.add_subplot(x, y, i * y + 3)
    img = cv2.imread(get_filename(image_id, 'Train'), cv2.IMREAD_GRAYSCALE)
    ax.imshow(img, 'gray')
    ax.set_title('Gray')

    img = cv2.imread(get_filename(image_id, 'Train'), cv2.IMREAD_GRAYSCALE)
    ax = fig.add_subplot(x, y, i * y + 4)
    ax.set_title('Hist')
    ax.hist(img.ravel(), 256, [0, 256])

plt.show()
plt.suptitle('Smoothed Images')