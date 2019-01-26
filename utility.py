import os
import pandas as pd
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

df = pd.read_csv('../ShipDetection/5percent/_data.csv')
path_train = '../input/train/'
path_test = '../input/test/'
train_ids = df.ImageId.values
df = df.set_index('ImageId')

tile_size = (512, 512)

exclude_list = ['6384c3e78.jpg', '13703f040.jpg', '14715c06d.jpg',  '33e0ff2d5.jpg',
                '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg',
                'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
                'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg'] #corrupted image


def get_filename(image_id, image_type):
    check_dir = False
    if "Train" == image_type:
        data_path = "../ShipDetection/TrainFull"
    elif "Test" == image_type:
        data_path = "../ShipDetection/TestFull"
    elif "10k" == image_type:
        data_path = "../_10kRun/BeforeEM"
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
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
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


def pgk_xt(pixel, miu_list, sigma_list, pg_list, k, cluster_number):
    sum = 0.0
    pgk = 0.0
    for i in range(cluster_number):
        val = gauss_pdf(pixel, miu_list[i], sigma_list[i]) * pg_list[i]
        sum = sum + val
        if i == k:
            pgk = val
    return pgk / sum


def gauss_pdf(pixel, miu, sigma):
    return 1.0 / (math.sqrt(2.0 * math.pi * sigma)) \
           * math.pow(math.e, - (1.0 / 2.0) * ((pixel - miu) ** 2) / sigma)


def p_xt(pixel, miu_list, sigma_list, pg_list, cluster_number):
    sum_p = 0.0
    for i in range(cluster_number):
        sum_p = sum_p + gauss_pdf(pixel, miu_list[i], sigma_list[i]) * pg_list[i]
    return sum_p


def create_prediction_image(img, miu, sigma, pgk, clusters):
    copy_img = img.copy()
    height, width = copy_img.shape
    for i in range(height):
        for j in range(width):
            if pgk_xt(copy_img[i, j], miu, sigma, pgk, 0, clusters) > pgk_xt(copy_img[i, j], miu, sigma, pgk, 1,
                                                                             clusters):
                copy_img[i, j] = 1
            else:
                copy_img[i, j] = 0
    return copy_img


def create_prediction_matrix(img, miu, sigma, pgk, clusters):
    res = []
    height, width = img.shape
    print(width)
    print(height)
    for i in range(height):
        line = []
        for j in range(width):
            if pgk_xt(img[i, j], miu, sigma, pgk, 0, clusters) > pgk_xt(img[i, j], miu, sigma, pgk, 1,
                                                                        clusters):
                line.append(1)
            else:
                line.append(0)
        res.append(line)
    return res


def plot_image(img):
    fig = plt.figure(1, figsize=(15, 15))
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(img)
    ax.set_title('EM')

    plt.show()
