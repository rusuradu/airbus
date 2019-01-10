import os
import time
import math
import random


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.filters import gaussian
from matplotlib.backends.backend_pdf import PdfPages

from skimage import color
from skimage import io

print("gigel")

df = pd.read_csv('../ShipDetection/5percent/_data.csv')
path_train = '../input/train/'
path_test = '../input/test/'
train_ids = df.ImageId.values
df = df.set_index('ImageId')

tile_size = (512, 512)


def get_filename(image_id, image_type):
    check_dir = False
    if "Train" == image_type:
        data_path = "../ShipDetection/5percent"
    elif "Test" == image_type:
        data_path = "../ShipDetection/TestFull"
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
                  * math.pow(math.e, - (1.0/2.0) * ((pixel - miu) ** 2) / sigma)


def p_xt(pixel, miu_list, sigma_list, pg_list, cluster_number):
    sum_p = 0.0
    for i in range(cluster_number):
        sum_p = sum_p + gauss_pdf(pixel, miu_list[i], sigma_list[i]) * pg_list[i]
    return sum_p


max_step = 10
clusters = 2
eps = 0.00001
log_like = 0
miu = [0.0] * clusters
for i in range(clusters):
    miu[i] = random.uniform(0.1, 1.0)
sigma = [0.0] * clusters
for i in range(clusters):
    sigma[i] = random.uniform(0.1, 1.0)
pgk = [0.0] * clusters
pgk[0] = 0.9
pgk[1] = 0.1

# one step
with open('basemodel.csv', 'r') as f:
    lines = f.readlines()

header = lines.pop(0)

fig = plt.figure(1, figsize=(15, 15))

im_no = 0

imageees = ['6e547e3cb.jpg', '0d648f99c.jpg', '1ad12be83.jpg', '1e9238eb9.jpg', '5e28f5ddb.jpg']

#for line in lines:
#    parts = line.split(',')

pdf = PdfPages('foo.pdf')
for image_id in imageees:
    #image_id = parts[0]
    #image_id = '0d648f99c.jpg'
    #image_id = '1ad12be83.jpg'
    #image_id = '1e9238eb9.jpg'
    #image_id = '5e28f5ddb.jpg'

    print(image_id + " " + str(im_no))
    im_no = im_no + 1;
    #if random.random() > 0.01:
    #    continue

    img = cv2.imread(get_filename(image_id, 'Train'), cv2.IMREAD_GRAYSCALE)
    img = gaussian(img, 0.5)
    img = cv2.resize(img, dsize=tile_size)

    step = 1
    while step < max_step:
        sum_pgkxt = [0.0] * clusters
        miu_est = [0.0] * clusters
        sigma_est = [0.0] * clusters
        total = 0

        for pixel_line in img:
            for pixel in pixel_line:
                for i in range(clusters):
                    val = pgk_xt(pixel, miu, sigma, pgk, i, clusters)
                    sum_pgkxt[i] = sum_pgkxt[i] + val
                    miu_est[i] = miu_est[i] + val * pixel
                    sigma_est[i] = sigma_est[i] + val * ((pixel - miu[i]) ** 2)
                total = total + 1

        for i in range(clusters):
            miu_est[i] = miu_est[i] / sum_pgkxt[i]
            sigma_est[i] = sigma_est[i] / sum_pgkxt[i]
            sum_pgkxt[i] = sum_pgkxt[i] / total

        miu = miu_est
        sigma = sigma_est
        pgk = sum_pgkxt

        logll = 0.0

        # for pixel_line in img:
        #     for pixel in pixel_line:
        #         logll = logll + math.log10(p_xt(pixel, miu, sigma, pgk, clusters))

        # if math.fabs(logll - log_like) < eps:
        #     break
        # else:
        #     log_like = logll

        step = step + 1
        print(step)
        print(miu)
        print(sigma)
        print(pgk)

    copy_img = img.copy()
    height, width = copy_img.shape
    for i in range(height):
        for j in range(width):
            if pgk_xt(copy_img[i, j], miu, sigma, pgk, 0, clusters) > pgk_xt(copy_img[i, j], miu, sigma, pgk, 1, clusters):
                copy_img[i, j] = 1
            else:
                copy_img[i, j] = 0

    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(img)
    ax.set_title('Original')

    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(copy_img)
    ax.set_title('EM')

    pdf.savefig(fig)

    #plt.show()
    #break

pdf.close()



    # for pixel_line in img:
    #     for pixel in pixel_line:
    #         for i in range(clusters):
    #             val = pgk_xt(pixel, miu, sigma, pgk, i, clusters)
    #             sigma_est[i] = sigma_est[i] + val * (pixel - miu)



# _train_ids = list(train_ids)
# np.random.seed(time.gmtime())
# fig = plt.figure(1, figsize=(15, 15))
# x = 2
# y = 4
# for i in range(x):
#     image_id = _train_ids[np.random.randint(0, len(_train_ids))]
#
#     ax = fig.add_subplot(x, y, i * y + 1)
#     img = get_image_data(image_id, 'Train')
#     img = cv2.resize(img, dsize=tile_size)
#     ax.imshow(img)
#     ax.set_title('Original')
#
#     ax = fig.add_subplot(x, y, i * y + 2)
#     img = get_image_data(image_id, 'Train')
#     im = gaussian(img, 0.5)
#     img = cv2.resize(img, dsize=tile_size)
#     ax.imshow(img)
#     ax.set_title('Smoothed Image')
#
#     ax = fig.add_subplot(x, y, i * y + 3)
#     img = cv2.imread(get_filename(image_id, 'Train'), cv2.IMREAD_GRAYSCALE)
#     ax.imshow(img, 'gray')
#     ax.set_title('Gray')
#
#     img = cv2.imread(get_filename(image_id, 'Train'), cv2.IMREAD_GRAYSCALE)
#     ax = fig.add_subplot(x, y, i * y + 4)
#     ax.set_title('Hist')
#     ax.hist(img.ravel(), 256, [0, 256])
#
# plt.show()
# plt.suptitle('Smoothed Images')