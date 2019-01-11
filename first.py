import os
import time

import random
import matplotlib.pyplot as plt
import cv2
from skimage.filters import gaussian
from matplotlib.backends.backend_pdf import PdfPages

from skimage import color
from skimage import io
from utility import *
from em import EM
from thread_r import my_process

print("gigel")

# one step
with open('basemodel.csv', 'r') as f:
    lines = f.readlines()

header = lines.pop(0)

fig = plt.figure(1, figsize=(15, 15))

im_no = 0

#imageees = ['6e547e3cb.jpg', '0d648f99c.jpg', '1ad12be83.jpg', '1e9238eb9.jpg', '5e28f5ddb.jpg']
imageees = ['9a92ef162.jpg']
#imageees = ['6e547e3cb.jpg', '0d648f99c.jpg']

#for line in lines:
#    parts = line.split(',')

pdf = PdfPages('foo.pdf')
for image_id in imageees:
    #image_id = parts[0]
    print(image_id + " " + str(im_no))
    im_no = im_no + 1

    def f1():
        img = cv2.imread(get_filename('6e547e3cb.jpg', 'Train'), cv2.IMREAD_GRAYSCALE)
        img = gaussian(img, 0.5)
        # img = cv2.resize(img, dsize=tile_size)

        em_object = EM(img)
        em_object.run_it()

    def f2():
        img = cv2.imread(get_filename('0d648f99c.jpg', 'Train'), cv2.IMREAD_GRAYSCALE)
        img = gaussian(img, 0.5)
        # img = cv2.resize(img, dsize=tile_size)

        em_object = EM(img)
        em_object.run_it()

    my_process(f1).start()
    my_process(f2).start()

    # copy_img = create_prediction_image(img, em_object.miu, em_object.sigma, em_object.pgk, em_object.clusters)
    #
    # ax = fig.add_subplot(1, 2, 1)
    # ax.imshow(img)
    # ax.set_title('Original')
    #
    # ax = fig.add_subplot(1, 2, 2)
    # ax.imshow(copy_img)
    # ax.set_title('EM')
    #
    # pdf.savefig(fig)

pdf.close()


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