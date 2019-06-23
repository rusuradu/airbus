from utility import *
from metric import *
import matplotlib.pyplot as plt
from propose_regions import *
from skimage.filters import gaussian
from scipy.stats import norm
import matplotlib.mlab as mlab
from matplotlib.patches import Rectangle
import scipy

df = pd.read_csv('train_ship_segmentations_v2.csv')

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 15
fig_size[1] = 10
plt.rcParams["figure.figsize"] = fig_size


train_ids_img = [
    '000d42241.jpg',
    '00113a75c.jpg',
    '001234638.jpg',
    '001566f7c.jpg',
    '0019fc4d8.jpg',
    '001aee007.jpg',
    '0038cbe45.jpg',
    '0041d7084.jpg',
    '0060f6de0.jpg',
    '008127d89.jpg',
    '008ceaeb0.jpg',
    '009c1116c.jpg',
    '00a52cd2a.jpg',
    '00b3e6991.jpg',
    '00c498ef4.jpg',
    '00cad4541.jpg',
    '00ce2c1c0.jpg',
    '00e603959.jpg',
    '00ea572a2.jpg',
    '00f34434e.jpg'
]

test_ids_img = [
    '00a3ab3cc.jpg',
    '00e90efc3.jpg',
    '0a6dca616.jpg',
    '0d2fa42e5.jpg'
]


for idx, image_id in enumerate(train_ids_img):
    # image_id = '00abc623a.jpg'
    locale = 'Train'
    img_orig = cv2.imread(get_filename(image_id, locale), cv2.IMREAD_COLOR)

    x = read_flat_mask(image_id, df)

    cv2.imwrite('../UNetPrepare/train/image/%d.jpg' % idx, cv2.resize(img_orig, (256, 256)), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    cv2.imwrite('../UNetPrepare/train/label/%d.tiff' % idx, cv2.resize(abs(x + (-1)) * 255, (256, 256)))

for idx, image_id in enumerate(test_ids_img):
    # image_id = '00abc623a.jpg'
    locale = 'Test'
    img_orig = cv2.imread(get_filename(image_id, locale), cv2.IMREAD_COLOR)

    cv2.imwrite('../UNetPrepare/test/%d.jpg' % idx, cv2.resize(img_orig, (256, 256)),
                [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    # fig = plt.figure(1, figsize=(20, 15))
    #
    # ax = fig.add_subplot(1, 2, 1)
    # ax.imshow(img_orig)
    # ax.set_title('Original')
    #
    # ax = fig.add_subplot(1, 2, 2)
    # ax.imshow(x)
    # ax.set_title('mask')
    #
    # plt.show()


