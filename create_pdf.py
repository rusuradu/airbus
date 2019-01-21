from utility import *
from em import EM
from skimage.filters import gaussian
import math
from thread_r import my_process
from pylab import imshow, show
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


df = pd.read_csv('gigel.csv')
img_ids = df.ImageId.values
locales = ['Test', 'Test', 'Test', 'Test', 'Test', 'Train', 'Train', 'Train', 'Train', 'Train']


pdf = PdfPages('foo222.pdf')

for page in range(10):
    image_id = img_ids[page]
    locale = locales[page]
    img_orig = cv2.imread(get_filename(image_id, locale), cv2.IMREAD_COLOR)
    img = cv2.imread(get_filename(image_id, locale), cv2.IMREAD_GRAYSCALE)
    img = gaussian(img, 0.5)

    img_pred = img.copy()
    lines = [line.rstrip('\n ') for line in open("../EM_Result/%s" % image_id.split(".")[0])]
    i = 0
    j = 0
    for line in lines:
        vals = line.split(" ")
        j = 0
        for val in vals:
            img_pred[i, j] = int(val)
            j = j + 1
        i = i + 1
    fig = plt.figure(1, figsize=(20, 15))

    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(img_orig)
    ax.set_title('Original')

    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(img)
    ax.set_title('GrayScale + Gaussian Filter')

    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(img_pred)
    ax.set_title('Prediction')

    pdf.savefig(fig)

pdf.close()

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