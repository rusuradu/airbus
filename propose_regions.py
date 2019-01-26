#crop_img = img[y:y+h, x:x+w]
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from utility import *

IMAGE_SIZE_GLOBAL = 768

MIN_AREA = 35
WINDOW_SIZE = 224
STRIDE = 136


def load_prediction_matrix(image_id, folder = "../EM_Result1"):
    im_pred = np.zeros((IMAGE_SIZE_GLOBAL, IMAGE_SIZE_GLOBAL), np.uint8)
    lines = [line.rstrip('\n ') for line in open("%s/%s" % (folder, image_id.split(".")[0]))]
    i = 0
    j = 0
    for line in lines:
        vals = line.split(" ")
        j = 0
        for val in vals:
            im_pred[i, j] = int(val)
            j = j + 1
        i = i + 1
    cv2.fastNlMeansDenoising(im_pred, im_pred)
    return im_pred


def compute_sum(em_prediction):

    def sw(vl):
        return int(1) - int(vl)

    em_sum = []
    for i in range(len(em_prediction)):
        ln = []
        for j in range(len(em_prediction[i])):
            if i == 0 and j == 0:
                ln.append(sw(em_prediction[i][j]))
            elif i == 0:
                ln.append(ln[j - 1] + sw(em_prediction[i][j]))
            elif j == 0:
                ln.append(em_sum[i - 1][j] + sw(em_prediction[i][j]))
            else:
                ln.append(em_sum[i - 1][j] + ln[j - 1] - em_sum[i - 1][j - 1] + sw(em_prediction[i][j]))
        em_sum.append(ln)
    return em_sum


def get_proposals(em_prediction):
    res = []
    em_sum = compute_sum(em_prediction)
    for x in range(0, 545, STRIDE):
        for y in range(0, 545, STRIDE):
            x1 = x + WINDOW_SIZE - 1
            y1 = y + WINDOW_SIZE - 1
            if x == 0 and y == 0:
                sm = em_sum[x1][y1]
            elif x == 0:
                sm = em_sum[x1][y1] - em_sum[x1][y - 1]
            elif y == 0:
                sm = em_sum[x1][y1] - em_sum[x - 1][y1]
            else:
                sm = em_sum[x1][y1] - em_sum[x - 1][y1] - em_sum[x1][y - 1] + em_sum[x - 1][y - 1]
            if sm >= MIN_AREA:
                res.append((x, y))
    return res


image_id = '0b8cde107.jpg'
locale = 'Test'

if __name__ == '__main__':
    img = cv2.imread(get_filename(image_id, locale), cv2.IMREAD_GRAYSCALE)
    start = timer()
    matrix = load_prediction_matrix(image_id)
    propose = get_proposals(matrix)
    print(timer() - start)
    print(len(propose))
    fig = plt.figure(1, figsize=(30, 50))

    ax = fig.add_subplot(1, len(propose) + 2, 1)
    ax.imshow(img)

    ax = fig.add_subplot(1,  len(propose) + 2, 2)
    ax.imshow(matrix)

    ind = 1
    for (x, y) in propose:
        ax = fig.add_subplot(1, len(propose) + 2, ind + 2)
        crop_img = img[x:x + WINDOW_SIZE, y:y + WINDOW_SIZE]

        ax.imshow(crop_img)
        ind = ind + 1
    plt.show()
