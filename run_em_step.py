from utility import *
from em import EM
from skimage.filters import gaussian
import math
from thread_r import my_process


def process_image(image_id, thread_number):
    print("%d start" % thread_number)
    img = cv2.imread(get_filename(image_id, 'Test'), cv2.IMREAD_GRAYSCALE)
    img = gaussian(img, 0.5)

    print("%d go EM" % thread_number)
    em = EM(img, thread_number)
    em.run_it()

    print("%d done EM" % thread_number)
    prediction_matrix = create_prediction_image(img, em.miu, em.sigma, em.pgk, em.clusters)

    out_file = open("../EM_Result/%s" % image_id.split(".")[0], "w")

    for line in prediction_matrix:
        for el in line:
            out_file.write("%d " % el)
        out_file.write("\n")
    out_file.close()

    thread_file = open("../EM_Result/_Thread_%d" % thread_number, "a")
    thread_file.write(image_id)
    thread_file.write("\n")
    thread_file.close()


MAX_THREADS = 4

df = pd.read_csv('gigel.csv')
img_ids = df.ImageId.values

thread_no = 0
values_no = math.floor(len(img_ids) / MAX_THREADS)
if __name__ == '__main__':
    while thread_no < MAX_THREADS:
        if thread_no != MAX_THREADS:
            thread_imgs = img_ids[(thread_no * values_no):((thread_no + 1) * values_no)]
        else:
            thread_imgs = img_ids[(thread_no * values_no):len(img_ids)]
        th = my_process(process_image, thread_imgs, thread_no)
        th.start()
        thread_no = thread_no + 1


