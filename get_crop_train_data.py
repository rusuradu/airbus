from propose_regions import *
import shutil
import random
import cv2


def has_common_area(x, y, lst):
    area = 0
    for ship in lst:
        vect = ship.split(" ")
        for i in range(0, len(vect) - 1, 2):
            start = int(vect[i])
            end = start + int(vect[i + 1]) - 1
            start = start - 1
            end = end - 1

            xs = start % IMAGE_SIZE_GLOBAL
            ys = start / IMAGE_SIZE_GLOBAL

            xe = end % IMAGE_SIZE_GLOBAL
            ye = end / IMAGE_SIZE_GLOBAL

            if (x <= xs <= x + WINDOW_SIZE - 1 or x <= xe <= x + WINDOW_SIZE - 1) \
                    and (y <= ys <= y + WINDOW_SIZE - 1):
                if x <= xs and xe <= x + WINDOW_SIZE - 1:
                    area = area + (xe - xs + 1)
                elif xs <= x <= xe <= x + WINDOW_SIZE - 1:
                    area = area + (xe - x + 1)
                elif x <= xs <= x + WINDOW_SIZE - 1 <= xe:
                    area = area + (xe - x + WINDOW_SIZE)
    return area >= MIN_AREA


lines = [line.rstrip("\n") for line in open('../_10kRun/Raw/_data.csv', "r")]
lines.pop(0)

dc = {}

for line in lines:
    key = line.split(",")[0]
    value = line.split(",")[1]
    if dc.get(key) is None:
        dc[key] = [value]
    else:
        dc[key] = dc[key] + [value]

labels_file = open('../_10kRun/Propose/labels.csv', "w")
labels_file.write("img,tag\n")

numar = 0

print('start img')
print(len(dc.keys()))
for img_key in dc.keys():
    lst = dc[img_key]
    if len(lst[0]) != 0:
        img = cv2.imread('../_10kRun/Raw/train/%s' % img_key)
        st = {-1}
        for index in range(5):
            rnd = 24#int(random.uniform(0, 25))
            while rnd in st:
                rnd = int(random.uniform(0, 25))
            st.add(rnd)
            x = int(rnd / 5) * STRIDE
            y = (rnd % 5) * STRIDE

            crop_img = img[x:x + WINDOW_SIZE - 1, y:y + WINDOW_SIZE - 1]
            fl_name = img_key.split(".")[0] + ("_%s" % rnd) + ".jpg"
            cv2.imwrite('../_10kRun/Propose/train/%s' % fl_name, crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            if has_common_area(x, y, lst):
                labels_file.write("%s,1\n" % fl_name)
            else:
                labels_file.write("%s,0\n" % fl_name)
    numar = numar + 1
    if numar % 10 == 0:
        print(numar)


labels_file.close()

