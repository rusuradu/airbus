from utility import *

lines = [line.rstrip("\n") for line in open('../_FinalRun/First_class_full11111.csv', "r")]
lines.pop(0)

asd = 0
print(len(lines))
locale = 'Test'

for line in lines:
    imgFile = line.split("|")[0]
    pred = line.split("|")[1].split(",")[0].split(" ")[1]
    ipred = int(pred)

    # if imgFile != '3e63ffa7f.jpg':
    #     continue
    if ipred == 1:
        img_orig = cv2.imread(get_filename(imgFile, locale), cv2.IMREAD_COLOR)

        img_pred = img_orig.copy()
        lines = [line.rstrip('\n ') for line in open("../Final_EM_Result/%s" % imgFile.split(".")[0])]
        i = 0
        j = 0
        for line in lines:
            vals = line.split(" ")
            j = 0
            for val in vals:
                if int(val) == 0:
                    img_pred[i, j] = img_orig[i, j]
                else:
                    img_pred[i, j] = [0, 0, 0]
                j = j + 1
            i = i + 1
        cv2.imwrite('../ExcludePixels/%s' % imgFile, img_pred, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    asd = asd + 1
    print(asd)
