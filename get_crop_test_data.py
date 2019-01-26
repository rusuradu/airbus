from propose_regions import *

lines = [line.rstrip("\n") for line in open('../_10kRun/BeforeEM/First_class.csv', "r")]\

lines.pop(0)

outFile = open('../_10kRun/Propose/_test_data.csv', "w")

ii = 0

print(len(lines))

for line in lines:
    imgFile = line.split("|")[0]
    pred = line.split("|")[1].split(",")[0].split(" ")[1]
    ipred = int(pred)
    if ipred == 1:
        img_id = imgFile.split(".")[0]
        matrix = load_prediction_matrix(img_id)
        props = get_proposals(matrix)
        for (x, y) in props:
            rnd = int(x / 136) * 5 + int(y / 163)
            img = cv2.imread('../_10kRun/BeforeEM/%s' % imgFile)
            crop_img = img[x:x + WINDOW_SIZE - 1, y:y + WINDOW_SIZE - 1]
            fl_name = imgFile.split(".")[0] + ("_%s" % rnd) + ".jpg"
            cv2.imwrite('../_10kRun/Propose/test/%s' % fl_name, crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            outFile.write("%s\n" % fl_name)
    ii = ii + 1
    print(ii)

outFile.close()
