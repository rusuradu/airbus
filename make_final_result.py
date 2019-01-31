from utility import *
from propose_regions import *

lines = [line.rstrip("\n") for line in open("../_FinalRun/SecondResNet.csv", "r")]
lines.pop(0)
dc = {}

for line in lines:
    first = line.split("|")[0]
    iPred = int(line.split("|")[1].split(",")[0].split(" ")[1])
    img_id = first.split("_")[0]
    img_part = int(first.split("_")[1].split(".")[0])
    if dc.get(img_id) is None:
        dc[img_id] = [(img_part, iPred)]
    else:
        dc[img_id] = dc[img_id] + [(img_part, iPred)]

lines = [line.rstrip("\n") for line in open('../_FinalRun/First_class_full.csv', "r")]
lines.pop(0)

asd = 0
print(len(lines))

for line in lines:
    imgFile = line.split("|")[0]
    pred = line.split("|")[1].split(",")[0].split(" ")[1]
    ipred = int(pred)

    # if imgFile != '3e63ffa7f.jpg':
    #     continue
    if ipred == 1:
        final_mat = np.ones((IMAGE_SIZE_GLOBAL, IMAGE_SIZE_GLOBAL), np.uint8)
        pred_mat = load_prediction_matrix(imgFile)

        if dc.get(imgFile.split(".")[0]) is None:
            out_file = open("../FULLRUN_Matrix/%s" % imgFile.split(".")[0], "w")
            for l in pred_mat:
                for el in l:
                    out_file.write("%d " % 1)
                out_file.write("\n")
            out_file.close()
            continue

        for (buc, prd_radu) in dc[imgFile.split(".")[0]]:
            if prd_radu == 1:
                x = int(buc / 5) * 136
                y = (buc % 5) * 136
                for i in range(x, x + WINDOW_SIZE):
                    for j in range(y, y + WINDOW_SIZE):
                        final_mat[i, j] = int(pred_mat[i, j])
        out_file = open("../FULLRUN_Matrix/%s" % imgFile.split(".")[0], "w")

        for l in final_mat:
            for el in l:
                out_file.write("%d " % el)
            out_file.write("\n")
        out_file.close()
    asd = asd + 1
    print(asd)
