from utility import *
from propose_regions import *



def load_final_matrix(image_id, folder = "../Final_Matrix_Result"):
    im_pred = np.zeros((IMAGE_SIZE_GLOBAL, IMAGE_SIZE_GLOBAL), np.uint8)
    lines = [line.rstrip('\n ') for line in open("%s/%s" % (folder, image_id.split(".")[0]))]
    i = 0
    j = 0
    for line in lines:
        vals = line.split(" ")
        j = 0
        for val in vals:
            im_pred[i, j] = 1 - int(val) # todo don't forget here
            j = j + 1
        i = i + 1
    return im_pred


outFile = open('../_10kRun/final.csv', "w")

outFile.write('ImageId,EncodedPixels\n')

lines = [line.rstrip("\n") for line in open('../_10kRun/BeforeEM/First_class.csv', "r")]
lines.pop(0)

asd=0
print(len(lines))

for line in lines:
    imgFile = line.split("|")[0]
    pred = line.split("|")[1].split(",")[0].split(" ")[1]
    ipred = int(pred)
    if ipred == 0:
        outFile.write('%s,\n' % imgFile)
    else:
        mat = load_final_matrix(imgFile).transpose()
        rs = multi_rle_encode(mat)
        wr = False
        for l in rs:
            pr = l.split(" ")
            sm = 0
            for i in range(0, len(pr) - 1, 2):
                sm += int(pr[i + 1])
            if sm >= MIN_AREA:
                outFile.write("%s,%s\n" % (imgFile, l))
                wr = True
        if wr is False:
            outFile.write('%s,\n' % imgFile)
    asd = asd + 1
    print(asd)

outFile.close()
