import os
import shutil
import random

with open('input.csv') as f:
    lines = f.readlines()

outFile = open("../ShipDetection/1percent/_data.csv", "w")
labelFile = open("../ShipDetection/1percent/labels.csv", "w")

header = lines.pop(0)
outFile.write(header)
labelFile.write("image,tag\n")

index = 0

print(len(lines))

for line in lines:
    x = random.random()
    index = index + 1
    if x < 0.01:
        print(index)
        outFile.write(line)
        imgFile = line.split(",")[0]
        if len(line.split(",")[1].rstrip("\n")) == 0:
            labelFile.write("%s,0\n" % imgFile)
        else:
            labelFile.write("%s,1\n" % imgFile)
        shutil.copy("../ShipDetection/TrainFull/%s" % imgFile, "../ShipDetection/1percent/train/%s" % imgFile)

outFile.close()
labelFile.close()
