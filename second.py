import os
import shutil
import random

with open('input.csv') as f:
    lines = f.readlines()

outFile = open("../ShipDetection/5percent/_data.csv", "w")

header = lines.pop(0)
outFile.write(header)

index = 0

print(len(lines))

for line in lines:
    x = random.random()
    index = index + 1
    if x < 0.01:
        print(index)
        outFile.write(line)
        imgFile = line.split(",")[0]
        shutil.copy("../ShipDetection/TrainFull/%s" % imgFile, "../ShipDetection/5percent/%s" % imgFile)

outFile.close()
