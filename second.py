import os
import shutil
import random

lines = [line.rstrip('\n ') for line in open('input.csv')]

trainFile = open("../_10kRun/Raw/_data.csv", "w")
testFile = open("../_10kRun/Raw/_test_Data.csv", "w")
labelFile = open("../_10kRun/Raw/labels.csv", "w")

header = lines.pop(0)
trainFile.write(header)
testFile.write(header)
labelFile.write("image,tag\n")

index = 0

print(len(lines))

dc = {}

for line in lines:
    key = line.split(",")[0]
    value = line.split(",")[1]
    if dc.get(key) is None:
        dc[key] = [value]
    else:
        dc[key] = dc[key] + [value]


for line in lines:
    x = random.random()
    index = index + 1
    if x < 0.05:
        print("%d" % index)
        rnd = random.random()
        if rnd < 0.95:
            imgFile = line.split(",")[0]
            lst = dc[imgFile]
            if len(lst) == 0:
                trainFile.write("%s,\n" % imgFile)
                labelFile.write("%s,0\n" % imgFile)
            else:
                for val in lst:
                    trainFile.write("%s,%s\n" % (imgFile, val))
                labelFile.write("%s,1\n" % imgFile)
            shutil.copy("../ShipDetection/TrainFull/%s" % imgFile, "../_10kRun/Raw/train/%s" % imgFile)
        else:
            imgFile = line.split(",")[0]
            lst = dc[imgFile]
            if len(lst) == 0:
                testFile.write("%s,\n" % imgFile)
            else:
                for val in lst:
                    testFile.write("%s,%s\n" % (imgFile, val))
            shutil.copy("../ShipDetection/TrainFull/%s" % imgFile, "../_10kRun/Raw/test/%s" % imgFile)

trainFile.close()
testFile.close()
labelFile.close()
