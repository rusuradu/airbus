import shutil

lines = [line.rstrip("\n") for line in open("../_10kRun/BeforeEM/First_class.csv")]

lines.pop(0)

for line in lines:
    imgFile = line.split("|")[0]
    pred = line.split("|")[1].split(",")[0].split(" ")[1]
    iPred = int(pred)
    if iPred == 1:
        shutil.copy("../_10kRun/Raw/test/%s" % imgFile, "../_10kRun/BeforeEM/%s" % imgFile)
