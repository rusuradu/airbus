
lines = [line.rstrip('\n ') for line in open('input.csv')]
outFile = open('input_one_line.csv', "w")

outFile.write("%s\n" % lines[0])

lines.pop(0)

dc = {}

for line in lines:
    key = line.split(",")[0]
    value = line.split(",")[1]
    if dc.get(key) is None:
        dc[key] = value
    else:
        dc[key] = dc[key] + (" %s" % value)

for line in lines:
    key = line.split(",")[0]
    if dc.get(key) is not None:
        outFile.write("%s,%s\n" %(key, dc[key]))
        del dc[key]
