from utility import *

# 768 * 768

out_file = open("ShipNumber.txt", "w")

lines = [line.rstrip('\n ') for line in open("train_ship_segmentations_v2.csv")]
lines.pop(0)

df = pd.read_csv('train_ship_segmentations_v2.csv')
df = df.reset_index()
df['ship_count'] = df.groupby('ImageId')['ImageId'].transform('count')
df.loc[df['EncodedPixels'].isnull().values, 'ship_count'] = 0  #see infocusp's comment

print(df['ship_count'].describe())

sum_area = 0.0
ship_no = 0
line_o = 0
for line in lines:
    image_id = line.split(",")[0]
    pixel_line = line.split(",")[1]
    line_o = line_o + 1
    print(line_o)
    sm = 0.0
        # out_file.write("%s %d" % (image_id, 0))
    if len(pixel_line) != 0:
        ship_no = ship_no + 1
        pixels = pixel_line.split(" ")
        for i in range(1, len(pixels), 2):
            sum_area = sum_area + int(pixels[i])
            sm = sm + int(pixels[i])
        out_file.write("%d\n" % sm)

avg_area = sum_area / ship_no

print(avg_area)