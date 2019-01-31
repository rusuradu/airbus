from utility import *
from PIL import *

thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

df_test = pd.read_csv('../_10kRun/Raw/_test_Data.csv')
df_pred = pd.read_csv('../_10kRun/Final.csv')


def read_masks(img_name, df):
    mask_list = df.loc[df['ImageId'] == img_name, 'EncodedPixels'].tolist()
    all_masks = np.zeros((len(mask_list), 768, 768))
    for idx, mask in enumerate(mask_list):
        if isinstance(mask, str):
            all_masks[idx] = rle_decode(mask)
    return all_masks


def read_flat_mask(img_name, df):
    all_masks = read_masks(img_name, df)
    return np.sum(all_masks, axis=0)


def iou(img_true, img_pred):
    i = np.sum((img_true * img_pred) > 0)
    u = np.sum((img_true + img_pred) > 0) + 0.0000000000000000001  # avoid division by zero
    return i/u


def f2(masks_true, masks_pred):
    # a correct prediction on no ships in image would have F2 of zero (according to formula),
    # but should be rewarded as 1
    if np.sum(masks_true) == np.sum(masks_pred) == 0:
        return 1.0

    f2_total = 0
    ious = {}
    for t in thresholds:
        tp, fp, fn = 0, 0, 0
        for i, mt in enumerate(masks_true):
            found_match = False
            for j, mp in enumerate(masks_pred):
                key = 100 * i + j
                if key in ious.keys():
                    miou = ious[key]
                else:
                    miou = iou(mt, mp)
                    ious[key] = miou  # save for later
                if miou >= t:
                    found_match = True
            if not found_match:
                fn += 1

        for j, mp in enumerate(masks_pred):
            found_match = False
            for i, mt in enumerate(masks_true):
                miou = ious[100 * i + j]
                if miou >= t:
                    found_match = True
                    break
            if found_match:
                tp += 1
            else:
                fp += 1
        f2 = (5 * tp) / (5 * tp + 4 * fn + fp)
        f2_total += f2

    return f2_total / len(thresholds)


lines = [line.rstrip('\n') for line in open("../_10kRun/final.csv")]
lines.pop(0)

df = {''}

sm = 0.0
imgs = 0

out_file = open("res.txt", "w")

for line in lines:
    img_id = line.split(",")[0]

    # if len(line.split(",")[1]) != 0 and img_id not in df:
    #     true_mask = read_flat_mask(img_id, df_test)
    #     pred_mask = read_flat_mask(img_id, df_pred)
    #     df.add(img_id)
    #     fig = plt.figure(1, figsize=(20, 15))  # create a new figure
    #
    #     ax = fig.add_subplot(1, 3, 1)
    #     img = cv2.imread("../_10kRun/Raw/test/%s" % img_id)
    #     ax.imshow(img)
    #     ax.set_title(img_id)
    #
    #     ax = fig.add_subplot(1, 3, 2)
    #     ax.imshow(true_mask)#, cmap='gray', interpolation='nearest')
    #     ax.set_title('True mask')
    #
    #     ax = fig.add_subplot(1, 3, 3)
    #     ax.imshow(pred_mask)#, cmap='gray', interpolation='nearest')
    #     ax.set_title('Pred mask')
    #
    #     plt.show()  # show the figure, non-blocking
    #     input()
    #     plt.close()  # close the figure to show the next one.

    if img_id not in df:
        if img_id == '2e3ce58af.jpg':
            jja = 5
        if img_id == '03eaa8a5e.jpg':
            jja =6
        true_masks = read_masks(img_id, df_test)
        pred_masks = read_masks(img_id, df_pred)
        df.add(img_id)
        t = f2(true_masks, pred_masks)
        sm += t
        out_file.write("%s %f\n" % (img_id, t))
    imgs = imgs + 1
    print(imgs)

out_file.close()

print(sm / imgs)


# 0.6910569105691057
# 0.7010657534795623 pred
# 0.7157691884708288
