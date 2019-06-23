from utility import *
import matplotlib.pyplot as plt
from propose_regions import *
from skimage.filters import gaussian
from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.patches as patches
import scipy

df = pd.read_csv('train_ship_segmentations_v2.csv')

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 15
fig_size[1] = 10
plt.rcParams["figure.figsize"] = fig_size


# print(df.head(10))

# fig = plt.figure(1, figsize=(15, 15))

# ax = fig.add_subplot(1, 1, 1)

# df['ship_count'] = df.groupby('ImageId')['ImageId'].transform('count')
# df.loc[df['EncodedPixels'].isnull().values, 'ship_count'] = 0  #see infocusp's comment
#
#
# srs = df.groupby('ship_count')['ship_count'].count()
#
# df1 = pd.DataFrame({'ships': srs.index, 'count': srs.values})
#
#
# n, bins, patches = plt.hist(x = df1.ships[1:], weights=df1['count'][1:], bins=16, histtype='bar')
# plt.grid(axis='x')
# plt.xlabel('Number of ship in image')
# plt.title('Ship count')
#
#

# locale = 'Train'
#
# img1 = cv2.imread(get_filename('0a50e22e5.jpg', locale), cv2.IMREAD_COLOR)
# img2 = cv2.imread(get_filename('0a90e302a.jpg', locale), cv2.IMREAD_COLOR)
# img3 = cv2.imread(get_filename('0a99243c0.jpg', locale), cv2.IMREAD_COLOR)
# # img4 = cv2.imread(get_filename('00d89dfdc.jpg', locale), cv2.IMREAD_COLOR)
# # img5 = cv2.imread(get_filename('0a3b48a9c.jpg', locale), cv2.IMREAD_COLOR)
# # img6 = cv2.imread(get_filename('0a78f9786.jpg', locale), cv2.IMREAD_COLOR)
# fig = plt.figure(1, figsize=(20, 15))
#
#
# ax = fig.add_subplot(1, 3, 1)
# ax.imshow(img1)
#
# ax = fig.add_subplot(1, 3, 2)
# ax.imshow(img2)
#
# ax = fig.add_subplot(1, 3, 3)
# ax.imshow(img3)
#
# plt.savefig('002ShipExampleatSea.png' ,bbox_inches = 'tight',    pad_inches = 0)


# img = cv2.imread(get_filename('0c704bba1.jpg', "Train"), cv2.IMREAD_GRAYSCALE)
# img = gaussian(img, 0.5)
#
# n, bins, patches  = plt.hist(img.flatten(), bins=200)
#
# x = np.linspace(0,1)
# mu = 0.13
# sigma = 0.03
# y = mlab.normpdf(bins, mu, sigma) * 6000
# plt.plot(bins, y)
# currentAxis = plt.gca()
# currentAxis.add_patch(Rectangle((0.3, 200), width=0.55, height=2000, alpha=1, fill=None, facecolor='none'))
# plt.axis([0.3, 0.85, 0, 2000])
# plt.show()

# new id 1b5fd69bc
# plt.savefig('gigel.png' ,bbox_inches = 'tight',    pad_inches = 0)

# 00dc34840.jpg, 0b7359c38.jpg, 1a36da3c8.jpg
# image_id = '2b98ba225.jpg'
# locale = 'Test'
# img_orig = cv2.imread(get_filename(image_id, locale), cv2.IMREAD_COLOR)
# img = cv2.imread(get_filename(image_id, locale), cv2.IMREAD_GRAYSCALE)
# img = gaussian(img, 0.5)

# img_pred = img_orig.copy()
# lines = [line.rstrip('\n ') for line in open("../Final_EM_Result/%s" % image_id.split(".")[0])]
# i = 0
# j = 0
# for line in lines:
#     vals = line.split(" ")
#     j = 0
#     for val in vals:
#         if int(val) == 0:
#             img_pred[i, j] = img_orig[i, j]
#         else:
#             img_pred[i, j] = [0, 0, 0]
#         j = j + 1
#     i = i + 1
# fig = plt.figure(1, figsize=(20, 15))
#
# ax = fig.add_subplot(1, 2, 1)
# ax.imshow(img_orig)
# ax.set_title('Original')
#
# # ax = fig.add_subplot(1, 3, 2)
# # ax.imshow(img)
# # ax.set_title('GrayScale + Gaussian Filter')
#
# ax = fig.add_subplot(1, 2, 2)
# ax.imshow(img_pred)
# ax.set_title('EM Prediction')
#
# cv2.imwrite('../ExcludePixels/%s' % image_id, img_pred, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

# image_id = '00dc34840.jpg'
# img_id = image_id
# locale = 'Test'
#

# image_id = '1b5fd69bc.jpg'
# locale = 'Test'
# img = cv2.imread(get_filename(image_id, locale), cv2.IMREAD_COLOR)
# matrix = load_prediction_matrix(image_id)
# propose = get_proposals(matrix)
# fig = plt.figure(1, figsize=(30, 15))

# ax = fig.add_subplot(1, 2, 1)
# ax.imshow(img)

# ax = fig.add_subplot(1, 1, 1)
# ax.imshow(matrix)
# ax.set_title('Denoising')
# plt.show()

# print(len(propose))
#
# ind = 1
# fig.suptitle('Proposals')
# for (x, y) in propose:
#     ax = fig.add_subplot(3, 3, ind)
#     crop_img = img[x:x + WINDOW_SIZE, y:y + WINDOW_SIZE]
#
#     ax.imshow(crop_img)
#     ind = ind + 1
# #plt.show()

# here is the good part


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


############## PREDICTION MASKS
# df_pred = pd.read_csv('../unet/unet_final_result.csv')
# df_pred_init = pd.read_csv('../_FinalRun/final_result.csv')
#
# img_id = "161dfbbdc.jpg"
# pred_mask = read_flat_mask(img_id, df_pred)
# pred_mask_init = read_flat_mask(img_id, df_pred_init)
# df.add(img_id)
# fig = plt.figure(1, figsize=(20, 15))  # create a new figure
#
# ax = fig.add_subplot(1, 3, 1)
# img = cv2.imread("../ShipDetection/TestFull/%s" % img_id)
# ax.imshow(img)
# ax.set_title('Original')
#
# print(img.shape)
# height, width = pred_mask.shape
# print(width)
# print(height)
# for i in range(height):
#     for j in range(width):
#         pred_mask[i, j] = 1 - pred_mask[i, j]
#         pred_mask_init[i, j] = 1 - pred_mask_init[i, j]
#
# ax = fig.add_subplot(1, 3, 2)
# ax.imshow(pred_mask)
# ax.set_title('Predicted mask')
#
# ax = fig.add_subplot(1, 3, 3)
# ax.imshow(pred_mask_init)
# ax.set_title('Init Predicted mask')

# img_id = "161dfbbdc.jpg"
# img = cv2.imread("../ShipDetection/TestFull/%s" % img_id)
#
# fig, ax = plt.subplots(1)
#
# # Display the image
# ax.imshow(img)
#
# # Create a Rectangle patch
# rect = patches.Rectangle((50, 100), 40, 30, linewidth=1, edgecolor='r', facecolor='none')
#
# WINDOW_SIZE = 224
# STRIDE = 136
# idx = 1
# idy = 1
# for x in range(0, 545, STRIDE):
#     for y in range(0, 545, STRIDE):
#         if idx % 2 == 1 and idy % 2 == 1:
#             rect = patches.Rectangle((x, y), WINDOW_SIZE, WINDOW_SIZE, linewidth=1, edgecolor='r', facecolor='none')
#             ax.add_patch(rect)
#         elif idy % 2 == 0:
#             rect = patches.Rectangle((x, y), WINDOW_SIZE, WINDOW_SIZE, linewidth=1, edgecolor='g', facecolor='none')
#             ax.add_patch(rect)
#         idy = idy + 1
#     idx = idx + 1
#
# plt.show()
