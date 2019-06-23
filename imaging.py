from utility import *
import matplotlib.pyplot as plt
from propose_regions import *
from skimage.filters import gaussian
from scipy.stats import norm
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.patches as patches
import scipy

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

df = pd.read_csv('train_ship_segmentations_v2.csv')

matplotlib.rcParams.update({'font.size': 28})

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 15
fig_size[1] = 10
plt.rcParams["figure.figsize"] = fig_size
fig = plt.figure(1, figsize=(30, 15))

# new id 1b5fd69bc

image_id = '737bea3d6.jpg'
locale = 'Test'
img = cv2.imread(get_filename(image_id, locale), cv2.IMREAD_COLOR)
df_pred_rp = pd.read_csv('../_FinalRun/final_result.csv')
df_pred_unet = pd.read_csv('../unet/old_models/unet_final_result.csv')
pred_mask_rp = read_flat_mask(image_id, df_pred_rp)
pred_mask_unet = read_flat_mask(image_id, df_pred_unet)
height, width = pred_mask_rp.shape
for i in range(height):
    for j in range(width):
        pred_mask_rp[i, j] = 1 - pred_mask_rp[i, j]
        pred_mask_unet[i, j] = 1 - pred_mask_unet[i, j]

ax = fig.add_subplot(1, 3, 1)
ax.imshow(img)
ax.set_title('Original')

ax = fig.add_subplot(1, 3, 2)
ax.imshow(pred_mask_rp)
ax.set_title('Region Proposal')

ax = fig.add_subplot(1, 3, 3)
ax.imshow(pred_mask_unet)
ax.set_title('U-Net')
#plt.show()
plt.savefig('016Comparison5.png', bbox_inches='tight', pad_inches=0)
