import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-tamper', '--image', type=str, help='input image')
parser.add_argument('-npz', '--output', type=str, help='output file')
parser.add_argument('-gt', '--mask', type=str, default='', help='ground truth mask (optional)')
args = parser.parse_args()

image_path  = args.image
output_path = args.output
mask_path   = args.mask

result = np.load(output_path)

cols = 3
if mask_path != '':
    cols += 1
    mask = Image.open(mask_path)
else:
    mask = None

if 'np++' in result:
    cols += 1
    noisepr = result['np++']
else:
    noisepr = None

fig, axs = plt.subplots(1, cols)

# 这里的分数是该图片为伪造的概率
fig.suptitle('score: %.3f' % result['score'])

for ax in axs:
    ax.axis('off')

index = 0
ax = axs[index]
ax.imshow(Image.open(image_path)), ax.set_title('Image')

if mask is not None:
    index += 1
    ax = axs[index]
    ax.imshow(mask, cmap='gray'), ax.set_title('Ground Truth')
    ax.set_yticks(list()), ax.set_xticks(list()), ax.axis('on')

if noisepr is not None:
    index += 1
    ax = axs[index]
    # for a better visualization of the noiseprint++, we remove the border and do a down-sampling (useful if the image is too big)
    ax.imshow(noisepr[16:-16:5, 16:-16:5], cmap='gray'), ax.set_title('Noiseprint++')

index += 1
ax = axs[index]
ax.imshow(result['map'], cmap='RdBu_r', clim=[0,1]), ax.set_title('Localization map')

index += 1
ax = axs[index]
ax.imshow(result['conf'], cmap='gray', clim=[0,1]), ax.set_title('Confidence map')
ax.set_yticks(list()), ax.set_xticks(list()), ax.axis('on')

# 调整布局，防止标题重叠
plt.tight_layout(rect=[0, 0, 1, 0.96]) # rect 留出 suptitle 的空间

try:
    # 使用 savefig 保存，dpi=300 提供高分辨率，bbox_inches='tight' 减少白边
    plt.savefig('Sp_D_CNN_A_ani0049_ani0084_0266.jpg', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Visualization saved successfully")
except Exception as e:
    print(f"Error saving file: {e}")
