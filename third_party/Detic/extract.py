import os
import imghdr
from PIL import Image
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

train_path = '/home/ubuntu/efs/imagenet/ImageNet-21K/images/train'
img_folder_idx = [f.replace('.tar', '') for f in os.listdir(train_path) if f.endswith('.tar')]  # len=19167
print('len(img_folder_idx) = ', len(img_folder_idx))

img_paths = []
for i, folder_idx in enumerate(img_folder_idx):
    img_folder_path = os.path.join(train_path, folder_idx)
    if os.path.exists(img_folder_path):
        img_names = [f for f in os.listdir(img_folder_path)]
        current_img_paths = [os.path.join(img_folder_path, img_name) for img_name in img_names]
        img_paths.extend(current_img_paths)

print('len(img_paths) = ', len(img_paths))


def check(img_path):
    # if imghdr.what(img_path) is None:
    try:
        Image.open(img_path)
    except IOError:
        print(i, 'remove', img_path)
        os.remove(img_path)

with ThreadPoolExecutor(128) as executor:
    res = executor.map(check, img_paths)


