import pydicom
from scipy import ndimage
from PIL import Image
import os
import numpy as np

images_dir_path = ('../data/DRLung/test/original')
labels_dir_path = ('../data/DRLung/test/mask')

file_names = os.listdir(images_dir_path)
for file in file_names:
    image_path = os.path.join(images_dir_path, file)
    ds_iamge = pydicom.dcmread(image_path)
    image_arr = ds_iamge.pixel_array.astype(np.float32)  # Convert the pixel data to a NumPy array
    image_arr = (image_arr - image_arr.min()) * (255.0 / (image_arr.max() - image_arr.min()))
    [height, width] = image_arr.shape
    scale = [512 * 1.0 / height, 512 * 1.0 / width]
    image_arr = ndimage.interpolation.zoom(image_arr, scale, order=0)
    Image.fromarray(image_arr.astype(np.uint8)).save(f'{image_path[:-4]}.png')

    label_path = os.path.join(labels_dir_path, file)
    ds_label = pydicom.dcmread(label_path)
    label_arr = ds_label.pixel_array.astype(np.float32)
    label_arr[label_arr > 0] = 255.
    [height, width] = label_arr.shape
    scale = [512 * 1.0 / height, 512 * 1.0 / width]
    label_arr = ndimage.interpolation.zoom(label_arr, scale, order=0)
    Image.fromarray(label_arr.astype(np.uint8)).save(f'{label_path[:-4]}.png')
