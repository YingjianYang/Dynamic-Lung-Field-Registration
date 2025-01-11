from torch.utils.data import Dataset

from acregnet.utils.io import read_txt, read_image
from acregnet.utils.tensor import to_tensor, to_one_hot, swap_labels, relabel
from acregnet.utils.misc import get_pairs, get_image_info
import cv2
import numpy as np
import random


class ImagePairsDatasetTrain_V2(Dataset):
    """
    不使用 ImagePairsDataset_V1 两两配对的方式，而是使用对mov图像增强后再做为fix
    """

    def __init__(self, images_file_path, labels_file_path, mode='train'):
        assert mode in ['train', 'test']
        self.mode = mode

        images_files = read_txt(images_file_path)
        labels_files = read_txt(labels_file_path)

        self.images_files = {
            'mov': images_files,
            'fix': images_files
        }

        self.labels_files = {
            'mov': labels_files,
            'fix': labels_files
        }

        self.input_size, self.num_labels = get_image_info(labels_files[0], is_label=True)

    def __len__(self):
        return len(self.images_files['mov'])

    def _tranform(self, data, label):
        data = data.squeeze()  # 去除channel维
        label = label.squeeze()  # 去除channel维

        row, col = data.shape

        # 翻转
        # if random.randint(0, 1):
        #     angle1 = random.uniform(-10, 10)
        #     M1 = cv2.getRotationMatrix2D((row / 2.0, col / 2.0), angle1, 1)
        #     data = cv2.warpAffine(data, M1, (col, row))
        #     label = cv2.warpAffine(label, M1, (col, row))

        # 放缩
        # if random.randint(0, 1):
        #     scale = random.uniform(0.8, 0.99)  # [0.8, 1)
        #     hei, wid = int(row * scale), int(col * scale)
        #     x = random.randint(0, col - wid)
        #     y = random.randint(0, row - hei)
        #     cropped = data[y:y + hei, x:x + wid]
        #     cropped_l = label[y:y + hei, x:x + wid]
        #     data = cv2.resize(cropped, (row, col))
        #     label = cv2.resize(cropped_l, (row, col))

        # 仿射变换
        if random.randint(0, 1):
            P1 = np.float32([[0, 0], [col - 1, 0], [0, row - 1]])
            P2 = np.float32(
                [[0, row * random.uniform(0, 0.1)], [col * random.uniform(0.9, 1), row * random.uniform(0, 0.1)],
                 [col * random.uniform(0, 0.1), row * random.uniform(0.9, 1)]])
            M2 = cv2.getAffineTransform(P1, P2)
            data = cv2.warpAffine(data, M2, (col, row))
            label = cv2.warpAffine(label, M2, (col, row))

        # 高斯滤波
        # if random.randint(0, 1):
        #     data = cv2.GaussianBlur(data, (5, 5), 0)

        # 直方图均衡化
        # if random.randint(0, 1):
        #     data = np.uint8(data * 255)
        #     data = cv2.equalizeHist(data)
        #     data = data.astype(np.float32) / 255.

        data = np.expand_dims(data, axis=-1)  # 恢复channel维
        label = np.expand_dims(label, axis=-1)  # 恢复channel维
        return data, label

    def __getitem__(self, index):
        mov_image_path = self.images_files['mov'][index]
        fix_image_path = self.images_files['fix'][index]
        mov_label_path = self.labels_files['mov'][index]
        fix_label_path = self.labels_files['fix'][index]

        mov_image = read_image(mov_image_path) / 255.
        fix_image = read_image(fix_image_path) / 255.
        mov_label = read_image(mov_label_path)
        fix_label = read_image(fix_label_path)

        fix_image, fix_label = self._tranform(fix_image, fix_label)

        mov_label[mov_label > 0] = 1  # 将肺区的标签统一设为1
        fix_label[fix_label > 0] = 1

        mov_image = to_tensor(mov_image)
        fix_image = to_tensor(fix_image)
        mov_label = to_tensor(mov_label)
        fix_label = to_tensor(fix_label)

        mov_label = relabel(mov_label)
        fix_label = relabel(fix_label)

        if self.mode == 'train':
            mov_label = to_one_hot(mov_label)
            fix_label = to_one_hot(fix_label)

        return (mov_image, fix_image), (mov_label, fix_label)


class ImagePairsDatasetTrain_V4(Dataset):
    """
    在 ImagePairsDatasetTrain_V2 的基础上，使用肺实质图像替换原始图像
    """

    def __init__(self, images_file_path, labels_file_path, mode='train'):
        assert mode in ['train', 'test']
        self.mode = mode

        images_files = read_txt(images_file_path)
        labels_files = read_txt(labels_file_path)

        self.images_files = {
            'mov': images_files,
            'fix': images_files
        }

        self.labels_files = {
            'mov': labels_files,
            'fix': labels_files
        }

        self.input_size, self.num_labels = get_image_info(labels_files[0], is_label=True)

    def __len__(self):
        return len(self.images_files['mov'])

    def _tranform(self, data, label):
        data = data.squeeze()  # 去除channel维
        label = label.squeeze()  # 去除channel维

        row, col = data.shape

        # 翻转
        # if random.randint(0, 1):
        #     angle1 = random.uniform(-10, 10)
        #     M1 = cv2.getRotationMatrix2D((row / 2.0, col / 2.0), angle1, 1)
        #     data = cv2.warpAffine(data, M1, (col, row))
        #     label = cv2.warpAffine(label, M1, (col, row))

        # 放缩
        # if random.randint(0, 1):
        #     scale = random.uniform(0.8, 0.99)  # [0.8, 1)
        #     hei, wid = int(row * scale), int(col * scale)
        #     x = random.randint(0, col - wid)
        #     y = random.randint(0, row - hei)
        #     cropped = data[y:y + hei, x:x + wid]
        #     cropped_l = label[y:y + hei, x:x + wid]
        #     data = cv2.resize(cropped, (row, col))
        #     label = cv2.resize(cropped_l, (row, col))

        # 仿射变换
        if random.randint(0, 1):
            P1 = np.float32([[0, 0], [col - 1, 0], [0, row - 1]])
            P2 = np.float32(
                [[0, row * random.uniform(0, 0.1)], [col * random.uniform(0.9, 1), row * random.uniform(0, 0.1)],
                 [col * random.uniform(0, 0.1), row * random.uniform(0.9, 1)]])
            M2 = cv2.getAffineTransform(P1, P2)
            data = cv2.warpAffine(data, M2, (col, row))
            label = cv2.warpAffine(label, M2, (col, row))

        # 高斯滤波
        # if random.randint(0, 1):
        #     data = cv2.GaussianBlur(data, (5, 5), 0)

        # 直方图均衡化
        # if random.randint(0, 1):
        #     data = np.uint8(data * 255)
        #     data = cv2.equalizeHist(data)
        #     data = data.astype(np.float32) / 255.

        data = np.expand_dims(data, axis=-1)  # 恢复channel维
        label = np.expand_dims(label, axis=-1)  # 恢复channel维
        return data, label

    def __getitem__(self, index):
        mov_image_path = self.images_files['mov'][index]
        fix_image_path = self.images_files['fix'][index]
        mov_label_path = self.labels_files['mov'][index]
        fix_label_path = self.labels_files['fix'][index]

        mov_image = read_image(mov_image_path) / 255.
        mov_label = read_image(mov_label_path)
        mov_image[mov_label == 0] = 0  # 将非肺区设为0

        fix_image = read_image(fix_image_path) / 255.
        fix_label = read_image(fix_label_path)
        fix_image[fix_label == 0] = 0

        fix_image, fix_label = self._tranform(fix_image, fix_label)

        mov_label[mov_label > 0] = 1  # 将肺区的标签统一设为1
        fix_label[fix_label > 0] = 1

        mov_image = to_tensor(mov_image)
        fix_image = to_tensor(fix_image)
        mov_label = to_tensor(mov_label)
        fix_label = to_tensor(fix_label)

        mov_label = relabel(mov_label)
        fix_label = relabel(fix_label)

        if self.mode == 'train':
            mov_label = to_one_hot(mov_label)
            fix_label = to_one_hot(fix_label)

        return (mov_image, fix_image), (mov_label, fix_label)


class ImagePairsDatasetTest_V4(Dataset):
    """
    在 ImagePairsDataset_V1 基础上，使用肺实质图像替换原始图像
    """

    def __init__(self, images_file_path, labels_file_path, mode='train'):
        assert mode in ['train', 'test']
        self.mode = mode

        images_files = read_txt(images_file_path)
        labels_files = read_txt(labels_file_path)

        mov_images_files, fix_images_files = get_pairs(images_files)
        mov_labels_files, fix_labels_files = get_pairs(labels_files)

        self.images_files = {
            'mov': mov_images_files,
            'fix': fix_images_files
        }

        self.labels_files = {
            'mov': mov_labels_files,
            'fix': fix_labels_files
        }

        self.input_size, self.num_labels = get_image_info(labels_files[0], is_label=True)

    def __len__(self):
        return len(self.images_files['mov'])

    def __getitem__(self, index):
        mov_image_path = self.images_files['mov'][index]
        fix_image_path = self.images_files['fix'][index]
        mov_label_path = self.labels_files['mov'][index]
        fix_label_path = self.labels_files['fix'][index]

        mov_file_name = mov_image_path.split('/')[-1][:-2]
        fix_file_name = fix_image_path.split('/')[-1][:-2]

        mov_image = to_tensor(read_image(mov_image_path)) / 255.
        fix_image = to_tensor(read_image(fix_image_path)) / 255.
        mov_label = to_tensor(read_image(mov_label_path))
        fix_label = to_tensor(read_image(fix_label_path))

        mov_label[mov_label > 0] = 1
        fix_label[fix_label > 0] = 1

        mov_image[mov_label == 0] = 0  # 将非肺区置0
        fix_image[fix_label == 0] = 0

        mov_label = relabel(mov_label)
        fix_label = relabel(fix_label)

        if self.mode == 'train':
            mov_label = to_one_hot(mov_label)
            fix_label = to_one_hot(fix_label)

        return (mov_image, fix_image), (mov_label, fix_label), (mov_file_name, fix_file_name)


class ImagePairsDataset_V1(Dataset):
    def __init__(self, images_file_path, labels_file_path, mode='train'):
        assert mode in ['train', 'test']
        self.mode = mode

        images_files = read_txt(images_file_path)
        labels_files = read_txt(labels_file_path)

        mov_images_files, fix_images_files = get_pairs(images_files)
        mov_labels_files, fix_labels_files = get_pairs(labels_files)

        self.images_files = {
            'mov': mov_images_files,
            'fix': fix_images_files
        }

        self.labels_files = {
            'mov': mov_labels_files,
            'fix': fix_labels_files
        }

        self.input_size, self.num_labels = get_image_info(labels_files[0], is_label=True)

    def __len__(self):
        return len(self.images_files['mov'])

    def __getitem__(self, index):
        mov_image_path = self.images_files['mov'][index]
        fix_image_path = self.images_files['fix'][index]
        mov_label_path = self.labels_files['mov'][index]
        fix_label_path = self.labels_files['fix'][index]

        mov_file_name = mov_image_path.split('/')[-1][:-4]
        fix_file_name = fix_image_path.split('/')[-1][:-4]

        mov_image = to_tensor(read_image(mov_image_path)) / 255.
        fix_image = to_tensor(read_image(fix_image_path)) / 255.
        mov_label = to_tensor(read_image(mov_label_path))
        fix_label = to_tensor(read_image(fix_label_path))

        mov_label[mov_label > 0] = 1
        fix_label[fix_label > 0] = 1

        mov_label = relabel(mov_label)
        fix_label = relabel(fix_label)

        if self.mode == 'train':
            mov_label = to_one_hot(mov_label)
            fix_label = to_one_hot(fix_label)

        return (mov_image, fix_image), (mov_label, fix_label), (mov_file_name, fix_file_name)


class LabelsDataset(Dataset):

    def __init__(self, labels_file_path, mode='train'):
        assert mode in ['train', 'test']
        self.mode = mode

        self.labels_files = read_txt(labels_file_path)
        self.input_size, self.num_labels = get_image_info(self.labels_files[0], is_label=True)

    def __len__(self):
        return len(self.labels_files)

    def __getitem__(self, index):
        target_path = self.labels_files[index]
        target_label = to_tensor(read_image(target_path))
        target_label[target_label > 0] = 1  # 将肺区的标签统一设为1
        target_label = relabel(target_label).long()

        if self.mode == 'train':
            input_label = swap_labels(target_label, p=0.1)

            input_label = to_one_hot(input_label)
            target_label = to_one_hot(target_label)

            return input_label, target_label

        target_label = to_one_hot(target_label)

        return target_label
