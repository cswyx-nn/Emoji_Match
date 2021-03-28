from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import torch
import os


class ImageLoader(Dataset):
    def __init__(self, path, size, channel, number_class):
        super(ImageLoader, self).__init__()
        self.path = path
        self.dataset = []
        self.dataset.extend(open(self.path + '/labels.txt').readlines())
        self.images = np.zeros((len(self.dataset), channel, size, size), dtype='float')
        self.images_files = None
        self.images_labels = np.zeros((len(self.dataset), number_class), dtype='int64')
        self.images_size = size

    def image_normalization(self, image):
        if len(image.shape) == 3:
            h, w, ch = image.shape
        else:
            h, w = image.shape
            ch = 1
        if ch == 3:
            img_r = image[:, :, 0]
            img_g = image[:, :, 1]
            img_b = image[:, :, 2]
            mean_r, mean_g, mean_b = (np.mean(img_r), np.mean(img_g), np.mean(img_b))
            std_r, std_g, std_b = (np.std(img_r), np.std(img_g), np.std(img_b))

            image_out = np.zeros(image.shape, dtype='float32')
            image_out[:, :, 0] = np.array((img_r - mean_r) / std_r, dtype='float32')
            image_out[:, :, 1] = np.array((img_g - mean_g) / std_g, dtype='float32')
            image_out[:, :, 2] = np.array((img_b - mean_b) / std_b, dtype='float32')
            image_out = cv2.resize(image_out, (self.images_size, self.images_size))
            image_out = np.reshape(image_out, (3, self.images_size, self.images_size))
            return image_out
        else:
            img_mean = np.mean(image)
            img_std = np.std(image)
            image_out = np.zeros((h, w, 1), dtype='float32')
            image_out[:, :, 0] = np.array((image - img_mean) / img_std, dtype='float32')
            image_out = cv2.resize(image_out, (self.images_size, self.images_size))
            image_out = np.reshape(image_out, (1, self.images_size, self.images_size))
            return image_out



    def __len__(self):
        return len(self.dataset)

    def one_hot_encode(self, label):
        label = int(label)
        one_hot_out = [0] * label + [1] + [0] * (5 - label)
        return np.array(one_hot_out)


if __name__ == '__main__':
    path = 'Data/Emotion_Detection_data_kaggle/train'
    load_data = ImageLoader(path, 48)

    images_path = path
    labels_file = path + '/labels.txt'

    load_data.images_files = []

    number = 0
    for datalines in load_data.dataset:
        image, label = datalines.strip('\n').split(' ')
        image_path = os.path.join(images_path, image)
        if os.path.isfile(image_path):
            image_file = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image_file = cv2.cvtColor(image_file, cv2.COLOR_BGR2RGB)
            if image_file.mean() == 0 or image_file.std() == 0:
                continue
            image_file = load_data.image_normalization(image_file)
            load_data.images[number] = image_file
            load_data.images_files.append(image_path)
            load_data.images_labels[number] = load_data.one_hot_encode(label)
            number += 1
