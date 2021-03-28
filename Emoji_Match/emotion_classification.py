import torch
import torch.nn as nn
import cv2
import os
import numpy as np

model_path = 'classification_checkpoint/model_best.pth.tar'
model = torch.load(model_path)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

label_dic = {'0': 'angry',
             '1': 'fearful',
             '2': 'happy',
             '3': 'neutral',
             '4': 'sad',
             '5': 'surprised'}


def image_normalization(image, size):
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
        image_out = cv2.resize(image_out, (size, size))
        image_out = np.reshape(image_out, (3, size, size))
        return image_out
    else:
        img_mean = np.mean(image)
        img_std = np.std(image)
        image_out = np.zeros((h, w, 1), dtype='float32')
        image_out[:, :, 0] = np.array((image - img_mean) / img_std, dtype='float32')
        image_out = cv2.resize(image_out, (size, size))
        image_out = np.reshape(image_out, (1, size, size))
        return image_out


def emotion_classification(img_test, image_size=48, channel=1):
    test_images = np.zeros((1, channel, image_size, image_size), dtype='float')
    img = image_normalization(img_test, image_size)
    test_images[0] = img
    with torch.no_grad():
        test = torch.Tensor(test_images)
        test = test.cuda(device, non_blocking=True)
        logits = model(test)
        out = torch.softmax(logits, dim=1)
        result = out.topk(1, 1, True, True)
        result = result.indices.cpu().numpy()[0][0]
        out_label = label_dic[str(result)]
        return out_label



