import cv2
import os
from PIL import Image
import numpy as np


emoji_path = [os.path.join('emoji', i) for i in os.listdir('emoji')]
emoji_dic = {'angry': None,
             'fearful': None,
             'happy': None,
             'neutral': None,
             'sad': None,
             'surprised': None}
for emoji in emoji_path:
    emoji_image = Image.open(emoji)
    emoji_image = emoji_image.resize((64, 64))
    label = emoji.split('\\')[-1].split('.')[0]
    emoji_dic[label] = emoji_image


def add_emoji(image, label, bbox):
    x1, y1, x2, y2 = bbox
    image_original = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    emoji_img = emoji_dic[label]
    layer = Image.new('RGBA', image_original.size, (0, 0, 0, 0))

    layer.paste(emoji_img, (x1+(x2-x1)//2-32, y1 - 100))
    out = Image.composite(layer, image_original, layer)
    img = cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR)
    return img


if __name__ == '__main__':
    image = cv2.imread('test/test.jpg')
    img = add_emoji(image, 'happy', [50, 300, 80, 400])
    cv2.imshow('1', img)
    cv2.waitKey()




