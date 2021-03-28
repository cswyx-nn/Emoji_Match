import cv2
import os

work_path = os.getcwd()
train_path = 'Data/MMAFEDB/train'
valid_path = 'Data/MMAFEDB/valid'

train_list = os.listdir(train_path)
# label_dic = {'angry': '0',
#              'fearful': '1',
#              'happy': '2',
#              'neutral': '3',
#              'sad': '4',
#              'surprised': '5'}
label_dic = {'angry': '0',
             'fear': '1',
             'happy': '2',
             'neutral': '3',
             'sad': '4',
             'surprise': '5'}



train_labels = []
valid_labels = []

for i in train_list:
    label = label_dic[i]
    train_image_path = train_path + '/' + i
    valid_image_path = valid_path + '/' + i
    train_images_name = os.listdir(train_image_path)
    valid_images_name = os.listdir(valid_image_path)
    for image in train_images_name:
        label_item = i+'/'+image + ' ' + label
        train_labels.append(label_item)
    for image in valid_images_name:
        label_item = i+'/'+image + ' ' + label
        valid_labels.append(label_item)

print(len(train_labels))
label_file = 'Data/MMAFEDB/train/labels.txt'
with open(label_file, 'w') as lf:
    for i in train_labels:
        data = i + '\n'
        lf.writelines(data)
    lf.close()

label_file = 'Data/MMAFEDB/valid/labels.txt'
with open(label_file, 'w') as lf:
    for i in valid_labels:
        data = i + '\n'
        lf.writelines(data)
    lf.close()