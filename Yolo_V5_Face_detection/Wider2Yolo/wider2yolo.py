import cv2
import os

label_file_path = 'Wider/wider_face_split/wider_face_val_bbx_gt.txt'
with open(label_file_path, 'r') as f:
    labels = f.readlines()

image_names = []
image_no = []

for i, data in enumerate(labels):
    dataline = data.strip('\n')
    if dataline.split('.')[-1] == 'jpg':
        image_names.append(dataline)
        image_no.append(int(i))

image_no.append(int(len(labels)-1))

print(image_names[1])
print(image_no[0])
dataset = dict()
for i in range(len(image_names)):
    dataset[image_names[i]] = labels[image_no[i]+2: image_no[i+1]]

image_path = 'Wider/WIDER_val/images'
for i in dataset.keys():
    image_name = os.path.join(image_path, i)
    print(image_name)
    image = cv2.imread(image_name)
    h, w, ch = image.shape
    label_list = dataset[i]
    name = i.split('/')[-1].split('.')[0] + '.txt'
    out_data = ''
    for j in label_list:
        label_data = j.strip(' \n').split(' ')
        if int(label_data[7]) == 0 and int(label_data[8]) == 0 and int(label_data[9]) == 0:
            out_data = out_data + '0 '
            x1 = float(int(label_data[0]) + int(label_data[2])//2)/w
            y1 = float(int(label_data[1]) + int(label_data[3])//2)/h
            w1 = float(label_data[2])/w
            h1 = float(label_data[3])/h
            out_data = out_data + '%s %s %s %s\n' % (str(x1), str(y1), str(w1), str(h1))

    with open('Wider/WIDER_val/labels/' + name, 'w') as fn:
        fn.writelines(out_data)
        fn.close()
