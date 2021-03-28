import torch
import torch.nn as nn
from torchsummary import torchsummary
from model import EmotionModel
from dateset_loader import ImageLoader
from utils import train, validate, save_checkpoint, adjust_learning_rate
from torch.utils.data import TensorDataset, DataLoader
import tqdm
from dateset_loader import ImageLoader
import cv2
import os

epochs = 500
start_epoch = 0
batch_size = 128
learning_rate = 0.05
momentum = 0.9
weight_decay = 1e-4
print_freq = 100
resume = ''
world_size = -1
rank = -1
dist_backend = 'nccl'
seed = None
gpu = 0


train_path = 'Data/Emotion_Detection_data_kaggle/train'
valid_path = 'Data/Emotion_Detection_data_kaggle/valid'


def data_load(path, image_type='GRAY'):
    images_path = path
    labels_file = path + '/labels.txt'

    load_data = ImageLoader(path, 48, channel=1, number_class=6)
    load_data.normal = True
    with open(labels_file, 'r') as lf:
        labels = lf.readlines()
        lf.close()
    load_data.images_files = []

    number = 0
    for datalines in tqdm.tqdm(load_data.dataset):
        image, label = datalines.strip('\n').split(' ')
        image_path = os.path.join(images_path, image)
        if os.path.isfile(image_path):
            if image_type == 'GRAY':
                image_file = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            else:
                image_file = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image_file = cv2.cvtColor(image_file, cv2.COLOR_BGR2RGB)
            if image_file.mean() == 0 or image_file.std() == 0:
                continue
            image_file = load_data.image_normalization(image_file)
            load_data.images[number] = image_file
            load_data.images_files.append(image_path)
            load_data.images_labels[number] = load_data.one_hot_encode(label)
            number += 1
    return load_data


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('GPU Type:', torch.cuda.get_device_name(0))
model = EmotionModel(number_class=6)
model.to(device)
torchsummary.summary(model, input_size=(1, 48, 48))

train_data = data_load(train_path)
valid_data = data_load(valid_path)

print('Train images shape:', train_data.images.shape)
print('Train labels shape:', train_data.images_labels.shape)
print('Validation images shape:', valid_data.images.shape)
print('Validation labels shape:', valid_data.images_labels.shape)

train_torch_dataset = TensorDataset(torch.FloatTensor(train_data.images), torch.FloatTensor(train_data.images_labels))
train_dataloader = DataLoader(dataset=train_torch_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
valid_torch_dataset = TensorDataset(torch.FloatTensor(valid_data.images), torch.FloatTensor(valid_data.images_labels))
valid_dataloader = DataLoader(dataset=valid_torch_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

best_acc1 = 0
criterion = nn.BCEWithLogitsLoss().cuda(0)
optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum, weight_decay)
for epoch in range(epochs):
    adjust_learning_rate(optimizer, epoch, learning_rate)
    train(train_dataloader, model, criterion, optimizer, epoch, gpu, print_freq)
    acc1 = validate(valid_dataloader, model, criterion, gpu, print_freq)
    is_best = acc1 > best_acc1
    best_acc1 = max(acc1, best_acc1)
    save_checkpoint(model, {'epoch': epoch + 1,
                            'state_dict': model,
                            'best_acc1': best_acc1,
                            'optimizer': optimizer.state_dict(), }, is_best)
