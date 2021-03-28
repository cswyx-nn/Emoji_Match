import torch
import torch.nn as nn
from torchsummary import torchsummary
from model import EmotionModel
import argparse
from dateset_loader import ImageLoader
from utils import train, validate, save_checkpoint, adjust_learning_rate
from torch.utils.data import TensorDataset, DataLoader
import tqdm
from dateset_loader import ImageLoader
import cv2
import os


# Load Function
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


def main(opt):
    # Set up train parameter
    epochs = opt.epochs
    batch_size = opt.batch_size
    learning_rate = opt.learning_rate
    momentum = opt.momentum
    weight_decay = opt.weight_decay
    print_freq = opt.print_freq
    resume = ''
    world_size = -1
    rank = -1
    dist_backend = 'nccl'
    seed = None
    gpu = opt.gpu
    data_path = opt.data_path
    train_path = data_path + '/train'
    valid_path = data_path + '/valid'

    # Set up execution equipment
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('GPU Type:', torch.cuda.get_device_name(0))
    # Load Model to GPU
    model = EmotionModel(number_class=6)
    model.to(device)
    # Show model structure
    torchsummary.summary(model, input_size=(1, 48, 48))

    # Load train and validation data
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

    # Ser up loss function and optimizer
    best_acc1 = 0
    criterion = nn.BCEWithLogitsLoss().cuda(0)
    optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum, weight_decay)
    # Start Train
    for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch, learning_rate)
        train(train_dataloader, model, criterion, optimizer, epoch, gpu, print_freq)
        acc1 = validate(valid_dataloader, model, criterion, gpu, print_freq)
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        # If accuracy > best accuracy then save model
        save_checkpoint(model, {'epoch': epoch + 1,
                                'state_dict': model,
                                'best_acc1': best_acc1,
                                'optimizer': optimizer.state_dict(), }, is_best)


if __name__ == '__main__':
    # Read CMD input parameter
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='Data', help='Train Data Path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--learning-rate', type=float, default=0.05, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight-decay')
    parser.add_argument('--print_freq', type=int, default=15, help='train process information print frequency')
    parser.add_argument('--gpu', type=int, default=0, help='whether GPU')

    opt = parser.parse_args()

    main(opt)