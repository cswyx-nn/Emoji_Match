import torch
import torch.nn as nn
from torch.nn import functional as F
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential


class EmotionModel(nn.Module):
    def __init__(self, number_class=6):
        super(EmotionModel, self).__init__()
        self.number_class = number_class
        self.Conv_block_1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5, 5)),
                                          nn.ReLU(),
                                          nn.MaxPool2d((5, 5), (2, 2)))
        self.Conv_block_2 = nn.Sequential(nn.Conv2d(64, 64, (3, 3)),
                                          nn.ReLU(),
                                          nn.Conv2d(64, 64, (3, 3)),
                                          nn.ReLU(),
                                          nn.AvgPool2d((3, 3), (2, 2)))
        self.Conv_block_3 = nn.Sequential(nn.Conv2d(64, 128, (3, 3)),
                                          nn.ReLU(),
                                          nn.Conv2d(128, 128, (3, 3)),
                                          nn.ReLU(),
                                          nn.AvgPool2d((3, 3), (2, 2)))
        self.flatten = nn.Flatten()

        self.Dense_block = nn.Sequential(nn.Linear(128, 1024),
                                         nn.ReLU(),
                                         nn.Dropout(0.2),
                                         nn.Linear(1024, 1024),
                                         nn.ReLU(),
                                         nn.Dropout(0.2),
                                         nn.Linear(1024, self.number_class),
                                         nn.Softmax(dim=1))

    def forward(self, inputs):
        x = inputs
        x = self.Conv_block_1(x)
        x = self.Conv_block_2(x)
        x = self.Conv_block_3(x)
        x = self.flatten(x)
        x = self.Dense_block(x)
        return x




