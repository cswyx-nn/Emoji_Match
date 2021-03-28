# Emotion_Classification

## Introduction

The focus of this project is to classify emotions based on pictures of human faces.

## Dataset
FER-2013 from Kaggle.

The data consists of 48x48 pixel grayscale images of faces. 
The faces have been automatically registered so that the face is more or less centred and occupies about the same amount of space in each image.

The task is to categorize each face based on the emotion shown in the facial expression into one of seven categories:  
    
    0 = Angry
    1 = Disgust
    2 = Fear
    3 = Happy
    4 = Sad
    5 = Surprise
    6 = Neutral 

The training set consists of 28,709 examples and the public test set consists of 3,589 examples.

## Model
### Introduction
This model refers to the expression classification model of Deop Face. I used Pytorch to build and train the classification model.

In the FER-2013 data set, the amount of Disgust label images is too small, so during the model training process, I discarded the label and set the classification to 6 categories.

### Structure

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                   Input            [-1, 1, 48, 48]               0 
                Conv2d-1           [-1, 64, 44, 44]           1,664
                  ReLU-2           [-1, 64, 44, 44]               0
             MaxPool2d-3           [-1, 64, 20, 20]               0
                Conv2d-4           [-1, 64, 18, 18]          36,928
                  ReLU-5           [-1, 64, 18, 18]               0
                Conv2d-6           [-1, 64, 16, 16]          36,928
                  ReLU-7           [-1, 64, 16, 16]               0
             AvgPool2d-8             [-1, 64, 7, 7]               0
                Conv2d-9            [-1, 128, 5, 5]          73,856
                 ReLU-10            [-1, 128, 5, 5]               0
               Conv2d-11            [-1, 128, 3, 3]         147,584
                 ReLU-12            [-1, 128, 3, 3]               0
            AvgPool2d-13            [-1, 128, 1, 1]               0
              Flatten-14                  [-1, 128]               0
               Linear-15                 [-1, 1024]         132,096
                 ReLU-16                 [-1, 1024]               0
              Dropout-17                 [-1, 1024]               0
               Linear-18                 [-1, 1024]       1,049,600
                 ReLU-19                 [-1, 1024]               0
              Dropout-20                 [-1, 1024]               0
               Linear-21                    [-1, 6]           6,150
              Softmax-22                    [-1, 6]               0
    ================================================================
    Total params: 1,484,806
    Trainable params: 1,484,806
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.01
    Forward/backward pass size (MB): 2.79
    Params size (MB): 5.66
    Estimated Total Size (MB): 8.46
    ----------------------------------------------------------------


## Usage
