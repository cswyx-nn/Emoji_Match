# Emotion_Classification

## Introduction

The focus of this project is to classify emotions based on pictures of human faces.

This model can classify human face emotion into 6 categories: happy, sad, neutral, surprised, fearful and angry.

## Dataset

The data used to train this model is FER-2013 facial expression dataset from Kaggle.

Here is link to this dataset: https://www.kaggle.com/msambare/fer2013


## Usage
### Data storage structure
Save your data according to the following structure:

    Data
    └─Emotion_Detection_data_kaggle
        ├─train
        │  ├─category_0
        │  ├─category_1
        │  ├─category_2
        │  ├─category_3
        │  ├─...
        │  └─category_n
        └─valid
            ├─category_0
            ├─category_1
            ├─category_2
            ├─category_3
            ├─...
            └─category_n
### Data preprocess
#### For windows:
Edit the run_Data_preprocess.bat as following structure:

    @echo off
    call "**Anaconda_Path**\Scripts\activate.bat" "**Environment_Path**"
    d:
    cd **Work_Path**
    set datapath=Data/Emotion_Detection_data_kaggle
    set label=angry,fearful,happy,neutral,sad,surprised
    echo ----Start Main----
    python "**Work_Path**\Data_preprocess.py"  %datapath% %label%
    pause
    
Double click to run
#### For Mac and Linux:
    python "**Work_Path**\Data_preprocess.py"  Data/Emotion_Detection_data_kaggle angry,fearful,happy,neutral,sad,surprised

### Train
#### For windows:
Edit the run_train.bat as following structure:

    @echo off
    call "**Anaconda_Path**\Scripts\activate.bat" "**Environment_Path**"
    d:
    cd **Work_Path**
    echo ----Start Main----
    python "**Work_Path**\train.py" --data-path Data/Emotion_Detection_data_kaggle --batch-size 128 --print_freq 50
    pause
Double click to run
#### For Mac and Linux:
    python "**Work_Path**\train.py"  --data-path Data/Emotion_Detection_data_kaggle --batch-size 128 --print_freq 50

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
Visualization：
![image](https://github.com/cswyx-nn/Emoji_Match/blob/main/Emotion_Classification/image/Model_Structure.png)

Structure：

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



