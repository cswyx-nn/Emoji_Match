# YOLO_V5 Face datection Project

## Introduction
This is a sub-project of the Emoji_Match.


## Dataset
Use Wider2Yolo.py to transfer the label of Wider Face into Yolo structure.
Wider Face dataset is a face detection benchmark dataset, of which images are selected from the publicly available WIDER dataset. 
WIDER FACE is also one of the most difficult face detection datasets, which includes 32,203 images and label 393,703 faces with a high degree of variability in scale, pose and occlusion as depicted in the sample images. 

    @inproceedings{yang2016wider,
    Author = {Yang, Shuo and Luo, Ping and Loy, Chen Change and Tang, Xiaoou},
    Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    Title = {WIDER FACE: A Face Detection Benchmark},
    Year = {2016}}


## Model

Use YOLO_V5s model to do Transfer learning and train my face detection model.
YOLO_V5 is one of the most powerful models, which can perform classification tasks and regression tasks at the same time. 
The model can perform detection and classification of 80 different targets in the COCO data set. 
At the same time, the model is famous for its small model parameters but powerful performance. 
The YOLO_V5 series model parameters are as follows:

    Model	  size  SpeedV100  FPSV100   params	
    YOLOv5s	  640    2.2ms      455      7.3M	
    YOLOv5m	  640    2.9ms      345      21.4M	
    YOLOv5l	  640    3.8ms      264      47.0M	
    YOLOv5x	  640    6.0ms      167      87.7M	

YOLO_V5s has only 7.3M parameters, which allows the model to have a very fast response on devices with average performance or even mobile devices.
Based on these characteristic, I choose YOLO_V5s as the face detection model.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4418161.svg)](https://doi.org/10.5281/zenodo.4418161)
