# Emoji_MATCH (EM)

## Introduction
This project uses a trained face detection and a trained emotion classification model to recognize the facial expressions of people in videos and pictures.

This model will complete face detection first, then it will classify human facial expression into 6 categories: happy, sad, neutral, fearful, surprised and angry.

Then this model will add emoji with the same label to the input photo.

The emoji used are as below:

😡  for angry
😱  for fearful
😁  for happy
😐  for neutral
😭  for sad
😮  for surprised


## Usage
The model and pre-trained weight files are already included in the file. Only need to follow the usage introduction as follows to realize the recognition of character expressions.

### For Windows:
Edit Run.bat as follow:

    @echo off
    call "**Path of anaconda/Scripts/activate.bat**" "**Your Environment Path**"
    d:
    cd **Emoji_Match work path**
    echo ----Start Main----
    python **Emoji_Match work path/main.py** **Show Label(1 or 0)** **Show Bounding Box(1 or 0)** **Image source**
    pause

   1. Your Environment Path is folder of the python environment.
   2. Both "Show Label" and "Show Bounding Box" use 1 and 0 to control the result display of the video (1 = True, 0 = False).   
   3. Image source can be a video, Camera number or webcam address

Double-click run.bat to start execution.

### For MAC and Linux:
1.Open the mac terminal and enter the Emoji_Match path: 
    
    cd **Your_Working_Path** 
2.Grant execution permission to main.py: 
    
    chmod a+x Your_Working_Path/main.py
3.Input command: 

    python main.py  *Show_Label(1or0)*  *Show_BoundingBox(1or0)*  *Image_source*


