# Emoji_MATCH (EM)

## Usage
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
