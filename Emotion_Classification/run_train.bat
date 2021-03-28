@echo off
call "**Anaconda_Path**\Scripts\activate.bat" "**Environment_Path**"
d:
cd **Work_Path**
echo ----Start Main----
python "**Work_Path**\train.py" --data-path Data/Emotion_Detection_data_kaggle --batch-size 128 --print_freq 50
pause
