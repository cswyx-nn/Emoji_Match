@echo off
call "C:\anaconda3\Scripts\activate.bat" "D:\Project\Miaomiao\Project\Environment"
d:
cd D:\XieNing_Porject\Emoji_Match\emotion_classification_v2
echo ----Start Main----
python "D:\XieNing_Porject\Emoji_Match\emotion_classification_v2\train.py" --data-path Data/Emotion_Detection_data_kaggle --batch-size 128 --print_freq 50
pause