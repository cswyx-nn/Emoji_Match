@echo off
call "C:\anaconda3\Scripts\activate.bat" "D:\Project\Miaomiao\Project\Environment"
d:
cd D:\XieNing_Porject\Emoji_Match\emotion_classification_v2
set datapath=Data/Emotion_Detection_data_kaggle
set label=angry,fearful,happy,neutral,sad,surprised
echo ----Start Main----
python "D:\XieNing_Porject\Emoji_Match\emotion_classification_v2\Data_preprocess.py"  %datapath% %label%
pause