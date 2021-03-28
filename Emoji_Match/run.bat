@echo off
call "C:\anaconda3\Scripts\activate.bat" "D:\Project\Miaomiao\Project\Environment"
d:
cd D:\XieNing_Porject\Emoji_Match\Emoji_Match
echo ----Start Main----
python "D:\XieNing_Porject\Emoji_Match\Emoji_Match\main.py" 1 1 http://192.168.1.138:4747/video
pause