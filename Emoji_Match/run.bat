@echo off
call "**Path of anaconda/Scripts/activate.bat**" "**Your Environment Path**"
d:
cd **Emoji_Match work path**
echo ----Start Main----
python **Emoji_Match work path/main.py** **Show Label(1 or 0)** **Show Bounding Box(1 or 0)** **http:// + webcam address**
pause
