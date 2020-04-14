@echo off
FOR %%f IN (*.webm) DO (
  @echo "Processing %%f file..."
  REM take action on each file. %%f store current file name 
  if not exist frames\%%~nf mkdir frames\%%~nf
  ffmpeg -i "%%f" -f image2 frames\%%~nf\%%06d.jpg
)
