@echo off
FOR /R %%f IN (*.webm) DO (
  @echo "Processing %%f file..."
  REM take action on each file. %%f store current file name 
  if not exist "%%~dpf\frames\%%~nf" mkdir "%%~dpf\frames\%%~nf"
  ffmpeg -i "%%f" -f image2 "%%~dpf\frames\%%~nf\%%06d.jpg"
)
