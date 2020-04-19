#!/bin/bash

for f in *.mp4
do
  echo "Processing $f file..."
  # take action on each file. $f store current file name\
  basename=$(ff=${f%.ext} ; echo ${ff##*/})
  name=$(echo $basename | cut -d'.' -f1)
  mkdir -p frames/"$name"
  ffmpeg -i "$f" -f image2 frames/"$name"/%06d.jpg
done

