#!/bin/bash

#make

file=$1
g++ "${file}" -I./ -O2 -c -o "${file%.cpp}.o" -Wall
g++ "${file%.cpp}.o" aux.o -lopencv_{core,highgui,imgproc,video} -o "${file%.cpp}.bin"
