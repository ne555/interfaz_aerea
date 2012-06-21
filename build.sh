#!/bin/bash

#make

file=$1
g++ "${file}" -I./ -ggdb -c -o "${file%.cpp}.o" 
g++ "${file%.cpp}.o" aux.o -lopencv_{core,highgui,imgproc,video} -o "${file%.cpp}.bin"
