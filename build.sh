#!/bin/bash

#make

file=$1
g++ "${file}" -I./ -O2 -c -o "${file%.cpp}.o" 
g++ "${file%.cpp}.o" aux.o -lopencv_{core,highgui,imgproc} -o "${file%.cpp}.bin"
