CC=g++
Program= ejemplo.bin
FICHERO= ejemplo.cpp
LIBS=  -lopencv_core -lopencv_imgproc -lopencv_calib3d -lopencv_video -lopencv_features2d -lopencv_ml -lopencv_highgui -lopencv_objdetect -lopencv_contrib -lopencv_legacy
CXXFLAGS=-O2  

$(Program): $(FICHERO) 
	$(CC) $(CXXFLAGS) $(FICHERO) $(LIBS)  $(LIB)  -o $(Program) 
	ls -l $(Program)

