#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "aux.h"

typedef unsigned char byte;

int main(int argc, char **argv){
	cv::VideoCapture capture(0);
	cv::Mat frame, img;
	std::string 
		windows[] = {"hue", "saturation", "value"};
	for(size_t K=0; K<3; ++K)
		cv::namedWindow(windows[K], CV_WINDOW_KEEPRATIO);

	
	int value[3] = {7,117,65};
	int alpha[3] = {6,55,35};
	cv::namedWindow("alpha", CV_WINDOW_KEEPRATIO);
	cv::namedWindow("logico", CV_WINDOW_KEEPRATIO);
	cv::createTrackbar("alpha0","alpha", &alpha[0], 255, NULL, NULL);
	cv::createTrackbar("alpha1","alpha", &alpha[1], 255, NULL, NULL);
	cv::createTrackbar("alpha2","alpha", &alpha[2], 255, NULL, NULL);

	for(size_t K=0; K<3; ++K)
		cv::createTrackbar(windows[K],windows[K], value+K, 255, NULL, NULL);

	std::vector<cv::Mat> viejo( 3, cv::Mat::zeros(480,640,CV_8U));
	
	while( true ){
		capture>>frame;
		cv::flip(frame,frame,1);
		std::vector<cv::Mat> hsv;
		cv::cvtColor(frame, frame, CV_BGR2HSV);
		cv::split(frame, hsv);

		for(size_t K=0; K<hsv.size(); ++K)
			cv::imshow(windows[K]+"orig", hsv[K]);

		cv::Scalar hsv_min(value[0]-alpha[0], value[1]-alpha[1], value[2]-alpha[2], 0);
		cv::Scalar hsv_max(value[0]+alpha[0], value[1]+alpha[1], value[2]+alpha[2], 0);


		for(size_t K=0; K<hsv.size(); ++K){
			cv::inRange(hsv[K], cv::Scalar::all(hsv_min[K]), cv::Scalar::all(hsv_max[K]), hsv[K]);
		}
 

		bitwise_and(hsv[1],hsv[0],img);
		bitwise_and(img,hsv[2],img);
		
		cv::imshow("logico", img);
		
		for(size_t K=0; K<hsv.size(); ++K)
			cv::imshow(windows[K], hsv[K]);

		if(cv::waitKey(30)>=0) break;
	}



	return 0;
}


