#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "aux.h"

int main(int argc, char **argv){
	cv::VideoCapture capture(0);
	cv::Mat frame;

	std::string 
		windows[] = {"hue", "saturation", "value"};
	for(size_t K=0; K<3; ++K)
		cv::namedWindow(windows[K], CV_WINDOW_KEEPRATIO);

	int value[3] = {0};
	int alpha = 5;
	cv::namedWindow("alpha", CV_WINDOW_KEEPRATIO);
	cv::createTrackbar("alpha","alpha", &alpha, 255, NULL, NULL);

	for(size_t K=0; K<3; ++K)
		cv::createTrackbar(windows[K],windows[K], value+K, 255, NULL, NULL);

	while( true ){
		capture>>frame;
		std::vector<cv::Mat> hsv;
		cv::cvtColor(frame, frame, CV_BGR2HSV);
		cv::medianBlur(frame, frame, 5); //quitar ruido impulsivo

		cv::split(frame, hsv);

		cv::equalizeHist(hsv[2], hsv[2]);
		cv::merge(hsv, frame);

		//umbralizacion
		cv::Scalar hsv_min(value[0]-alpha, value[1]-alpha, value[2]-alpha, 0);
		cv::Scalar hsv_max(value[0]+alpha, value[1]+alpha, value[2]+alpha, 0);
		for(size_t K=0; K<hsv.size(); ++K){
			cv::Mat aux;
			cv::inRange(hsv[K], cv::Scalar::all(hsv_min[K]), cv::Scalar::all(hsv_max[K]), hsv[K]);
		}

		cv::medianBlur(frame, frame, 3); //eliminar huecos internos

		for(size_t K=0; K<hsv.size(); ++K)
			cv::imshow(windows[K], hsv[K]);

		cv::cvtColor(frame, frame, CV_HSV2BGR);

		cv::imshow("alpha", frame);

		if(cv::waitKey(10)>=0) break;
	}
	return 0;
}


