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

	while( true ){
		capture>>frame;
		std::vector<cv::Mat> hsv;
		cv::cvtColor(frame, frame, CV_BGR2HSV);
		cv::medianBlur(frame, frame, 5);

		cv::split(frame, hsv);

		for(size_t K=0; K<hsv.size(); ++K)
			cv::imshow(windows[K], hsv[K]);
		if(cv::waitKey(10)>=0) break;
	}
	return 0;
}


