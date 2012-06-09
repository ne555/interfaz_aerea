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

	int value[3] = {8};
	int alpha = 7;
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

		cv::merge(hsv, frame);

		//umbralizacion
		cv::Scalar hsv_min = cv::Scalar::all(value[0]-alpha);
		cv::Scalar hsv_max = cv::Scalar::all(value[0]+alpha);
		cv::inRange(hsv[0], cv::Scalar::all(hsv_min[0]), cv::Scalar::all(hsv_max[0]), hsv[0]);

		//eliminar huecos internos
		cv::medianBlur(hsv[0], hsv[0], 5); 
		cv::dilate(hsv[0], hsv[0], cv::Mat());


		//enmascarar los otros canales
		for(size_t K=1; K<hsv.size(); ++K)
			for(size_t L=0; L<hsv[K].rows; ++L)
				for(size_t M=0; M<hsv[K].cols; ++M)
					hsv[K].at<byte>(L,M) *= (hsv[0].at<byte>(L,M)==255);

		byte color = 50;
		for(size_t K=0; K<hsv[0].rows; ++K)
			for(size_t L=0; L<hsv[0].cols; ++L){
				if(hsv[0].at<byte>(K,L)==255){
					cv::floodFill(hsv[0], cv::Point(L,K), color);
					color += 10;
				}
			}



		for(size_t K=0; K<hsv.size(); ++K)
			cv::imshow(windows[K], hsv[K]);

		cv::cvtColor(frame, frame, CV_HSV2BGR);

		cv::imshow("alpha", frame);

		if(cv::waitKey(10)>=0) break;
	}
	return 0;
}


