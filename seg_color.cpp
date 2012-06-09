#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "aux.h"

typedef unsigned char byte;

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

	std::vector<cv::Mat> viejo( 3, cv::Mat::zeros(480,640,CV_8U));

	while( true ){
		capture>>frame;
		std::vector<cv::Mat> hsv;
		cv::cvtColor(frame, frame, CV_BGR2HSV);
		cv::split(frame, hsv);

		{
			cv::Rect roi(0,0,10,10);
			cv::Mat 
				h=cv::Mat(hsv[0],roi),
				s=cv::Mat(hsv[1],roi),
				v=cv::Mat(hsv[2],roi);
		}


		cv::Scalar hsv_min(value[0]-alpha, value[1]-alpha, value[2]-alpha, 0);
		cv::Scalar hsv_max(value[0]+alpha, value[1]+alpha, value[2]+alpha, 0);

		//for(size_t K=0; K<hsv.size(); ++K)
		#if 0
		{
			cv::Mat aux;
			cv::inRange(hsv[2], cv::Scalar::all(alpha), cv::Scalar::all(255-alpha), aux);
			for(size_t K=0; K<hsv.size(); ++K)
				for(size_t L=0; L<hsv[K].rows; ++L)
					for(size_t M=0; M<hsv[K].cols; ++M)
						hsv[K].at<byte>(L,M) *= (aux.at<byte>(L,M)==255);
		}
		#endif

		#if 1
		for(size_t K=0; K<hsv.size(); ++K){
			cv::Mat aux;
			cv::inRange(hsv[K], cv::Scalar::all(hsv_min[K]), cv::Scalar::all(hsv_max[K]), aux);
			#if 1
			for(size_t L=0; L<hsv[K].rows; ++L)
				for(size_t M=0; M<hsv[K].cols; ++M)
					if(K==0){
						if(aux.at<byte>(L,M)==255)
							hsv[K].at<byte>(L,M) *= -1;
						else 
							hsv[K].at<byte>(L,M) = 0;

					}
					else
						hsv[K].at<byte>(L,M) *= aux.at<byte>(L,M)==255;
			//hsv[K] = aux.mul(hsv[K]);
			#endif
		}
		#endif 

		#if 0
		for(int i=0;i<hsv.size();i++){
			cv::Mat aux = hsv[i].clone();
			hsv[i]= hsv[i] - viejo[i];
			viejo[i]=aux;
		}
		#endif
		
		#if 0
		for(size_t K=0; K<hsv.size(); ++K){
			cv::GaussianBlur(hsv[K], hsv[K], cv::Size(9,9), 1.5, 1.5);
			cv::Canny(hsv[K], hsv[K], 0, 50, 3, true);
		}
		#endif
		
		#if 0
		for(size_t K=0; K<hsv.size(); ++K){
			std::vector<cv::Vec3f> storage;
			int max_r = frame.rows/4 , min_r = 10;
			cv::HoughCircles(hsv[K], storage, CV_HOUGH_GRADIENT, 1, frame.rows/4 , 100, 50, min_r, max_r);

			for(size_t L=0; L<storage.size(); ++L)
				cv::circle(
					hsv[K],
					cv::Point(storage[L][0],storage[L][1]),
					storage[L][2],
					cv::Scalar::all(1),
					1
				);

		}
		#endif
		
		
		for(size_t K=0; K<hsv.size(); ++K)
			cv::imshow(windows[K], hsv[K]);
		//cv::imshow("alpha", frame);

		if(cv::waitKey(30)>=0) break;
	}



	return 0;
}


