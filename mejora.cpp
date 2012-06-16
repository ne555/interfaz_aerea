#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "aux.h"
#include <algorithm>

double dist(double centr_x,double centr_y,int L,int K){
	return (double) sqrt((centr_x-L)*(centr_x-L)+(centr_y-K)*(centr_y-K));
}

int main(int argc, char **argv){
	cv::VideoCapture capture(0);
	cv::Mat frame;

	std::string 
		windows[] = {"hue", "saturation", "value"};
	for(size_t K=0; K<3; ++K)
		cv::namedWindow(windows[K], CV_WINDOW_KEEPRATIO);
	
	cv::namedWindow("Dibujo", CV_WINDOW_AUTOSIZE);	

	int value[3] = {8};
	int alpha = 7;
	cv::namedWindow("alpha", CV_WINDOW_KEEPRATIO);
	cv::createTrackbar("alpha","alpha", &alpha, 255, NULL, NULL);

	for(size_t K=0; K<3; ++K)
		cv::createTrackbar(windows[K],windows[K], value+K, 255, NULL, NULL);
	cv::Mat frame2=cv::Mat::zeros(480,640,CV_8U);
	while( true ){
		capture>>frame;
		cv::flip(frame, frame, 1);
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
		cv::dilate(hsv[0], hsv[0], cv::Mat::ones(5,5,CV_8U));



		//identificacion de regiones
		byte color = 50;
		for(size_t K=0; K<hsv[0].rows; ++K)
			for(size_t L=0; L<hsv[0].cols; ++L){
				if(hsv[0].at<byte>(K,L)==255){
					cv::floodFill(hsv[0], cv::Point(L,K), color);
					color += 10;
				}
			}

		//Sobreviven los mas grandes
		cv::Mat freq = histogram(hsv[0]);
		*freq.begin<float>() = 0;
		cv::MatIterator_<float> max1 = std::max_element( freq.begin<float>(), freq.end<float>() );
		double n1 = *max1;
		*max1 = 0;
		cv::MatIterator_<float> max2 = std::max_element( freq.begin<float>(), freq.end<float>() );
		double n2 = *max2;	
		*max2 = 0;
		size_t color1 = std::distance( freq.begin<float>(), max1 );
		size_t color2 = std::distance( freq.begin<float>(), max2 );

		int ncolor1=0;
		int ncolor2=0;
		double centr1_x=0;
		double centr1_y=0;
		double centr2_x=0;
		double centr2_y=0;


		for(size_t K=0; K<hsv[0].rows; ++K)
			for(size_t L=0; L<hsv[0].cols; ++L){
				if(hsv[0].at<byte>(K,L)==color1){
					hsv[0].at<byte>(K,L)= 100;
					centr1_x += L;
					centr1_y += K;	
					ncolor1++;
				}
				else if(hsv[0].at<byte>(K,L)==color2){
					hsv[0].at<byte>(K,L)= 200;
					centr2_x += L;
					centr2_y += K;
					ncolor2++;
				}
				else
					hsv[0].at<byte>(K,L)= 0;
			}
		
		centr1_x/=(double)ncolor1;
		centr1_y/=(double)ncolor1;
		centr2_x/=(double)ncolor2;
		centr2_y/=(double)ncolor2;

		double lejando1_x=0;
		double lejando1_y=0;
		double lejando2_x=0;
		double lejando2_y=0;

		double distmax1=0;
		double distmax2=0;

		for(size_t K=0; K<hsv[0].rows; ++K)
			for(size_t L=0; L<hsv[0].cols; ++L){
				if(hsv[0].at<byte>(K,L)==100){
					double d = dist(centr1_x,centr1_y,L,K);
					if(distmax1<d){
						distmax1 = d;
						lejando1_x=L;
						lejando1_y=K;
					}	
				}
				else if(hsv[0].at<byte>(K,L)==200){
					double d = dist(centr2_x,centr2_y,L,K);
					if(distmax2<d){
						distmax2 = d;
						lejando2_x=L;
						lejando2_y=K;
					}
				}
			}
		//std::cout<<centr1_x<<"  "<<centr1_y<<"  "<<distmax1<<"  "<<std::endl;
		cv::circle(hsv[0],cv::Point((int)centr1_x,(int)centr1_y),(int)distmax1,cv::Scalar::all(255),1);
		cv::circle(hsv[0],cv::Point((int)centr2_x,(int)centr2_y),(int)distmax2,cv::Scalar::all(255),1);
			
		double c1 = distmax1*distmax1*3.1416;
		double c2 = distmax2*distmax2*3.1416;

		double razon1 = n1/c1;
		double razon2 = n2/c2;

		double indicex = 0;
		double indicey = 0;
		//std::cout<<"r1: "<<razon1<<" "<<"r2: "<<razon2<<std::endl;
		if(razon2<razon1){ 
			for(size_t K=0; K<hsv[0].rows; ++K)
			for(size_t L=0; L<hsv[0].cols; ++L){
				if(hsv[0].at<byte>(K,L)==100){
					hsv[0].at<byte>(K,L)=0;
				}
		
			}
			indicex = lejando2_x;
			indicey = lejando2_y;
		}
		else{
			for(size_t K=0; K<hsv[0].rows; ++K)
			for(size_t L=0; L<hsv[0].cols; ++L){
				if(hsv[0].at<byte>(K,L)==200){
					hsv[0].at<byte>(K,L)=0;
				}
			}
			indicex = lejando1_x;
			indicey = lejando1_y;
		}					
		//enmascarar los otros canales
		for(size_t K=1; K<hsv.size(); ++K)
			for(size_t L=0; L<hsv[K].rows; ++L)
				for(size_t M=0; M<hsv[K].cols; ++M)
					hsv[K].at<byte>(L,M) *= (hsv[0].at<byte>(L,M)>0);

		//cv::cvSetAt(frame2, cv::cvScalar( 1, 1, 1, 0),indicey,indicex);
		//frame2.at<unsigned char>(indicey,indicex)= 255;
		cv::circle(frame2,cv::Point((int)indicex,(int)indicey),(int)5,cv::Scalar::all(255),-1);
		cv::imshow("Dibujo", frame2);
		frame2*= 0.8;

		for(size_t K=0; K<hsv.size(); ++K)
			cv::imshow(windows[K], hsv[K]);

		cv::cvtColor(frame, frame, CV_HSV2BGR);

		cv::imshow("alpha", frame);

		if(cv::waitKey(10)>=0) break;
	}
	return 0;
}


