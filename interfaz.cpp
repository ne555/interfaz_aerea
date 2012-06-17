#include <opencv2/opencv.hpp>
#include <vector>
typedef unsigned char byte;

int main(int argc, char **argv){
	cv::VideoCapture capture(0);
	if(not capture.isOpened()) return 1;
	cv::Mat frame, interfaz;//, mask;
	interfaz = cv::Mat::zeros(capture.get(CV_CAP_PROP_FRAME_HEIGHT),capture.get(CV_CAP_PROP_FRAME_WIDTH),CV_8UC3);
	cv::Scalar blanco(255, 255, 255), negro (0, 0, 0), rojo  (255, 0, 0), verde (0, 255, 0), azul  (0, 0, 255);
	
	std::vector< cv::Scalar > vectorcolor (5);
	
	vectorcolor[0] = blanco;
	vectorcolor[1] = negro;
	vectorcolor[2] = rojo;
	vectorcolor[3] = azul;
	vectorcolor[4] = verde;
	//int p[]= {capture.get(CV_CAP_PROP_FRAME_HEIGHT),capture.get(CV_CAP_PROP_FRAME_WIDTH),3};
	//mask = cv::Mat::ones(3,&p[0],CV_8UC3);
	
	int a = 70, b =70;
	for(int i=0;i<5;++i){
		cv::Point p1(0,i*a);
		cv::Point p2(b,(i+1)*a);
		rectangle(interfaz,p1,p2,vectorcolor[i],CV_FILLED);	
		//rectangle(mask,p1,p2,cv::Scalar::all(0),CV_FILLED);	
	}

	cv::Rect mask(0,0, b, 5*a);

	cv::namedWindow("interfaz", CV_WINDOW_KEEPRATIO);
	while(true){
		capture>>frame;	
		
		cv::Mat roi = cv::Mat(frame, mask);
		cv::Mat(interfaz, mask).copyTo(roi);
		//frame = interfaz + frame.mul(mask);
		cv::flip(frame, frame, 1);
		
		cv::imshow("interfaz",frame);
		
		if(cv::waitKey(30)>=0) break;
	}
	return 0;
}
