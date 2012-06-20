#include <opencv2/opencv.hpp>
#include <set>
#include <vector>
#include <string>
#include "aux.h"
#include <algorithm>
#include <utility>

using namespace std;

//los datos de centroide, lejanos, radio, razones estan ordenados segun el de mayor area;

double dist(double x,double y,int x0,int y0){
	return (double) sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0));
}

vector<double> obtener_areas(cv::Mat &freq,vector<size_t> &colores,int c){
	vector<double> r;
	cv::MatIterator_<float> max;
	for(int k=0;k<c;k++){
		max= std::max_element( freq.begin<float>(), freq.end<float>() );
		r.push_back( *max );
		*max= 0;
		size_t color = std::distance( freq.begin<float>(), max );
		colores.push_back( color );
	}
	return r;			
}

vector< pair<double,double> > obtener_centroides(cv::Mat &hsv,vector<size_t> &colores){
		vector<pair <double,double> >centroides(colores.size(),make_pair(0.0,0.0));
		vector<double>acum(colores.size(),0.0);
		for(size_t K=0; K<hsv.rows; ++K)
			for(size_t L=0; L<hsv.cols; ++L){
				vector<size_t>::iterator p = find(colores.begin(),colores.end(),hsv.at<byte>(K,L));
				if( p!= colores.end()){
					centroides[p-colores.begin()].first += L;
					centroides[p-colores.begin()].second+= K;
					acum[p-colores.begin()]++;	
				}
				else
					hsv.at<byte>(K,L)= 0;
			}
		for(int k=0;k<centroides.size();k++){
			centroides[k].first /=acum[k];
			centroides[k].second/=acum[k];  	
		}
		return centroides;
}

vector< pair<double,double> > obtener_lejanos(cv::Mat &hsv,vector<size_t> &colores,vector< pair <double,double> > &centroides,vector<double> &radios){
		vector<pair <double,double> >lejanos(colores.size());
		radios.clear();
		radios.resize(colores.size(),-1.0);
		for(size_t K=0; K<hsv.rows; ++K)
			for(size_t L=0; L<hsv.cols; ++L){
				vector<size_t>::iterator p = find(colores.begin(),colores.end(),hsv.at<byte>(K,L));
				if( p!= colores.end()){
					double d = dist(L,K,centroides[p-colores.begin()].first,centroides[p-colores.begin()].second);
					if(d>radios[p-colores.begin()])	{
						radios[p-colores.begin()] = d;
						lejanos[p-colores.begin()].first = L;
						lejanos[p-colores.begin()].second = K;									
					}
				}
			}
		
		return lejanos;
}

vector<double> obtener_razones(vector<double> &areas,vector<double> & radios){
	vector<double>razones(areas.size(),0.0);
	for(int k =0;k<razones.size();k++){
		razones[k]= areas[k]/(3.1416*radios[k]*radios[k]);	
	}
	return razones;	
}

set<pair <double,double> > obtener_punteros(cv::Mat &hsv,vector<double> razones,vector<size_t> &colores,vector< pair<double,double> >lejanos){
		set<pair <double,double> >punteros;
		vector<double>::iterator q = max_element(razones.begin(),razones.end());	
		for(size_t K=0; K<hsv.rows; ++K)
			for(size_t L=0; L<hsv.cols; ++L){
				vector<size_t>::iterator p = find(colores.begin(),colores.end(),hsv.at<byte>(K,L));
				if( p!= colores.end()){
					if(razones[p-colores.begin()] == *q){
						hsv.at<byte>(K,L)= 0;
											
					}
					else{
						punteros.insert(lejanos[p-colores.begin()]);
					}
				}
		
			}

		return punteros;
}



int main(int argc, char **argv){
	cv::VideoCapture capture(0);
	cv::Mat frame, framergb,fondo, 	img1, img2, img3, img4, img5;
	
	std::string windows[] = {"hue", "saturation", "value"};
	for(size_t K=0; K<3; ++K)
		cv::namedWindow(windows[K], CV_WINDOW_KEEPRATIO);
	
	//cv::namedWindow("Dibujo", CV_WINDOW_AUTOSIZE);	

	int value[3] = {8};
	int alpha = 7, umbral=10;
	cv::namedWindow("alpha", CV_WINDOW_KEEPRATIO);
	cv::namedWindow("mask", CV_WINDOW_KEEPRATIO);
	cv::namedWindow("huee", CV_WINDOW_KEEPRATIO);
	cv::namedWindow("resultado", CV_WINDOW_KEEPRATIO);
	cv::namedWindow("fondo", CV_WINDOW_KEEPRATIO);
	cv::createTrackbar("alpha","alpha", &alpha, 255, NULL, NULL);
	cv::createTrackbar("umbral", "mask", &umbral, 255, NULL, NULL);

	for(size_t K=0; K<3; ++K)
		cv::createTrackbar(windows[K],windows[K], value+K, 255, NULL, NULL);
	
	cv::Mat dibujo=cv::Mat::zeros(480,640,CV_8U);
	
	cv::waitKey(4000);

	capture>>fondo;
	cv::flip(fondo, fondo, 1);	
	fondo = fondo.clone();

	cv::imshow("fondo", fondo);
	//int umb=0;
	//cv::createTrackbar("umbral","hue", &umb, 255, NULL, NULL);
	while( true ){
		capture>>frame;
		cv::flip(frame, frame, 1);	
		framergb = frame;
		std::vector<cv::Mat> rgb;
		std::vector<cv::Mat> rgbfondo;
		cv::split(framergb,rgb);
		cv::split(fondo,rgbfondo);
		
		cv::threshold(cv::abs(rgb[0]-rgbfondo[0]),img3,umbral,255,cv::THRESH_BINARY);
		cv::threshold(cv::abs(rgb[1]-rgbfondo[1]),img4,umbral,255,cv::THRESH_BINARY); 
		cv::threshold(cv::abs(rgb[2]-rgbfondo[2]),img5,umbral,255,cv::THRESH_BINARY);
		
		bitwise_or(img3,img4,img2);	
		bitwise_or(img2,img5,img2);

		cv::imshow("mask", img2);
		
		std::vector<cv::Mat> hsv;
		cv::cvtColor(frame, frame, CV_BGR2HSV);
		
		cv::medianBlur(frame, frame, 5); //quitar ruido impulsivo
		cv::split(frame, hsv);

		//umbralizacion
		cv::Scalar hsv_min = cv::Scalar::all(value[0]-alpha);
		cv::Scalar hsv_max = cv::Scalar::all(value[0]+alpha);
		cv::inRange(hsv[0], cv::Scalar::all(hsv_min[0]), cv::Scalar::all(hsv_max[0]), hsv[0]);
		
		threshold(hsv[0],img1,0,255,cv::THRESH_BINARY); 
		
		cv::imshow("huee",hsv[0]);
		
		
		dilate(img1,img1,cv::Mat::ones(3,3,CV_8U));		
		dilate(img2,img2,cv::Mat::ones(3,3,CV_8U));

		bitwise_and(img1,img2,img1);
			
		cv::imshow("resultado",img1);
		
		//eliminar huecos internos
		cv::medianBlur(hsv[0], hsv[0], 5); 
		//cv::dilate(hsv[0], hsv[0], cv::Mat::ones(3,3,CV_8U));

		#if 0
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
		#if(1)		
		cv::Mat freq = histogram(hsv[0]);
		*freq.begin<float>() = 0;
		vector<size_t> colores; 
		vector<double> areas= obtener_areas(freq,colores,3);

		/*for(int k=0;k<areas.size();k++){
			std::cout<<areas[k]<<endl;		
		}
		std::cin.get();*/
		vector< pair<double,double> >centroides = obtener_centroides(hsv[0],colores);
		
		vector<double>radios;
		vector< pair<double,double> >lejanos = obtener_lejanos(hsv[0],colores,centroides,radios);		
	
		//cv::circle(hsv[0],cv::Point((int)centr1_x,(int)centr1_y),(int)distmax1,cv::Scalar::all(255),1);
		//cv::circle(hsv[0],cv::Point((int)centr2_x,(int)centr2_y),(int)distmax2,cv::Scalar::all(255),1);
			
		vector<double>razones = obtener_razones(areas,radios);
		
		
		//std::cout<<razones[0]<<"    "<<razones[1]<<"    "<<razones[2]<<endl;		
		

		
		//double umbral = (double)umb/255.00;
		set< pair<double,double> > punteros = obtener_punteros(hsv[0],razones,colores,lejanos);

					
		//enmascarar los otros canales
		for(size_t K=1; K<hsv.size(); ++K)
			for(size_t L=0; L<hsv[K].rows; ++L)
				for(size_t M=0; M<hsv[K].cols; ++M)
					hsv[K].at<byte>(L,M) *= (hsv[0].at<byte>(L,M)>0);
		
		set< pair < double,double > >::iterator p = punteros.begin();
		while(p!= punteros.end()){
			cv::circle(hsv[0],cv::Point((int)(*p).first,(int)(*p).second),5,cv::Scalar::all(255),-1);
			p++;	
		}
		#endif
		//cv::imshow("Dibujo", frame2);
		//frame2*= 0.8;
		#endif

		for(size_t K=0; K<hsv.size(); ++K)
			cv::imshow(windows[K], hsv[K]);

		cv::cvtColor(frame, frame, CV_HSV2BGR);

		cv::imshow("alpha", frame);

		if(cv::waitKey(10)>=0) break;
	}
	return 0;
}


