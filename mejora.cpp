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

set<pair <double,double> > obtener_punteros(cv::Mat &hsv,vector<double> razones,vector<size_t> &colores,vector< pair<double,double> >lejanos,vector<double> &areas){
		set<pair <double,double> >punteros;
		vector<double>::iterator q = max_element(razones.begin(),razones.end());
		vector<double>::iterator q2 = max_element(areas.begin(),areas.end());
		double aux = *q2;
		*q2 = 0.00;
		vector<double>::iterator q3 = max_element(areas.begin(),areas.end());
		*q2 = aux;
		for(size_t K=0; K<hsv.rows; ++K)
			for(size_t L=0; L<hsv.cols; ++L){
				vector<size_t>::iterator p = find(colores.begin(),colores.end(),hsv.at<byte>(K,L));
				if( p!= colores.end()){
					if(areas[p-colores.begin()]/ (*q3)<0.25 || razones[p-colores.begin()] == *q ){
						hsv.at<byte>(K,L)= 0;
											
					}
					else{
						punteros.insert(lejanos[p-colores.begin()]);
					}
				}
		
			}

		return punteros;
}

cv::Mat submuestreo(const cv::Mat &image,int scale){
	cv::Mat result(image.rows/scale, image.cols/scale, image.type() );
	for(size_t K=0;K<result.rows; K++)
		for(size_t L=0;L<result.cols; L++)
			result.at<byte>(K,L) = image.at<byte>(scale*K,scale*L);

	return result;
}

int main(int argc, char **argv){
	cv::VideoCapture capture(0);
	cv::Mat frame,framergb,fondo;

	std::string windows[] = {"hue", "saturation", "value"};
	for(size_t K=0; K<1; ++K)
		cv::namedWindow(windows[K], CV_WINDOW_KEEPRATIO);	

	int value[3] = {7,117,65};
	int alpha[3] = {6,55,35},umbral=30;

	cv::namedWindow("mask", CV_WINDOW_KEEPRATIO);
	cv::namedWindow("fondo", CV_WINDOW_KEEPRATIO);

	cv::createTrackbar("umbral", "mask", &umbral, 255, NULL, NULL);

	cv::namedWindow("original", CV_WINDOW_KEEPRATIO);	
	cv::namedWindow("resultado", CV_WINDOW_KEEPRATIO);	
	for(size_t K=0; K<1; ++K)
		cv::createTrackbar(windows[K],windows[K], value+K, 255, NULL, NULL);

	while(true){
		capture>>fondo;
		cv::imshow("fondo", fondo);
		if( cv::waitKey(30) != -1)
			break;
	}

	fondo=fondo.clone();
	cv::flip(fondo, fondo, 1);
	std::vector<cv::Mat> rgbfondo;
	cv::split(fondo,rgbfondo);
	cv::imshow("fondo", fondo);
	while( true ){
		capture>>frame;			
		cv::flip(frame, frame, 1);
		framergb = frame;
		std::vector<cv::Mat> rgb;
		cv::split(framergb,rgb);

		cv::Mat resta_rojo, resta_verde, resta_azul;

		cv::threshold(cv::abs(rgb[0]-rgbfondo[0]),resta_rojo,umbral,255,cv::THRESH_BINARY);
		cv::threshold(cv::abs(rgb[1]-rgbfondo[1]),resta_verde,umbral,255,cv::THRESH_BINARY); 
		cv::threshold(cv::abs(rgb[2]-rgbfondo[2]),resta_azul,umbral,255,cv::THRESH_BINARY);
		
		cv::Mat mask_diff = resta_rojo bitor resta_verde bitor resta_azul;
		cv::imshow("mask", mask_diff);

		std::vector<cv::Mat> hsv;
		cv::cvtColor(frame, frame, CV_BGR2HSV);
		
		#if(1)
		//quitar ruido impulsivo
		cv::medianBlur(frame, frame, 5); 
		#endif
		
		//separa en 3 canales 
		cv::split(frame, hsv);
		
		#if(0)	
		//cuantizacion
		for(int K = 0;K<3;K++)
			hsv[K] = submuestreo(hsv[K],4);
		
		#endif

		//umbralizacion
		cv::Scalar hsv_min(value[0]-alpha[0], value[1]-alpha[1], value[2]-alpha[2], 0);
		cv::Scalar hsv_max(value[0]+alpha[0], value[1]+alpha[1], value[2]+alpha[2], 0);
		for(size_t K=0; K<hsv.size(); ++K)
			cv::inRange(hsv[K], cv::Scalar::all(hsv_min[K]), cv::Scalar::all(hsv_max[K]),hsv[K]);

		#if(1)	
		//eliminar huecos internos
		cv::dilate(hsv[0], hsv[0], cv::Mat::ones(5,5,CV_8U));
		cv::dilate(hsv[1], hsv[1], cv::Mat::ones(5,5,CV_8U));	
		#endif
		
		#if(1)	
		//recupera forma
		cv::erode(hsv[0], hsv[0], cv::Mat::ones(5,5,CV_8U));
		cv::erode(hsv[1], hsv[1], cv::Mat::ones(5,5,CV_8U));	
		#endif

		//bitwise_and(hsv[0],hsv[0],mask_hue);
		cv::Mat mask_hue = hsv[0];

		#if(1)
		//eliminar huecos internos
		cv::dilate(mask_hue, mask_hue, cv::Mat::ones(5,5,CV_8U));
		#endif

		#if(1)	
		//recupera forma
		cv::erode(mask_hue, mask_hue, cv::Mat::ones(5,5,CV_8U));
		
		#endif
		//threshold(mask_hue,mask_hue,0,255,cv::THRESH_BINARY); 
		dilate(mask_hue,mask_hue,cv::Mat::ones(3,3,CV_8U));		
		dilate(mask_diff,mask_diff,cv::Mat::ones(3,3,CV_8U));

		//bitwise_and(mask_hue,mask_diff,mask_hue);
		cv::Mat mask_total = mask_hue bitand mask_diff;

		cv::floodFill(mask_total, cv::Point(0,0), 128);
		
		//Sobrevive lo que no es gris
		{
			cv::Mat aux;
			threshold(mask_total,aux,100,255,cv::THRESH_BINARY); 	
			threshold(mask_total,mask_total,150,255,cv::THRESH_BINARY_INV); 	
			bitwise_and(aux,mask_total,mask_total);
			bitwise_not(mask_total,mask_total);
		}

		cv::erode(mask_total, mask_total, cv::Mat::ones(5,5,CV_8U));
		cv::dilate(mask_total, mask_total, cv::Mat::ones(5,5,CV_8U));

		#if(1)
		//identificacion de regiones
		byte color = 50;
		for(size_t K=0; K<mask_total.rows; ++K)
			for(size_t L=0; L<mask_total.cols; ++L){
				if(mask_total.at<byte>(K,L)==255){
					cv::floodFill(mask_total, cv::Point(L,K), color);
					color += 10;
				}
			}

		//Sobreviven los mas grandes	
		cv::Mat freq = histogram(mask_total);
		*freq.begin<float>() = 0;
		vector<size_t> colores; 
		vector<double> areas= obtener_areas(freq,colores,3);

		//obtiene los centroides
		vector< pair<double,double> >centroides = obtener_centroides(mask_total,colores);
		
		//obtiene los radios y los dedos indices	
		vector<double>radios;
		vector< pair<double,double> >lejanos = obtener_lejanos(mask_total,colores,centroides,radios);		
		
		//obtiene las razones		
		vector<double>razones = obtener_razones(areas,radios);
		
		//elimina los objetos que no interesan y obtiene los punteros finales	
		set< pair<double,double> > punteros = obtener_punteros(mask_total,razones,colores,lejanos,areas);
		
		//dibuja los punteros
		set< pair < double,double > >::iterator p = punteros.begin();
		while(p!= punteros.end()){
			cv::circle(mask_total,cv::Point((int)(*p).first,(int)(*p).second),5,cv::Scalar::all(255),-1);
			p++;	
		}
		#endif


		//muestra los 3 canales 
		for(size_t K=0; K<1; ++K)
			cv::imshow(windows[K], hsv[K]);
		
		//muestra el resultado
		cv::imshow("resultado", mask_total);		
		
		//muestra el original			
		cv::cvtColor(frame, frame, CV_HSV2BGR);

		cv::imshow("original", frame);


		if(cv::waitKey(10)>=0) break;



	}
	return 0;
}


