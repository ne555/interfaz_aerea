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

cv::Mat submuestreo(const cv::Mat &image, int scale);

int main(int argc, char **argv){
	cv::VideoCapture capture(0);
	cv::Mat frame,img;

	std::string windows[] = {"hue", "saturation", "value"};
	for(size_t K=0; K<3; ++K)
		cv::namedWindow(windows[K], CV_WINDOW_KEEPRATIO);	

	int value[3] = {7,117,65};
	int alpha[3] = {6,55,35};
	cv::namedWindow("parametros", CV_WINDOW_KEEPRATIO);
	cv::createTrackbar("alpha0","parametros", &alpha[0], 255, NULL, NULL);
	cv::createTrackbar("alpha1","parametros", &alpha[1], 255, NULL, NULL);
	cv::createTrackbar("alpha2","parametros", &alpha[2], 255, NULL, NULL);
	
<<<<<<< HEAD
	cv::namedWindow("Dibujo", CV_WINDOW_KEEPRATIO);	

	int value[3] = {8};
	int alpha = 7;
	cv::namedWindow("alpha", CV_WINDOW_KEEPRATIO);
	cv::createTrackbar("alpha","alpha", &alpha, 255, NULL, NULL);

	for(size_t K=0; K<3; ++K)
		cv::createTrackbar(windows[K],windows[K], value+K, 255, NULL, NULL);
	cv::Mat frame2=cv::Mat::zeros(480/4,640/4,CV_8U);
=======
	cv::namedWindow("original", CV_WINDOW_KEEPRATIO);	
	cv::namedWindow("resultado", CV_WINDOW_KEEPRATIO);	
	for(size_t K=0; K<3; ++K)
		cv::createTrackbar(windows[K],windows[K], value+K, 255, NULL, NULL);

>>>>>>> fede/laposta
	while( true ){
		capture>>frame;			
		cv::flip(frame, frame, 1);
		std::vector<cv::Mat> hsv;
		cv::cvtColor(frame, frame, CV_BGR2HSV);
		
		#if(1)
		//quitar ruido impulsivo
		cv::medianBlur(frame, frame, 5); 
		#endif
		
		//separa en 3 canales 
		cv::split(frame, hsv);
<<<<<<< HEAD
		for(size_t K=0; K<hsv.size(); ++K)
			hsv[K] = submuestreo(hsv[K], 4);

		//umbralizacion
		cv::Scalar hsv_min = cv::Scalar::all(value[0]-alpha);
		cv::Scalar hsv_max = cv::Scalar::all(value[0]+alpha);
		//hsv[0] = submuestreo(hsv[0], 2);
		cv::inRange(hsv[0], cv::Scalar::all(hsv_min[0]), cv::Scalar::all(hsv_max[0]), hsv[0]);

		//eliminar huecos internos
		cv::medianBlur(hsv[0], hsv[0], 5); 
		cv::GaussianBlur(hsv[0], hsv[0], cv::Size(5,5),0); 
		hsv[0] = hsv[0].mul( hsv[0]>128 );

		cv::Mat max_min;
		cv::dilate(hsv[0], hsv[0], cv::Mat::ones(5,5,CV_8U));
		//cv::erode(hsv[0], max_min, cv::Mat::ones(5,5,CV_8U));
=======
		
		#if(0)	
		//cuantizacion
		for(int K = 0;K<3;K++)
			hsv[K] = submuestreo(hsv[K],4);
		
		#endif

		//umbralizacion
		cv::Scalar hsv_min(value[0]-alpha[0], value[1]-alpha[1], value[2]-alpha[2], 0);
		cv::Scalar hsv_max(value[0]+alpha[0], value[1]+alpha[1], value[2]+alpha[2], 0);
		for(size_t K=0; K<hsv.size(); ++K){
			cv::inRange(hsv[K], cv::Scalar::all(hsv_min[K]), cv::Scalar::all(hsv_max[K]),hsv[K]);
		}
		
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

		bitwise_and(hsv[0],hsv[1],img);
		
		#if(1)
		//eliminar huecos internos
		cv::dilate(img, img, cv::Mat::ones(5,5,CV_8U));
		#endif

		#if(1)	
		//recupera forma
		cv::erode(img, img, cv::Mat::ones(5,5,CV_8U));
		
		#endif
>>>>>>> fede/laposta

		//hsv[0] -= max_min;
		//cv::dilate(hsv[1], hsv[1], cv::Mat::ones(5,5,CV_8U));

<<<<<<< HEAD
		#if 1
		//identificacion de regiones
		byte color = 50;
		for(size_t K=0; K<hsv[0].rows; ++K)
			for(size_t L=0; L<hsv[0].cols; ++L)
				if(hsv[0].at<byte>(K,L)==255){
					cv::floodFill(hsv[0], cv::Point(L,K), color);
=======
		#if(1)
		//identificacion de regiones
		byte color = 50;
		for(size_t K=0; K<img.rows; ++K)
			for(size_t L=0; L<img.cols; ++L){
				if(img.at<byte>(K,L)==255){
					cv::floodFill(img, cv::Point(L,K), color);
>>>>>>> fede/laposta
					color += 10;
				}

		//Sobreviven los mas grandes	
		cv::Mat freq = histogram(img);
		*freq.begin<float>() = 0;
		vector<size_t> colores; 
		vector<double> areas= obtener_areas(freq,colores,3);

		/*for(int k=0;k<areas.size();k++){
			std::cout<<areas[k]<<endl;		
		}
		*/
		
		//obtiene los centroides
		vector< pair<double,double> >centroides = obtener_centroides(img,colores);
		
		//obtiene los radios y los dedos indices	
		vector<double>radios;
		vector< pair<double,double> >lejanos = obtener_lejanos(img,colores,centroides,radios);		
		
		//obtiene las razones		
		vector<double>razones = obtener_razones(areas,radios);
		
		//elimina los objetos que no interesan y obtiene los punteros finales	
		set< pair<double,double> > punteros = obtener_punteros(img,razones,colores,lejanos,areas);
		
		//dibuja los punteros
		set< pair < double,double > >::iterator p = punteros.begin();
		while(p!= punteros.end()){
			cv::circle(img,cv::Point((int)(*p).first,(int)(*p).second),5,cv::Scalar::all(255),-1);
			p++;	
		}
<<<<<<< HEAD
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
=======
>>>>>>> fede/laposta
		#endif

		//muestra los 3 canales 
		for(size_t K=0; K<hsv.size(); ++K)
			cv::imshow(windows[K], hsv[K]);
		
		//muestra el resultado
		cv::imshow("resultado", img);		
		
		//muestra el original			
		cv::cvtColor(frame, frame, CV_HSV2BGR);
		cv::imshow("original", frame);

		if(cv::waitKey(10)>=0) break;
	}
	return 0;
}

cv::Mat submuestreo(const cv::Mat &image, int scale){
	cv::Mat result( image.rows/scale, image.cols/scale, image.type() );
	for(size_t K=0; K<result.rows; ++K)
		for(size_t L=0; L<result.cols; ++L)
			result.at<byte>(K,L) = image.at<byte>(scale*K,scale*L);

	return result;
}

