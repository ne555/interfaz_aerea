#include <fstream>
#include "aux.h"

typedef unsigned char byte;

namespace{
	cv::Mat polar_combine(const cv::Mat &magnitud, const cv::Mat &phase);
	double distance2( int x1, int y1, int x2, int y2 );
	template <class T>
	inline T square(T x){
		return x*x;
	}
}

void print_info(const cv::Mat &image, std::ostream &out){
	out << "Characteristics\n";
	out << "Size " << image.rows << 'x' << image.cols << '\n';
	out << "Channels " << image.channels() << '\n';
	out << "Depth ";
	switch(image.depth()){
		case CV_8U: out << "8-bit unsigned integers ( 0..255 )\n"; break;
		case CV_8S: out << "8-bit signed integers ( -128..127 )\n"; break;
		case CV_16U: out << "16-bit unsigned integers ( 0..65535 )\n"; break;
		case CV_16S: out << "16-bit signed integers ( -32768..32767 )\n"; break;
		case CV_32S: out << "32-bit signed integers ( -2147483648..2147483647 )\n"; break;
		case CV_32F: out << "32-bit floating-point numbers ( -FLT_MAX..FLT_MAX, INF, NAN )\n"; break;
		case CV_64F: out << "64-bit floating-point numbers ( -DBL_MAX..DBL_MAX, INF, NAN )\n"; break;
	}
}

void print(const cv::Mat &image){
	for(size_t K=0; K<image.rows; ++K){
		for(size_t L=0; L<image.cols; ++L)
			std::cerr << image.at<float>(K,L) << ' ';
		std::cerr << '\n';
	}
}

cv::Mat equal_mosaic( const std::vector<cv::Mat> &images, size_t r, size_t c){
	if(images.empty()) return cv::Mat();

	size_t rows = images[0].rows, cols = images[0].cols;
	cv::Mat big = cv::Mat::zeros(r*rows, c*cols, images[0].type());

	for(size_t K=0; K<images.size(); ++K){
		cv::Rect submatrix ( (K%c)*cols, (K/c)*rows, images[K].cols, images[K].rows );
		cv::Mat region = cv::Mat(big, submatrix);
		images[K].copyTo(region);
	}

	return big;
}

cv::Mat mosaic( const cv::Mat &a, const cv::Mat &b, bool vertical ){
	std::vector< cv::Mat > images;
	images.push_back(a);
	images.push_back(b);

	if(vertical)
		return equal_mosaic(images, 2, 1);
	else 
		return equal_mosaic(images, 1, 2);
}

cv::Mat histogram(const cv::Mat &image){
	const int channels = 0;
	int size = 256;
	float range[] = {0, 256};
	const float *ranges[] = {range};

	cv::MatND hist;
	cv::calcHist(&image, 1, &channels, cv::Mat(), hist, 1, &size, ranges);
	return hist;
}

cv::Mat draw_histogram(const cv::Mat &image){
	size_t rows = image.rows, cols = image.cols;
	cv::Mat hist = histogram(image);
	cv::normalize(hist, hist, 0, 1, CV_MINMAX);
	cv::Mat hist_image = cv::Mat::zeros(rows, cols, CV_8U);


	for(size_t K=0; K<hist.rows-1; ++K){
		float value = hist.at<float>(K);
		float next = hist.at<float>(K+1);

		cv::Point p1(K, rows);
		cv::Point p2(K+1, rows);
		cv::Point p3(K+1, rows-next*rows);
		cv::Point p4(K, rows-value*rows);

		cv::Point p[] = {p1, p2, p3, p4, p1};

		cv::fillConvexPoly(hist_image, p, 5, cv::Scalar::all(255));
		//cv::line(hist_image, p3, p4, cv::Scalar::all(255));
	}

	return hist_image;
}

cv::Mat convolve(const cv::Mat &image, const cv::Mat &kernel){
	cv::Mat result = cv::Mat::zeros(image.rows, image.cols, CV_32F);
	//same bits as the image, kernel centered, no offset
	cv::filter2D(image, result, -1, kernel, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);
	return result;
}

cv::Mat local_equalization(const cv::Mat &original, size_t size){
	size_t rows = original.rows, cols = original.cols;
	cv::Mat local = cv::Mat::zeros(rows, cols, CV_8U);

	int half = size/2;
	for(size_t K=half; K<rows-half; ++K)
		for(size_t L=half; L<cols-half; ++L){
			cv::Mat roi = cv::Mat( original, cv::Rect(L-half,K-half,size,size) );
			cv::Mat equalized;
			cv::equalizeHist(roi, equalized);

			local.at<byte>(K,L) = equalized.at<byte>(half, half);
		}

	return local;
}

std::vector<buffer> cargar_paleta(const char *filename){
	std::ifstream input(filename);
	const size_t size = 256;

	std::vector<buffer> rgb(3);
	for(size_t K=0; K<size; ++K)
		for(size_t L=0; L<3; ++L){
			float color;
			input>>color;
			rgb[L][K] = 0xff*color;
		}

	return rgb;
}

cv::Mat hacer_compleja(const cv::Mat &image){
	size_t 
		rows = cv::getOptimalDFTSize(image.rows),
		cols = cv::getOptimalDFTSize(image.cols);

	cv::Mat result;
	cv::copyMakeBorder(image, result, 0, rows-image.rows, 0, cols-image.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

	cv::Mat 
		planes[] = {cv::Mat_<float>(result), cv::Mat::zeros(result.size(), CV_32F)};

	cv::merge(planes, 2, result);

	return result;
}

cv::Mat fourier_transform(const cv::Mat &image){
	cv::Mat transform(image.size(), image.type());
	dft(image, transform);
	return transform;
}

void swap(cv::Mat &a, cv::Mat &b){
	cv::Mat temp;
	a.copyTo(temp);
	b.copyTo(a);
	temp.copyTo(b);
}

void center(cv::Mat &image){
	int cx = image.cols/2, cy = image.rows/2;
	cv::Mat 
		q0 = cv::Mat(image, cv::Rect(0,0,cx,cy)), //TL
		q1 = cv::Mat(image, cv::Rect(cx,0,cx,cy)), //TR
		q2 = cv::Mat(image, cv::Rect(0,cy,cx,cy)), //BL
		q3 = cv::Mat(image, cv::Rect(cx,cy,cx,cy)); //BR
	//swap quadrants
	::swap(q0,q3);
	::swap(q1,q2);
}

cv::Mat magnitude(const cv::Mat &image){
	cv::Mat planes[2];
	cv::split(image, planes);

	cv::Mat result;
	cv::magnitude(planes[0], planes[1], result);
	return result;
}

cv::Mat magnitud_log(const cv::Mat &image){
	cv::Mat magnitud = magnitude(image);
	cv::log(magnitud, magnitud);
	cv::normalize(magnitud, magnitud, 0, 1, CV_MINMAX);

	return magnitud;
}

cv::Mat phase(const cv::Mat &image){
	cv::Mat phase, planes[2];
	cv::split(image, planes);
	cv::phase(planes[0], planes[1], phase);
	
	return phase;
}

cv::Mat real(const cv::Mat &image){
	cv::Mat planes[2];
	cv::split(image, planes);
	return planes[0];
}

cv::Mat ver_espectro(const cv::Mat &image){
	cv::Mat espectro = fourier_transform( hacer_compleja(image) );
	cv::Mat magnitud = magnitud_log(espectro);

	center(magnitud);
	cv::normalize(magnitud, magnitud, 0, 1, CV_MINMAX);
	return magnitud;
}

cv::Mat filtro_ideal(size_t rows, size_t cols, double corte){
	cv::Mat 
		magnitud = cv::Mat::zeros(rows, cols, CV_32F);
	cv::circle(
		magnitud,
		cv::Point(rows/2,cols/2),
		rows*corte,
		cv::Scalar::all(1),
		-1
	);

	center(magnitud);
	return magnitud;
}
cv::Mat filtro_butterworth(size_t rows, size_t cols, double corte, size_t order){
	cv::Mat 
		magnitud = cv::Mat::zeros(rows, cols, CV_32F);

	corte *= rows;
	corte *= corte;
	for(size_t K=0; K<rows; ++K)
		for(size_t L=0; L<cols; ++L){
			double distance = distance2(K,L,rows/2,cols/2);
			magnitud.at<float>(K,L) = 1.0/(1 + std::pow(distance/corte, order) );
		}

	center(magnitud);
	return magnitud;
}

cv::Mat filtro_gaussian(size_t rows, size_t cols, double corte){
	cv::Mat 
		magnitud = cv::Mat::zeros(rows, cols, CV_32F);

	corte *= rows;
	corte *= corte;
	for(size_t K=0; K<rows; ++K)
		for(size_t L=0; L<cols; ++L){
			double distance = distance2(K,L,rows/2,cols/2);
			magnitud.at<float>(K,L) = std::exp(-distance/(2*corte));
		}

	center(magnitud);
	return magnitud;

}

cv::Mat filtro_en_frecuencia(const cv::Mat &image, const cv::Mat &filter_magnitude){
	cv::Mat espectro = fourier_transform( hacer_compleja(image) );
	cv::Mat phase = cv::Mat::zeros(espectro.size(), CV_32F);
	
	cv::Mat filter = polar_combine(filter_magnitude, phase);
	cv::Mat result;
	cv::mulSpectrums(espectro, filter, result, 0);

	result = real( fourier_transform(result) );

	cv::normalize(result, result, 0, 1, CV_MINMAX);
	cv::flip(result,result,-1);
	return result;
}

namespace{
	cv::Mat polar_combine(const cv::Mat &magnitud, const cv::Mat &phase){
		cv::Mat x[2],result;
		cv::polarToCart(magnitud, phase, x[0], x[1]);
		cv::merge(x, 2, result);
		return result;
	}


	double distance2( int x1, int y1, int x2, int y2 ){
		return square(x2-x1)+square(y2-y1);
	}
}
