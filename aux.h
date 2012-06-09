#ifndef AUX_H
#define AUX_H 
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

typedef unsigned char byte;
struct buffer{
	byte array[256];
	byte& operator[](int index){
		return array[index];
	}
	byte operator[](int index) const{
		return array[index];
	}
};

void print_info(const cv::Mat &image, std::ostream &out=std::cerr);
void print(const cv::Mat &image);

void swap(cv::Mat &a, cv::Mat &b);

//all the images have the same size and type
cv::Mat equal_mosaic( const std::vector<cv::Mat> &images, size_t r, size_t c); 
cv::Mat mosaic( const cv::Mat &a, const cv::Mat &b, bool vertical=true ); 

cv::Mat histogram(const cv::Mat &image);
cv::Mat local_equalization(const cv::Mat &original, size_t size);
cv::Mat draw_histogram(const cv::Mat &image);

cv::Mat convolve(const cv::Mat &image, const cv::Mat &kernel);

std::vector<buffer> cargar_paleta(const char *filename);

cv::Mat hacer_compleja(const cv::Mat &image);
cv::Mat fourier_transform(const cv::Mat &image);
void center(cv::Mat &image);

cv::Mat magnitude(const cv::Mat &image);
cv::Mat magnitud_log(const cv::Mat &image);
cv::Mat phase(const cv::Mat &image);
cv::Mat real(const cv::Mat &image);

cv::Mat ver_espectro(const cv::Mat &image);

cv::Mat filtro_ideal(size_t rows, size_t cols, double corte);
cv::Mat filtro_butterworth(size_t rows, size_t cols, double corte, size_t order);
cv::Mat filtro_gaussian(size_t rows, size_t cols, double corte);
cv::Mat filtro_en_frecuencia(const cv::Mat &image, const cv::Mat &filter_magnitude);

#endif /* AUX_H */
