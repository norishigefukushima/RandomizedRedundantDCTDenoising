#pragma once
#include <labutility.hpp>

void cvtColorBGR2PLANE(const cv::Mat& src, cv::Mat& dest);
void cvtColorPLANE2BGR(const cv::Mat& src, cv::Mat& dest);
class RedundantDXTDenoise
{
public:
	enum BASIS
	{
		DCT = 0,
		DHT = 1,
		DWT = 2//under construction
	};
	bool isSSE;

	void init(cv::Size size_, int color_, cv::Size patch_size_);
	RedundantDXTDenoise(cv::Size size, int color, cv::Size patch_size_ = cv::Size(8, 8));
	RedundantDXTDenoise();
	virtual void operator()(cv::Mat& src, cv::Mat& dest, float sigma, cv::Size psize = cv::Size(8, 8), BASIS transform_basis = BASIS::DCT);

	//void shearable(cv::Mat& src, cv::Mat& dest, float sigma, cv::Size psize = cv::Size(8, 8), int transform_basis = 0, int direct = 0);
	//void weighted(cv::Mat& src, cv::Mat& dest, float sigma, cv::Size psize = cv::Size(8, 8), int transform_basis = 0);
	//void test(cv::Mat& src, cv::Mat& dest, float sigma, cv::Size psize = cv::Size(8, 8));

protected:
	float getThreshold(float sigmaNoise);
	BASIS basis;
	cv::Size patch_size;
	cv::Size size;
	cv::Mat buff;
	cv::Mat sum;

	cv::Mat im;

	int channel;

	virtual void body(float *src, float* dest, float Th);
	void body(float *src, float* dest, float* wmap, float Th);
	void body(float *src, float* dest, float Th, int dr);

	void div(float* inplace0, float* inplace1, float* inplace2, float* w0, float* w1, float* w2, const int size1);
	void div(float* inplace0, float* inplace1, float* inplace2, const int patch_area, const int size1);

	void div(float* inplace0, float* w0, const int size1);
	void div(float* inplace0, const int patch_area, const int size1);

	void decorrelateColorForward(float* src, float* dest, int width, int height);
	void decorrelateColorInvert(float* src, float* dest, int width, int height);
};

class RandomizedRedundantDXTDenoise : public RedundantDXTDenoise
{
public:
	enum SAMPLING
	{
		FULL = 0,
		LATTICE,
		POISSONDISK,
		RANDOM_IMAGE_LUT,
		RANDOM_SAMPLE_LUT
	};

	RandomizedRedundantDXTDenoise(){ ; };
	RandomizedRedundantDXTDenoise(cv::Size size, int color, cv::Size patch_size_ = cv::Size(8, 8)) :RedundantDXTDenoise(size, color, patch_size_)
	{
		generateSamplingMaps(size, patch_size_, 20, 0, SAMPLING::FULL);
	}

	void generateSamplingMaps(cv::Size imageSize, cv::Size patch_size, int number_of_LUT, int d, SAMPLING sampleType = SAMPLING::POISSONDISK);

	virtual void operator()(cv::Mat& src_, cv::Mat& dest, float sigma, cv::Size psize = cv::Size(8, 8), BASIS transform_basis = BASIS::DCT);
	void colorredundunt(cv::Mat& src_, cv::Mat& dest, float sigma, cv::Size psize = cv::Size(8, 8), BASIS transform_basis = BASIS::DCT);

protected:
	void div(float* inplace0, float* inplace1, float* inplace2, float* count, const int size1);
	void div(float* inplace0, float* inplace1, float* inplace2, float* inplace3, float* count, const int size1);

	virtual void body(float *src, float* dest, float Th);


	void getSamplingFromLUT(cv::Mat& samplingMap);
	void setSamplingMap(cv::Mat& samplingMap, SAMPLING samplingType, int d);


	std::vector<cv::Mat> samplingMapLUTs;
	cv::Mat samplingMap;
	std::vector<cv::Point> sampleLUT;//not used
};