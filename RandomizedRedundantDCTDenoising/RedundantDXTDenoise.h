#pragma once
#include <opencv2/opencv.hpp>

#define CV_VERSION_NUMBER CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#pragma comment(lib, "opencv_core"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_highgui"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_imgcodecs"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_imgproc"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_photo"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_xphoto"CV_VERSION_NUMBER"d.lib")
#else
#pragma comment(lib, "opencv_core"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_highgui"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_imgcodecs"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_imgproc"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_photo"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_xphoto"CV_VERSION_NUMBER".lib")
#endif

double YPSNR(cv::InputArray src1, cv::InputArray src2);
void addNoise(cv::InputArray src, cv::OutputArray dest, double sigma, double solt_papper_ratio = 0.0);
enum
{
	TIME_AUTO = 0,
	TIME_NSEC,
	TIME_MSEC,
	TIME_SEC,
	TIME_MIN,
	TIME_HOUR,
	TIME_DAY
};
class CalcTime
{
	int64 pre;
	std::string mes;

	int timeMode;

	double cTime;
	bool _isShow;

	int autoMode;
	int autoTimeMode();
	std::vector<std::string> lap_mes;
public:

	void start();
	void setMode(int mode);
	void setMessage(std::string src);
	void restart();
	double getTime();
	void show();
	void show(std::string message);
	void lap(std::string message);
	void init(std::string message, int mode, bool isShow);

	CalcTime(std::string message, int mode = TIME_AUTO, bool isShow = true);
	CalcTime();

	~CalcTime();
};

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

	void div(float* inplace0, float* inplace1, float* inplace2, float* w0, float* w1, float* w2, const int size1);
	void div(float* inplace0, float* inplace1, float* inplace2, const int patch_area, const int size1);

	void div(float* inplace0, float* w0, const int size1);
	void div(float* inplace0, const int patch_area, const int size1);

	void decorrelateColorForward(float* src, float* dest, int width, int height);
	void decorrelateColorInvert(float* src, float* dest, int width, int height);
};

class RRDXTDenoise : public RedundantDXTDenoise
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

	RRDXTDenoise(){ ; };
	RRDXTDenoise(cv::Size size, int color, cv::Size patch_size_ = cv::Size(8, 8)) :RedundantDXTDenoise(size, color, patch_size_)
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