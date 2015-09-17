#include "RedundantDXTDenoise.h"
using namespace cv;
using namespace std;

double YPSNR(InputArray src1, InputArray src2)
{
	Mat g1, g2;
	if (src1.channels() == 1) g1 = src1.getMat();
	else cvtColor(src1, g1, COLOR_BGR2GRAY); 
	if (src2.channels() == 1) g2 = src2.getMat();
	else cvtColor(src2, g2, COLOR_BGR2GRAY);
	
	return PSNR(g1, g2);
}

template <class T>
void addNoiseSoltPepperMono_(Mat& src, Mat& dest, double per)
{
	cv::RNG rng;
	for (int j = 0; j < src.rows; j++)
	{
		T* s = src.ptr<T>(j);
		T* d = dest.ptr<T>(j);
		for (int i = 0; i < src.cols; i++)
		{
			double a1 = rng.uniform((double)0, (double)1);

			if (a1 > per)
				d[i] = s[i];
			else
			{
				double a2 = rng.uniform((double)0, (double)1);
				if (a2 > 0.5)d[i] = (T)0.0;
				else d[i] = (T)255.0;
			}
		}
	}
}

void addNoiseSoltPepperMono(Mat& src, Mat& dest, double per)
{
	if (src.type() == CV_8U) addNoiseSoltPepperMono_<uchar>(src, dest, per);
	if (src.type() == CV_16U) addNoiseSoltPepperMono_<ushort>(src, dest, per);
	if (src.type() == CV_16S) addNoiseSoltPepperMono_<short>(src, dest, per);
	if (src.type() == CV_32S) addNoiseSoltPepperMono_<int>(src, dest, per);
	if (src.type() == CV_32F) addNoiseSoltPepperMono_<float>(src, dest, per);
	if (src.type() == CV_64F) addNoiseSoltPepperMono_<double>(src, dest, per);
}

void addNoiseMono_nf(Mat& src, Mat& dest, double sigma)
{
	Mat s;
	src.convertTo(s, CV_32S);
	Mat n(s.size(), CV_32S);
	randn(n, 0, sigma);
	Mat temp = s + n;
	temp.convertTo(dest, src.type());
}

void addNoiseMono_f(Mat& src, Mat& dest, double sigma)
{
	Mat s;
	src.convertTo(s, CV_64F);
	Mat n(s.size(), CV_64F);
	randn(n, 0, sigma);
	Mat temp = s + n;
	temp.convertTo(dest, src.type());
}

void addNoiseMono(Mat& src, Mat& dest, double sigma)
{
	if (src.type() == CV_32F || src.type() == CV_64F)
	{
		addNoiseMono_f(src, dest, sigma);
	}
	else
	{
		addNoiseMono_nf(src, dest, sigma);
	}
}

void addNoise(InputArray src_, OutputArray dest_, double sigma, double sprate)
{
	if (dest_.empty() || dest_.size() != src_.size() || dest_.type() != src_.type()) dest_.create(src_.size(), src_.type());
	Mat src = src_.getMat();
	Mat dest = dest_.getMat();
	if (src.channels() == 1)
	{
		addNoiseMono(src, dest, sigma);
		if (sprate != 0)addNoiseSoltPepperMono(dest, dest, sprate);
		return;
	}
	else
	{
		vector<Mat> s(src.channels());
		vector<Mat> d(src.channels());
		split(src, s);
		for (int i = 0; i < src.channels(); i++)
		{
			addNoiseMono(s[i], d[i], sigma);
			if (sprate != 0)addNoiseSoltPepperMono(d[i], d[i], sprate);
		}
		cv::merge(d, dest);
	}
}

void CalcTime::start()
{
	pre = getTickCount();
}

void CalcTime::restart()
{
	start();
}

void CalcTime::lap(string message)
{
	string v = message + format(" %f", getTime());
	switch (timeMode)
	{
	case TIME_NSEC:
		v += " NSEC";
		break;
	case TIME_SEC:
		v += " SEC";
		break;
	case TIME_MIN:
		v += " MIN";
		break;
	case TIME_HOUR:
		v += " HOUR";
		break;

	case TIME_MSEC:
	default:
		v += " msec";
		break;
	}


	lap_mes.push_back(v);
	restart();
}
void CalcTime::show()
{
	getTime();

	int mode = timeMode;
	if (timeMode == TIME_AUTO)
	{
		mode = autoMode;
	}

	switch (mode)
	{
	case TIME_NSEC:
		cout << mes << ": " << cTime << " nsec" << endl;
		break;
	case TIME_SEC:
		cout << mes << ": " << cTime << " sec" << endl;
		break;
	case TIME_MIN:
		cout << mes << ": " << cTime << " minute" << endl;
		break;
	case TIME_HOUR:
		cout << mes << ": " << cTime << " hour" << endl;
		break;

	case TIME_MSEC:
	default:
		cout << mes << ": " << cTime << " msec" << endl;
		break;
	}
}

void CalcTime::show(string mes)
{
	getTime();

	int mode = timeMode;
	if (timeMode == TIME_AUTO)
	{
		mode = autoMode;
	}

	switch (mode)
	{
	case TIME_NSEC:
		cout << mes << ": " << cTime << " nsec" << endl;
		break;
	case TIME_SEC:
		cout << mes << ": " << cTime << " sec" << endl;
		break;
	case TIME_MIN:
		cout << mes << ": " << cTime << " minute" << endl;
		break;
	case TIME_HOUR:
		cout << mes << ": " << cTime << " hour" << endl;
		break;
	case TIME_DAY:
		cout << mes << ": " << cTime << " day" << endl;
	case TIME_MSEC:
		cout << mes << ": " << cTime << " msec" << endl;
		break;
	default:
		cout << mes << ": error" << endl;
		break;
	}
}

int CalcTime::autoTimeMode()
{
	if (cTime > 60.0*60.0*24.0)
	{
		return TIME_DAY;
	}
	else if (cTime > 60.0*60.0)
	{
		return TIME_HOUR;
	}
	else if (cTime > 60.0)
	{
		return TIME_MIN;
	}
	else if (cTime > 1.0)
	{
		return TIME_SEC;
	}
	else if (cTime > 1.0 / 1000.0)
	{
		return TIME_MSEC;
	}
	else
	{

		return TIME_NSEC;
	}
}
double CalcTime::getTime()
{
	cTime = (getTickCount() - pre) / (getTickFrequency());

	int mode = timeMode;
	if (mode == TIME_AUTO)
	{
		mode = autoTimeMode();
		autoMode = mode;
	}

	switch (mode)
	{
	case TIME_NSEC:
		cTime *= 1000000.0;
		break;
	case TIME_SEC:
		cTime *= 1.0;
		break;
	case TIME_MIN:
		cTime /= (60.0);
		break;
	case TIME_HOUR:
		cTime /= (60 * 60);
		break;
	case TIME_DAY:
		cTime /= (60 * 60 * 24);
		break;
	case TIME_MSEC:
	default:
		cTime *= 1000.0;
		break;
	}
	return cTime;
}
void CalcTime::setMessage(string src)
{
	mes = src;
}
void CalcTime::setMode(int mode)
{
	timeMode = mode;
}

void CalcTime::init(string message, int mode, bool isShow)
{
	_isShow = isShow;
	timeMode = mode;

	setMessage(message);
	start();
}


CalcTime::CalcTime()
{
	init("time ", TIME_AUTO, true);
}

CalcTime::CalcTime(string message, int mode, bool isShow)
{
	init(message, mode, isShow);
}
CalcTime::~CalcTime()
{
	getTime();
	if (_isShow)	show();
	if (lap_mes.size() != 0)
	{
		for (int i = 0; i < lap_mes.size(); i++)
		{
			cout << lap_mes[i] << endl;
		}
	}
}


void cvtColorBGR2PLANE_8u(const Mat& src, Mat& dest)
{
	dest.create(Size(src.cols, src.rows * 3), CV_8U);

	const int size = src.size().area();
	const int ssesize = 3 * size - ((48 - (3 * size) % 48) % 48);
	const int ssecount = ssesize / 48;
	const uchar* s = src.ptr<uchar>(0);
	uchar* B = dest.ptr<uchar>(0);//line by line interleave
	uchar* G = dest.ptr<uchar>(src.rows);
	uchar* R = dest.ptr<uchar>(2 * src.rows);

	//BGR BGR BGR BGR BGR B	
	//GR BGR BGR BGR BGR BG
	//R BGR BGR BGR BGR BGR
	//BBBBBBGGGGGRRRRR shuffle
	const __m128i mask1 = _mm_setr_epi8(0, 3, 6, 9, 12, 15, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14);
	//GGGGGBBBBBBRRRRR shuffle
	const __m128i smask1 = _mm_setr_epi8(6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15);
	const __m128i ssmask1 = _mm_setr_epi8(11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

	//GGGGGGBBBBBRRRRR shuffle
	const __m128i mask2 = _mm_setr_epi8(0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13);
	//const __m128i smask2 = _mm_setr_epi8(6,7,8,9,10,0,1,2,3,4,5,11,12,13,14,15);
	const __m128i ssmask2 = _mm_setr_epi8(0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 5, 6, 7, 8, 9, 10);

	//RRRRRRGGGGGBBBBB shuffle -> same mask2
	//__m128i mask3 = _mm_setr_epi8(0,3,6,9,12,15, 2,5,8,11,14,1,4,7,10,13);

	//const __m128i smask3 = _mm_setr_epi8(6,7,8,9,10,0,1,2,3,4,5,6,7,8,9,10);
	//const __m128i ssmask3 = _mm_setr_epi8(11,12,13,14,15,0,1,2,3,4,5,6,7,8,9,10);

	const __m128i bmask1 = _mm_setr_epi8
		(255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

	const __m128i bmask2 = _mm_setr_epi8
		(255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0);

	const __m128i bmask3 = _mm_setr_epi8
		(255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

	const __m128i bmask4 = _mm_setr_epi8
		(255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0);

	__m128i a, b, c;

	for (int i = 0; i < ssecount; i++)
	{
		a = _mm_shuffle_epi8(_mm_load_si128((__m128i*)(s)), mask1);
		b = _mm_shuffle_epi8(_mm_load_si128((__m128i*)(s + 16)), mask2);
		c = _mm_shuffle_epi8(_mm_load_si128((__m128i*)(s + 32)), mask2);
		_mm_storeu_si128((__m128i*)(B), _mm_blendv_epi8(c, _mm_blendv_epi8(b, a, bmask1), bmask2));
		a = _mm_shuffle_epi8(a, smask1);
		b = _mm_shuffle_epi8(b, smask1);
		c = _mm_shuffle_epi8(c, ssmask1);
		_mm_storeu_si128((__m128i*)(G), _mm_blendv_epi8(c, _mm_blendv_epi8(b, a, bmask3), bmask2));

		a = _mm_shuffle_epi8(a, ssmask1);
		c = _mm_shuffle_epi8(c, ssmask1);
		b = _mm_shuffle_epi8(b, ssmask2);

		_mm_storeu_si128((__m128i*)(R), _mm_blendv_epi8(c, _mm_blendv_epi8(b, a, bmask3), bmask4));

		s += 48;
		R += 16;
		G += 16;
		B += 16;
	}
	for (int i = ssesize; i < 3 * size; i += 3)
	{
		B[0] = s[0];
		G[0] = s[1];
		R[0] = s[2];
		s += 3, R++, G++, B++;
	}
}

void cvtColorPLANEDCT32BGR_32f(const Mat& src, Mat& dest)
{
	const float c00 = 0.57735f;
	const float c01 = 0.707107f;
	const float c02 = 0.408248f;
	const float c12 = -0.816497f;

	const int height = src.rows / 3;
	const int width = src.cols;
	const int size1 = width*height;
	const int size2 = 2 * size1;

	float* s = (float*)src.ptr<float>(0);
//#pragma omp parallel for
	for (int j = 0; j < height; j++)
	{
		float* d = dest.ptr<float>(j);

		float* s0 = s + width*j;
		float* s1 = s0 + size1;
		float* s2 = s0 + size2;
		const __m128 mc00 = _mm_set1_ps(c00);
		const __m128 mc01 = _mm_set1_ps(c01);
		const __m128 mc02 = _mm_set1_ps(c02);
		const __m128 mc12 = _mm_set1_ps(c12);
		int i = 0;
		for (i = 0; i < width - 4; i += 4)
		{
			__m128 ms0 = _mm_load_ps(s0);
			__m128 ms1 = _mm_load_ps(s1);
			__m128 ms2 = _mm_load_ps(s2);

			__m128 cs000 = _mm_mul_ps(mc00, ms0);
			__m128 cs002 = _mm_add_ps(cs000, _mm_mul_ps(mc02, ms2));

			__m128 bval = _mm_add_ps(cs002, _mm_mul_ps(mc01, ms1));
			__m128 gval = _mm_add_ps(cs000, _mm_mul_ps(mc12, ms2));
			__m128 rval = _mm_sub_ps(cs002, _mm_mul_ps(mc01, ms1));

			__m128 a = _mm_shuffle_ps(rval, rval, _MM_SHUFFLE(3, 0, 1, 2));
			__m128 b = _mm_shuffle_ps(bval, bval, _MM_SHUFFLE(1, 2, 3, 0));
			__m128 c = _mm_shuffle_ps(gval, gval, _MM_SHUFFLE(2, 3, 0, 1));

			_mm_stream_ps((d), _mm_blend_ps(_mm_blend_ps(b, a, 4), c, 2));
			_mm_stream_ps((d + 4), _mm_blend_ps(_mm_blend_ps(c, b, 4), a, 2));
			_mm_stream_ps((d + 8), _mm_blend_ps(_mm_blend_ps(a, c, 4), b, 2));
			d += 12;
			s0 += 4, s1 += 4, s2 += 4;
		}
		for (; i < width; i++)
		{
			float v0 = c00* *s0 + c01* *s1 + c02* *s2;
			float v1 = c00* *s0 + c12* *s2;
			float v2 = c00* *s0 - c01* *s1 + c02* *s2;

			d[0] = v0;
			d[1] = v1;
			d[2] = v2;
			d+=3, s0++, s1++, s2++;
		}
	}
}

void cvtColorBGR2DCT3PLANE_32f(const Mat& src, Mat& dest)
{
	if (dest.empty() || dest.size() !=Size(src.cols, src.rows*3))
	dest.create(Size(src.cols, src.rows * 3), CV_32F);

	const int size = src.size().area();
	const int ssesize = 3 * size - ((12 - (3 * size) % 12) % 12);
	const int ssecount = ssesize / 12;
	const float* s = src.ptr<float>(0);
	float* B = dest.ptr<float>(0);//line by line interleave
	float* G = dest.ptr<float>(src.rows);
	float* R = dest.ptr<float>(2 * src.rows);

	const float c0 = 0.57735f;
	const float c1 = 0.707107f;
	const float c20 = 0.408248f;
	const float c21 = -0.816497f;
	const __m128 mc0 = _mm_set1_ps(c0);
	const __m128 mc1 = _mm_set1_ps(c1);
	const __m128 mc20 = _mm_set1_ps(c20);
	const __m128 mc21 = _mm_set1_ps(c21);
//#pragma omp parallel for
	for (int i = 0; i < ssecount; i++)
	{
		int v = 4*i;
		int j = 12 * i;
		const float* sp = &s[j];

		__m128 a = _mm_loadu_ps(sp);
		__m128 b = _mm_loadu_ps(sp + 4);
		__m128 c = _mm_loadu_ps(sp + 8);

		__m128 aa = _mm_shuffle_ps(a, a, _MM_SHUFFLE(1, 2, 3, 0));
		aa = _mm_blend_ps(aa, b, 4);
		__m128 cc = _mm_shuffle_ps(c, c, _MM_SHUFFLE(1, 3, 2, 0));
		__m128 bbbb = _mm_blend_ps(aa, cc, 8);
		
		aa = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 2, 0, 1));
		__m128 bb = _mm_shuffle_ps(b, b, _MM_SHUFFLE(2, 3, 0, 1));
		bb = _mm_blend_ps(bb, aa, 1);
		cc = _mm_shuffle_ps(c, c, _MM_SHUFFLE(2, 3, 1, 0));
		__m128 gggg = _mm_blend_ps(bb, cc, 8);

		aa = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 1, 0, 2));
		bb = _mm_blend_ps(aa, b, 2);
		cc = _mm_shuffle_ps(c, c, _MM_SHUFFLE(3, 0, 1, 2));
		__m128 rrrr = _mm_blend_ps(bb, cc, 12);
		
		__m128 ms02a = _mm_add_ps(bbbb, rrrr);

		_mm_storeu_ps((B+v), _mm_mul_ps(mc0, _mm_add_ps(gggg, ms02a)));
		_mm_storeu_ps((G+v), _mm_mul_ps(mc1, _mm_sub_ps(bbbb, rrrr)));
		_mm_storeu_ps((R+v), _mm_add_ps(_mm_mul_ps(mc20, ms02a), _mm_mul_ps(mc21, gggg)));
	}
	for (int i = ssesize; i < 3 * size; i += 3)
	{
		int v = i/3;

		float v0 = c0*(s[i+0] + s[i+1] + s[i+2]);
		float v1 = c1*(s[0] - s[2]);
		float v2 = (s[0] + s[2])*c20 + s[1] *c21;
		B[v] = v0;
		G[v] = v1;
		R[v] = v2;
	}
}


void cvtColorBGR2PLANE_32f(const Mat& src, Mat& dest)
{
	dest.create(Size(src.cols, src.rows * 3), CV_32F);

	const int size = src.size().area();
	const int ssesize = 3 * size - ((12 - (3 * size) % 12) % 12);
	const int ssecount = ssesize / 12;
	const float* s = src.ptr<float>(0);
	float* B = dest.ptr<float>(0);//line by line interleave
	float* G = dest.ptr<float>(src.rows);
	float* R = dest.ptr<float>(2 * src.rows);

	for (int i = 0; i < ssecount; i++)
	{
		__m128 a = _mm_load_ps(s);
		__m128 b = _mm_load_ps(s + 4);
		__m128 c = _mm_load_ps(s + 8);

		__m128 aa = _mm_shuffle_ps(a, a, _MM_SHUFFLE(1, 2, 3, 0));
		aa = _mm_blend_ps(aa, b, 4);
		__m128 cc = _mm_shuffle_ps(c, c, _MM_SHUFFLE(1, 3, 2, 0));
		aa = _mm_blend_ps(aa, cc, 8);
		_mm_storeu_ps((B), aa);

		aa = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 2, 0, 1));
		__m128 bb = _mm_shuffle_ps(b, b, _MM_SHUFFLE(2, 3, 0, 1));
		bb = _mm_blend_ps(bb, aa, 1);
		cc = _mm_shuffle_ps(c, c, _MM_SHUFFLE(2, 3, 1, 0));
		bb = _mm_blend_ps(bb, cc, 8);
		_mm_storeu_ps((G), bb);

		aa = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 1, 0, 2));
		bb = _mm_blend_ps(aa, b, 2);
		cc = _mm_shuffle_ps(c, c, _MM_SHUFFLE(3, 0, 1, 2));
		cc = _mm_blend_ps(bb, cc, 12);
		_mm_storeu_ps((R), cc);

		s += 12;
		R += 4;
		G += 4;
		B += 4;
	}
	for (int i = ssesize; i < 3 * size; i += 3)
	{
		B[0] = s[0];
		G[0] = s[1];
		R[0] = s[2];
		s += 3, R++, G++, B++;
	}
}

template <class T>
void cvtColorBGR2PLANE_(const Mat& src, Mat& dest, int depth)
{
	vector<Mat> v(3);
	split(src, v);
	dest.create(Size(src.cols, src.rows * 3), depth);

	memcpy(dest.data, v[0].data, src.size().area()*sizeof(T));
	memcpy(dest.data + src.size().area()*sizeof(T), v[1].data, src.size().area()*sizeof(T));
	memcpy(dest.data + 2 * src.size().area()*sizeof(T), v[2].data, src.size().area()*sizeof(T));
}

void cvtColorBGR2PLANE(const Mat& src, Mat& dest)
{
	if (src.channels() != 3)printf("input image must have 3 channels\n");

	if (src.depth() == CV_8U)
	{
		//cvtColorBGR2PLANE_<uchar>(src, dest, CV_8U);
		//Mat d2;
		cvtColorBGR2PLANE_8u(src, dest);

	}
	else if (src.depth() == CV_16U)
	{
		cvtColorBGR2PLANE_<ushort>(src, dest, CV_16U);
	}
	if (src.depth() == CV_16S)
	{
		cvtColorBGR2PLANE_<short>(src, dest, CV_16S);
	}
	if (src.depth() == CV_32S)
	{
		cvtColorBGR2PLANE_<int>(src, dest, CV_32S);
	}
	if (src.depth() == CV_32F)
	{
		cvtColorBGR2PLANE_32f(src, dest);
		//cvtColorBGR2PLANE_<float>(src, dest, CV_32F);
	}
	if (src.depth() == CV_64F)
	{
		cvtColorBGR2PLANE_<double>(src, dest, CV_64F);
	}
}

template <class T>
void cvtColorPLANE2BGR_(const Mat& src, Mat& dest, int depth)
{
	int width = src.cols;
	int height = src.rows / 3;
	T* b = (T*)src.ptr<T>(0);
	T* g = (T*)src.ptr<T>(height);
	T* r = (T*)src.ptr<T>(2 * height);

	Mat B(height, width, src.type(), b);
	Mat G(height, width, src.type(), g);
	Mat R(height, width, src.type(), r);
	vector<Mat> v(3);
	v[0] = B;
	v[1] = G;
	v[2] = R;
	merge(v, dest);
}

void cvtColorPLANE2BGR_8u_align(const Mat& src, Mat& dest)
{
	int width = src.cols;
	int height = src.rows / 3;

	if (dest.empty()) dest.create(Size(width, height), CV_8UC3);
	else if (width != dest.cols || height != dest.rows) dest.create(Size(width, height), CV_8UC3);
	else if (dest.type() != CV_8UC3) dest.create(Size(width, height), CV_8UC3);

	uchar* B = (uchar*)src.ptr<uchar>(0);
	uchar* G = (uchar*)src.ptr<uchar>(height);
	uchar* R = (uchar*)src.ptr<uchar>(2 * height);

	uchar* D = (uchar*)dest.ptr<uchar>(0);

	int ssecount = width*height * 3 / 48;

	const __m128i mask1 = _mm_setr_epi8(0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5);
	const __m128i mask2 = _mm_setr_epi8(5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10);
	const __m128i mask3 = _mm_setr_epi8(10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15);
	const __m128i bmask1 = _mm_setr_epi8(0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0);
	const __m128i bmask2 = _mm_setr_epi8(255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255);

	for (int i = ssecount; i--;)
	{
		__m128i a = _mm_load_si128((const __m128i*)B);
		__m128i b = _mm_load_si128((const __m128i*)G);
		__m128i c = _mm_load_si128((const __m128i*)R);

		a = _mm_shuffle_epi8(a, mask1);
		b = _mm_shuffle_epi8(b, mask2);
		c = _mm_shuffle_epi8(c, mask3);
		_mm_stream_si128((__m128i*)(D), _mm_blendv_epi8(c, _mm_blendv_epi8(a, b, bmask1), bmask2));
		_mm_stream_si128((__m128i*)(D + 16), _mm_blendv_epi8(b, _mm_blendv_epi8(a, c, bmask2), bmask1));
		_mm_stream_si128((__m128i*)(D + 32), _mm_blendv_epi8(c, _mm_blendv_epi8(b, a, bmask2), bmask1));

		D += 48;
		B += 16;
		G += 16;
		R += 16;
	}
}

void cvtColorPLANE2BGR_8u(const Mat& src, Mat& dest)
{
	int width = src.cols;
	int height = src.rows / 3;

	if (dest.empty()) dest.create(Size(width, height), CV_8UC3);
	else if (width != dest.cols || height != dest.rows) dest.create(Size(width, height), CV_8UC3);
	else if (dest.type() != CV_8UC3) dest.create(Size(width, height), CV_8UC3);

	uchar* B = (uchar*)src.ptr<uchar>(0);
	uchar* G = (uchar*)src.ptr<uchar>(height);
	uchar* R = (uchar*)src.ptr<uchar>(2 * height);

	uchar* D = (uchar*)dest.ptr<uchar>(0);

	int ssecount = width*height * 3 / 48;
	int rem = width*height * 3 - ssecount * 48;

	const __m128i mask1 = _mm_setr_epi8(0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5);
	const __m128i mask2 = _mm_setr_epi8(5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10);
	const __m128i mask3 = _mm_setr_epi8(10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15);
	const __m128i bmask1 = _mm_setr_epi8(0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0);
	const __m128i bmask2 = _mm_setr_epi8(255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255);

	for (int i = ssecount; i--;)
	{
		__m128i a = _mm_loadu_si128((const __m128i*)B);
		__m128i b = _mm_loadu_si128((const __m128i*)G);
		__m128i c = _mm_loadu_si128((const __m128i*)R);

		a = _mm_shuffle_epi8(a, mask1);
		b = _mm_shuffle_epi8(b, mask2);
		c = _mm_shuffle_epi8(c, mask3);

		_mm_storeu_si128((__m128i*)(D), _mm_blendv_epi8(c, _mm_blendv_epi8(a, b, bmask1), bmask2));
		_mm_storeu_si128((__m128i*)(D + 16), _mm_blendv_epi8(b, _mm_blendv_epi8(a, c, bmask2), bmask1));
		_mm_storeu_si128((__m128i*)(D + 32), _mm_blendv_epi8(c, _mm_blendv_epi8(b, a, bmask2), bmask1));

		D += 48;
		B += 16;
		G += 16;
		R += 16;
	}
	for (int i = rem; i--;)
	{
		D[0] = *B;
		D[1] = *G;
		D[2] = *R;
		D += 3;
		B++, G++, R++;
	}
}

void cvtColorPLANE2BGR(const Mat& src, Mat& dest)
{
	if (src.depth() == CV_8U)
	{
		//cvtColorPLANE2BGR_<uchar>(src, dest, CV_8U);	
		if (src.cols % 16 == 0)
			cvtColorPLANE2BGR_8u_align(src, dest);
		else
			cvtColorPLANE2BGR_8u(src, dest);
	}
	else if (src.depth() == CV_16U)
	{
		cvtColorPLANE2BGR_<ushort>(src, dest, CV_16U);
	}
	if (src.depth() == CV_16S)
	{
		cvtColorPLANE2BGR_<short>(src, dest, CV_16S);
	}
	if (src.depth() == CV_32S)
	{
		cvtColorPLANE2BGR_<int>(src, dest, CV_32S);
	}
	if (src.depth() == CV_32F)
	{
		cvtColorPLANE2BGR_<float>(src, dest, CV_32F);
	}
	if (src.depth() == CV_64F)
	{
		cvtColorPLANE2BGR_<double>(src, dest, CV_64F);
	}
}

void decorrelateColorInvert(float* src, float* dest, int width, int height)
{
	const float c00 = 0.57735f;
	const float c01 = 0.707107f;
	const float c02 = 0.408248f;
	const float c12 = -0.816497f;

	const int size1 = width*height;
	const int size2 = 2 * size1;

//#pragma omp parallel for
	for (int j = 0; j < height; j++)
	{
		float* s0 = src + width*j;
		float* s1 = s0 + size1;
		float* s2 = s0 + size2;
		float* d0 = dest + width*j;
		float* d1 = d0 + size1;
		float* d2 = d0 + size2;

		const __m128 mc00 = _mm_set1_ps(c00);
		const __m128 mc01 = _mm_set1_ps(c01);
		const __m128 mc02 = _mm_set1_ps(c02);
		const __m128 mc12 = _mm_set1_ps(c12);
		int i = 0;
		//#ifdef _SSE
		for (i = 0; i < width - 4; i += 4)
		{
			__m128 ms0 = _mm_load_ps(s0);
			__m128 ms1 = _mm_load_ps(s1);
			__m128 ms2 = _mm_load_ps(s2);

			__m128 cs000 = _mm_mul_ps(mc00, ms0);
			__m128 cs002 = _mm_add_ps(cs000, _mm_mul_ps(mc02, ms2));
			_mm_store_ps(d0, _mm_add_ps(cs002, _mm_mul_ps(mc01, ms1)));
			_mm_store_ps(d1, _mm_add_ps(cs000, _mm_mul_ps(mc12, ms2)));
			_mm_store_ps(d2, _mm_sub_ps(cs002, _mm_mul_ps(mc01, ms1)));

			d0 += 4, d1 += 4, d2 += 4, s0 += 4, s1 += 4, s2 += 4;
		}
		//#endif
		for (; i < width; i++)
		{
			float v0 = c00* *s0 + c01* *s1 + c02* *s2;
			float v1 = c00* *s0 + c12* *s2;
			float v2 = c00* *s0 - c01* *s1 + c02* *s2;

			*d0++ = v0;
			*d1++ = v1;
			*d2++ = v2;
			s0++, s1++, s2++;
		}
	}
}

void decorrelateColorForward(float* src, float* dest, int width, int height)
{
	const float c0 = 0.57735f;
	const float c1 = 0.707107f;
	const float c20 = 0.408248f;
	const float c21 = -0.816497f;

	const int size1 = width*height;
	const int size2 = 2 * size1;

//#pragma omp parallel for
	for (int j = 0; j < height; j++)
	{
		float* s0 = src + width*j;
		float* s1 = s0 + size1;
		float* s2 = s0 + size2;
		float* d0 = dest + width*j;
		float* d1 = d0 + size1;
		float* d2 = d0 + size2;
		const __m128 mc0 = _mm_set1_ps(c0);
		const __m128 mc1 = _mm_set1_ps(c1);
		const __m128 mc20 = _mm_set1_ps(c20);
		const __m128 mc21 = _mm_set1_ps(c21);
		int i = 0;
		//#ifdef _SSE
		for (i = 0; i < width - 4; i += 4)
		{
			__m128 ms0 = _mm_load_ps(s0);
			__m128 ms1 = _mm_load_ps(s1);
			__m128 ms2 = _mm_load_ps(s2);

			__m128 ms02a = _mm_add_ps(ms0, ms2);

			_mm_store_ps(d0, _mm_mul_ps(mc0, _mm_add_ps(ms1, ms02a)));
			_mm_store_ps(d1, _mm_mul_ps(mc1, _mm_sub_ps(ms0, ms2)));
			_mm_store_ps(d2, _mm_add_ps(_mm_mul_ps(mc20, ms02a), _mm_mul_ps(mc21, ms1)));

			d0 += 4, d1 += 4, d2 += 4, s0 += 4, s1 += 4, s2 += 4;
		}
		//#endif
		for (; i < width; i++)
		{
			float v0 = c0*(*s0 + *s1 + *s2);
			float v1 = c1*(*s0 - *s2);
			float v2 = (*s0 + *s2)*c20 + *s1 *c21;

			*d0++ = v0;
			*d1++ = v1;
			*d2++ = v2;
			s0++, s1++, s2++;
		}
	}
}