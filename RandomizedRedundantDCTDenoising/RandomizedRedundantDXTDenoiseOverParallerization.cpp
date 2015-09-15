#include "RedundantDXTDenoise.h"
using namespace std;
using namespace cv;

void transpose4x4(float* inplace);
void transpose4x4(const float* src, float* dest);
void transpose8x8(float* inplace);
void transpose8x8(const float* src, float* dest);
void transpose16x16(float* inplace);
void transpose16x16(const float* src, float* dest);

///////////////////////////////////////////////////////////////////////////////////////////////////
//DCT simd functions///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
void iDCT16x16(const float* src, float* dest);
void fDCT16x16(const float* src, float* dest);
void fDCT16x16_threshold_keep00_iDCT16x16(const float* src, float* dest, float th);

void iDCT8x8(const float* s, float* d, float* temp);
void iDCT8x8(const float* s, float* d);
void fDCT8x8(const float* s, float* d, float* temp);
void fDCT8x8(const float* s, float* d);
int fDCT8x8__threshold_keep00_iDCT8x8_nonzero(float* s, float threshold);
void fDCT8x8_threshold_keep00_iDCT8x8(float* s, float threshold);

void fDCT4x4(float* a, float* b, float* temp);
void fDCT4x4(float* a, float* b);
void iDCT4x4(float* a, float* b, float* temp);
void iDCT4x4(float* a, float* b);
int fDCT4x4_threshold_keep00_iDCT4x4_nonzero(float* s, float threshold);
void fDCT4x4_threshold_keep00_iDCT4x4(float* s, float threshold);

void iDCT2x2(float* src, float* dest, float* temp);
void fDCT2x2(float* src, float* dest, float* temp);
void fDCT2x2_2pack_thresh_keep00_iDCT2x2_2pack(float* src, float* dest, float thresh);

/////////////////////////////////////////////////////////////////////////////////////
//Hadamard simd//////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////
void Hadamard2D4x4(float* src);
void Hadamard2D4x4andThresh(float* src, float thresh);
void Hadamard2D4x4andThreshandIDHT(float* src, float thresh);
void Hadamard2D8x8andThresh(float* src, float thresh);
void Hadamard2D8x8(float* src);
void Hadamard2D8x8andThreshandIDHT(float* src, float thresh);

void Hadamard2D16x16andThreshandIDHT(float* src, float thresh);
void Hadamard2D16x16(float* src);

static void computeRandomizedCoutMap(Mat& dest, Mat& mask, Size patch_size)
{
	const int width = dest.cols;
	for (int j = 0; j < dest.rows - patch_size.height + 1; j++)
	{
		float* s0 = dest.ptr<float>(j);
		uchar* msk = mask.ptr<uchar>(j);
		if (patch_size.width == 16)
		{
			for (int i = 0; i < dest.cols - patch_size.width + 1; i++)
			{
				if (msk[i])
				{
					const __m128 mones = _mm_set1_ps(1.0);
					for (int jp = 0; jp < 16; jp++)
					{
						float* d = &s0[(jp)*width];
						__m128 sp1 = _mm_loadu_ps(d);
						_mm_storeu_ps(d, _mm_add_ps(sp1, mones));

						d += 4;

						sp1 = _mm_loadu_ps(d);
						_mm_storeu_ps(d, _mm_add_ps(sp1, mones));

						d += 4;

						sp1 = _mm_loadu_ps(d);
						_mm_storeu_ps(d, _mm_add_ps(sp1, mones));

						d += 4;

						sp1 = _mm_loadu_ps(d);
						_mm_storeu_ps(d, _mm_add_ps(sp1, mones));
					}
				}
				s0++;
			}
		}
		else if (patch_size.width == 8)
		{
			for (int i = 0; i < dest.cols - patch_size.width + 1; i++)
			{
				if (msk[i])
				{
					const __m128 mones = _mm_set1_ps(1.0);
					for (int jp = 0; jp < 8; jp++)
					{
						float* d = &s0[(jp)*width];
						__m128 sp1 = _mm_loadu_ps(d);
						_mm_storeu_ps(d, _mm_add_ps(sp1, mones));
						d += 4;
						sp1 = _mm_loadu_ps(d);
						_mm_storeu_ps(d, _mm_add_ps(sp1, mones));
					}
				}
				s0++;
			}
		}
	}
}

class RRDCTThresholdingInvorker16x16 : public cv::ParallelLoopBody
{
private:
	float* src;
	float* dest;
	float thresh;
	int width;
	int height;
	cv::Mat& randomMask;//for randomized sampling

public:
	
	RRDCTThresholdingInvorker16x16(float *sim, float* dim, Mat& rmask, float Th, int w, int h) : src(sim), dest(dim), width(w), height(h), thresh(Th), randomMask(rmask)
	{
	}
	virtual void operator() (const Range& range) const
	{
		const int size1 = width * height;
		const int hstep = height - 16 + 1;
		const int wstep = width - 16 + 1;
		const int w1 = 1 * width;
		const int w2 = 2 * width;
		const int w3 = 3 * width;
		const int w4 = 4 * width;
		const int w5 = 5 * width;
		const int w6 = 6 * width;
		const int w7 = 7 * width;
		const int w8 = 8 * width;
		const int w9 = 9 * width;
		const int w10 = 10 * width;
		const int w11 = 11 * width;
		const int w12 = 12 * width;
		const int w13 = 13 * width;
		const int w14 = 14 * width;
		const int w15 = 15 * width;
		int j;

		for (j = range.start; j != range.end; j++)
		{
			Mat patch(Size(16, 16), CV_32F);
			float* ptch = patch.ptr<float>(0);

			float* s0 = &src[width*j];
			float* d0 = &dest[width*j];

			uchar* msk = randomMask.ptr<uchar>(j);//
			const int sz = sizeof(float) * 16;
			for (int i = 0; i < wstep; i++)
			{
				if (msk[i])//
				{
					memcpy(ptch, s0, sz);
					memcpy(ptch + 16, s0 + w1, sz);
					memcpy(ptch + 32, s0 + w2, sz);
					memcpy(ptch + 48, s0 + w3, sz);
					memcpy(ptch + 64, s0 + w4, sz);
					memcpy(ptch + 80, s0 + w5, sz);
					memcpy(ptch + 96, s0 + w6, sz);
					memcpy(ptch + 112, s0 + w7, sz);
					memcpy(ptch + 128, s0 + w8, sz);
					memcpy(ptch + 144, s0 + w9, sz);
					memcpy(ptch + 160, s0 + w10, sz);
					memcpy(ptch + 176, s0 + w11, sz);
					memcpy(ptch + 192, s0 + w12, sz);
					memcpy(ptch + 208, s0 + w13, sz);
					memcpy(ptch + 224, s0 + w14, sz);
					memcpy(ptch + 240, s0 + w15, sz);

					fDCT16x16_threshold_keep00_iDCT16x16(patch.ptr<float>(0), patch.ptr<float>(0), thresh);

					//add data
					const __m128 mones = _mm_set1_ps(1.0);
					for (int jp = 0; jp < 16; jp++)
					{
						float* s = patch.ptr<float>(jp);
						float* d = &d0[(jp)*width];

						__m128 mp1 = _mm_load_ps(s);
						__m128 sp1 = _mm_loadu_ps(d);
						_mm_storeu_ps(d, _mm_add_ps(sp1, mp1));
						s += 4;
						d += 4;

						mp1 = _mm_load_ps(s);
						sp1 = _mm_loadu_ps(d);
						_mm_storeu_ps(d, _mm_add_ps(sp1, mp1));
						s += 4;
						d += 4;

						mp1 = _mm_load_ps(s);
						sp1 = _mm_loadu_ps(d);
						_mm_storeu_ps(d, _mm_add_ps(sp1, mp1));
						s += 4;
						d += 4;

						mp1 = _mm_load_ps(s);
						sp1 = _mm_loadu_ps(d);
						_mm_storeu_ps(d, _mm_add_ps(sp1, mp1));
					}
				}
				s0++;
				d0++;
			}
		}
	}
};

class RRDCTThresholdingInvorker8x8 : public cv::ParallelLoopBody
{
private:
	float* src;
	float* dest;
	float thresh;
	int width;
	int height;
	cv::Mat& randomMask;//for randomized sampling

public:

	RRDCTThresholdingInvorker8x8(float *sim, float* dim, Mat& rmask, float Th, int w, int h) : src(sim), dest(dim), width(w), height(h), thresh(Th), randomMask(rmask)
	{
	}

	virtual void operator() (const Range& range) const
	{
		const int size1 = width * height;
		const int hstep = height - 8 + 1;
		const int wstep = width - 8 + 1;
		const int w1 = 1 * width;
		const int w2 = 2 * width;
		const int w3 = 3 * width;
		const int w4 = 4 * width;
		const int w5 = 5 * width;
		const int w6 = 6 * width;
		const int w7 = 7 * width;

		for (int j = range.start; j != range.end; j++)
		{
			Mat patch(Size(8, 8), CV_32F);
			float* ptch = patch.ptr<float>(0);

			float* s0 = &src[width*j];
			float* d0 = &dest[width*j];
			const int sz = sizeof(float) * 8;

			uchar* msk = randomMask.ptr<uchar>(j);//

			for (int i = 0; i < wstep; i++)
			{
				if (msk[i])//
				{
					memcpy(ptch + 0, s0, sz);
					memcpy(ptch + 8, s0 + w1, sz);
					memcpy(ptch + 16, s0 + w2, sz);
					memcpy(ptch + 24, s0 + w3, sz);
					memcpy(ptch + 32, s0 + w4, sz);
					memcpy(ptch + 40, s0 + w5, sz);
					memcpy(ptch + 48, s0 + w6, sz);
					memcpy(ptch + 56, s0 + w7, sz);

					fDCT8x8_threshold_keep00_iDCT8x8(ptch, thresh);

					//add data
					for (int jp = 0; jp < 8; jp++)
					{
						float* s = patch.ptr<float>(jp);
						float* d = &d0[(jp)*width];

						__m128 mp1 = _mm_load_ps(s);
						__m128 sp1 = _mm_loadu_ps(d);
						_mm_storeu_ps(d, _mm_add_ps(sp1, mp1));

						s += 4;
						d += 4;
						mp1 = _mm_load_ps(s);
						sp1 = _mm_loadu_ps(d);
						_mm_storeu_ps(d, _mm_add_ps(sp1, mp1));
					}
				}
				s0++;
				d0++;
			}
		}
	}
};

void RandomizedRedundantDXTDenoise::body(float *src, float* dest, float Th)
{
	int numThread = 1;  getNumThreads();
	if (basis == BASIS::DCT)
	{
		if (isSSE)
		{
			if (patch_size.width == 2)
			{
				;
			}
			else if (patch_size.width == 4)
			{
				;
			}
			else if (patch_size.width == 8)
			{
				RRDCTThresholdingInvorker8x8 invork(src, dest, sampleMap, Th, size.width, size.height);
				parallel_for_(Range(0, size.height - patch_size.height), invork, 24);
			}
			else if (patch_size.width == 16)
			{
				RRDCTThresholdingInvorker16x16 invork(src, dest, sampleMap, Th, size.width, size.height);
				parallel_for_(Range(0, size.height - patch_size.height), invork, 24);
			}
			else
			{
				;
			}
		}
		else
		{
			;
		}
	}
	else if (basis == BASIS::DHT)
	{
		cout << "not supported" << endl;
	}
	else if (basis == BASIS::DWT)
	{
		cout << "not supported" << endl;
	}
}


void RandomizedRedundantDXTDenoise::div(float* inplace0, float* inplace1, float* inplace2, float* count, const int size1)
{
	float* s0 = inplace0;
	float* s1 = inplace1;
	float* s2 = inplace2;
	float* c = count;

	for (int i = 0; i < size1; i += 4)
	{
		__m128 md0 = _mm_load_ps(s0);
		__m128 md1 = _mm_load_ps(s1);
		__m128 md2 = _mm_load_ps(s2);
		__m128 mdiv = _mm_rcp_ps(_mm_load_ps(c));
		_mm_store_ps(s0, _mm_mul_ps(md0, mdiv));
		_mm_store_ps(s1, _mm_mul_ps(md1, mdiv));
		_mm_store_ps(s2, _mm_mul_ps(md2, mdiv));

		s0 += 4;
		s1 += 4;
		s2 += 4;
		c += 4;
	}
}

void RandomizedRedundantDXTDenoise::interlace(Mat& src_, Mat& dest, float sigma, Size psize, BASIS transform_basis, SAMPLING sampleType, int d)
{
	Mat src;
	if (src_.depth() != CV_32F)src_.convertTo(src, CV_MAKETYPE(CV_32F, src_.channels()));
	else src = src_;

	basis = transform_basis;
	if (src.size() != size || src.channels() != channel || psize != patch_size) init(src.size(), src.channels(), psize);

	int w = src.cols + 2 * psize.width;
	w = ((4 - w % 4) % 4);

	copyMakeBorder(src, im, psize.height, psize.height, psize.width, psize.width + w, cv::BORDER_REPLICATE);

	const int width = im.cols;
	const int height = im.rows;
	const int size1 = width*height;
	float* ipixels;
	float* opixels;

	// Threshold
	float Th = 3 * sigma;
	Mat cmap = Mat::zeros(im.size(), CV_32F);
	{
#ifdef _CALCTIME_
		CalcTime t("color");
#endif
		if (channel == 3)
		{
			cvtColorBGR2PLANE(im, buff);
		}
		else
		{
			buff = im.clone();
		}

		sum = Mat::zeros(buff.size(), CV_32F);
		ipixels = buff.ptr<float>(0);
		opixels = sum.ptr<float>(0);

		if (channel == 3)
		{
			decorrelateColorForward(ipixels, ipixels, width, height);
		}
	}

	//setSampling(SAMPLING::FULL, 0);
	if (d<0) setSampling(sampleType, psize.width / 3);
	else setSampling(sampleType, d);

	{
#ifdef _CALCTIME_
		CalcTime t("body");
#endif
		if (channel == 3)
		{
			body(ipixels, opixels, Th);
			body(ipixels + size1, opixels + size1, Th);
			body(ipixels + 2 * size1, opixels + 2 * size1, Th);
		}
		else
		{
			body(ipixels, opixels, Th);
		}
		computeRandomizedCoutMap(cmap, sampleMap, psize);
	}
	div(opixels, opixels + size1, opixels + 2 * size1, cmap.ptr<float>(0), size1);

	{
#ifdef _CALCTIME_
		CalcTime t("inv color");
#endif
		// inverse 3-point DCT transform in the color dimension
		if (channel == 3)
		{
			decorrelateColorInvert(opixels, opixels, width, height);
			cvtColorPLANE2BGR(sum, im);
		}
		else
		{
			im = sum;
		}

		Mat im2;
		if (src_.depth() != CV_32F) im.convertTo(im2, src_.type());
		else im2 = im;

		Mat(im2(Rect(patch_size.width, patch_size.height, src.cols, src.rows))).copyTo(dest);
	}
}