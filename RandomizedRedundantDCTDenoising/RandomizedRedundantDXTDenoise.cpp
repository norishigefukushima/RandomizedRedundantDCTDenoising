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
	for (int j = 0; j < dest.rows - patch_size.height; j++)
	{
		float* s0 = dest.ptr<float>(j);
		uchar* msk = mask.ptr<uchar>(j);
		if (patch_size.width == 16)
		{
			for (int i = 0; i < dest.cols - patch_size.width; i++)
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
			for (int i = 0; i < dest.cols - patch_size.width; i++)
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
		else if (patch_size.width == 4)
		{
			for (int i = 0; i < dest.cols - patch_size.width; i++)
			{
				if (msk[i])
				{
					const __m128 mones = _mm_set1_ps(1.0);
					for (int jp = 0; jp < 4; jp++)
					{
						float* d = &s0[(jp)*width];
						__m128 sp1 = _mm_loadu_ps(d);
						_mm_storeu_ps(d, _mm_add_ps(sp1, mones));
					}
				}
				s0++;
			}
		}
		else
		{
			for (int i = 0; i < dest.cols - patch_size.width; i++)
			{
				if (msk[i])
				{
					for (int jp = 0; jp < patch_size.height; jp++)
					{
						float* d = &s0[(jp)*width];
						for (int ip = 0; ip < patch_size.width; ip++)
						{
							d[ip] += 1.f;
						}
					}
				}
				s0++;
			}
		}
	}
}

class RRDCTThresholdingInvorker : public cv::ParallelLoopBody
{
private:
	float* src;
	float* dest;
	float thresh;
	int width;
	int height;
	Size patch_size;
	cv::Mat& randomMask;//for randomized sampling

public:
	int EvenOddFull;
	RRDCTThresholdingInvorker(float *sim, float* dim, Mat& rmask, float Th, int w, int h, Size psize) : src(sim), dest(dim), width(w), height(h), patch_size(psize), thresh(Th), randomMask(rmask)
	{
		EvenOddFull = 0;
	}

	virtual void operator() (const Range& range) const
	{
		int pwidth = patch_size.width;
		int pheight = patch_size.height;
		const int size1 = width * height;
		const int hstep = height - pheight;
		const int wstep = width - pwidth;

		Mat d = Mat(Size(width, height), CV_32F, dest);

		int start, end;
		if (EvenOddFull < 0)
		{
			start = range.start;
			end = range.end;
		}
		else if (EvenOddFull == 0)
		{
			start = range.start;
			end = range.start + (range.end - range.start) / 2;
		}
		if (EvenOddFull == 1)
		{
			start = range.start + (range.end - range.start) / 2;
			end = range.end;
		}
		for (int j = start; j != end; j++)
		{
			Mat patch(patch_size, CV_32F);
			float* ptch = patch.ptr<float>(0);

			float* s0 = &src[width*j];
			const int sz = sizeof(float)*patch_size.width;
			Mat mask;
			uchar* msk = randomMask.ptr<uchar>(j);//

			for (int i = 0; i < wstep; i++)
			{
				if (msk[i])//
				{
					for (int k = 0; k < patch_size.height; k++)
					{
						memcpy(ptch + k*patch_size.width, s0 + k*width, sz);
					}
					dct(patch, patch);

					float f0 = *(float*)patch.data;
					compare(abs(patch), thresh, mask, CMP_LT);
					patch.setTo(0.f, mask);
					*(float*)patch.data = f0;
					dct(patch, patch, DCT_INVERSE);

					Mat r = d(Rect(i, j, patch_size.width, patch_size.height));
					r += patch;
				}
				s0++;
			}
		}
	}
};

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
	int EvenOddFull;
	RRDCTThresholdingInvorker16x16(float *sim, float* dim, Mat& rmask, float Th, int w, int h) : src(sim), dest(dim), width(w), height(h), thresh(Th), randomMask(rmask)
	{
		EvenOddFull = 0;
	}
	virtual void operator() (const Range& range) const
	{
		const int size1 = width * height;
		const int hstep = height - 16;
		const int wstep = width - 16;
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

		int start, end;
		if (EvenOddFull < 0)
		{
			start = range.start;
			end = range.end;
		}
		else if (EvenOddFull == 0)
		{
			start = range.start;
			end = range.start + (range.end - range.start) / 2;
		}
		if (EvenOddFull == 1)
		{
			start = range.start + (range.end - range.start) / 2;
			end = range.end;
		}
		for (int j = start; j != end; j++)
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
	int EvenOddFull;
	RRDCTThresholdingInvorker8x8(float *sim, float* dim, Mat& rmask, float Th, int w, int h) : src(sim), dest(dim), width(w), height(h), thresh(Th), randomMask(rmask)
	{
		EvenOddFull = 0;
	}

	virtual void operator() (const Range& range) const
	{
		const int size1 = width * height;
		const int hstep = height - 8;
		const int wstep = width - 8;
		const int w1 = 1 * width;
		const int w2 = 2 * width;
		const int w3 = 3 * width;
		const int w4 = 4 * width;
		const int w5 = 5 * width;
		const int w6 = 6 * width;
		const int w7 = 7 * width;

		int start, end;
		if (EvenOddFull < 0)
		{
			start = range.start;
			end = range.end;
		}
		else if (EvenOddFull == 0)
		{
			start = range.start;
			end = range.start + (range.end - range.start) / 2;
		}
		if (EvenOddFull == 1)
		{
			start = range.start + (range.end - range.start) / 2;
			end = range.end;
		}
		for (int j = start; j != end; j++)
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

class RRDCTThresholdingInvorker4x4 : public cv::ParallelLoopBody
{
private:
	float* src;
	float* dest;
	float thresh;
	int width;
	int height;
	cv::Mat& randomMask;//for randomized sampling

public:
	int EvenOddFull;
	RRDCTThresholdingInvorker4x4(float *sim, float* dim, Mat& rmask, float Th, int w, int h) : src(sim), dest(dim), width(w), height(h), thresh(Th), randomMask(rmask)
	{
		EvenOddFull = 0;
	}

	virtual void operator() (const Range& range) const
	{
		const int size1 = width * height;
		const int hstep = height - 4;
		const int wstep = width - 4;
		const int w1 = 1 * width;
		const int w2 = 2 * width;
		const int w3 = 3 * width;

		int start, end;
		if (EvenOddFull < 0)
		{
			start = range.start;
			end = range.end;
		}
		else if (EvenOddFull == 0)
		{
			start = range.start;
			end = range.start + (range.end - range.start) / 2;
		}
		if (EvenOddFull == 1)
		{
			start = range.start + (range.end - range.start) / 2;
			end = range.end;
		}
		for (int j = start; j != end; j++)
		{
			Mat patch(Size(4, 4), CV_32F);
			float* ptch = patch.ptr<float>(0);

			float* s0 = &src[width*j];
			float* d0 = &dest[width*j];
			const int sz = sizeof(float) * 4;

			uchar* msk = randomMask.ptr<uchar>(j);//
			for (int i = 0; i < wstep; i++)
			{
				if (msk[i])//
				{
					memcpy(ptch, s0, sz);
					memcpy(ptch + 4, s0 + w1, sz);
					memcpy(ptch + 8, s0 + w2, sz);
					memcpy(ptch + 12, s0 + w3, sz);

					fDCT4x4_threshold_keep00_iDCT4x4(patch.ptr<float>(0), thresh);

					//add data
					for (int jp = 0; jp < 4; jp++)
					{
						float* s = patch.ptr<float>(jp);
						float* d = &d0[(jp)*width];
						__m128 mp1 = _mm_load_ps(s);
						__m128 sp1 = _mm_loadu_ps(d);

						_mm_storeu_ps(d, _mm_add_ps(sp1, mp1));
					}
				}
				s0++;
				d0++;
			}
		}
	}
};

class RRDCTThresholdingInvorker2x2 : public cv::ParallelLoopBody
{
private:
	float* src;
	float* dest;
	float thresh;
	int width;
	int height;
	cv::Mat& randomMask;//for randomized sampling

public:
	int EvenOddFull;
	RRDCTThresholdingInvorker2x2(float *sim, float* dim, Mat& rmask, float Th, int w, int h) : src(sim), dest(dim), width(w), height(h), thresh(Th), randomMask(rmask)
	{
		EvenOddFull = 0;
	}
	virtual void operator() (const Range& range) const
	{
		//2x2 patch
		const int size1 = width * height;
		const int hstep = height - 2;
		const int wstep = width - 2;

		int start, end;
		if (EvenOddFull < 0)
		{
			start = range.start;
			end = range.end;
		}
		else if (EvenOddFull == 0)
		{
			start = range.start;
			end = range.start + (range.end - range.start) / 2;
		}
		if (EvenOddFull == 1)
		{
			start = range.start + (range.end - range.start) / 2;
			end = range.end;
		}
		for (int j = start; j != end; j++)
		{
			Mat patch(Size(4, 2), CV_32F);
			float* ptch = patch.ptr<float>(0);

			float* s0 = &src[width*j];
			float* d0 = &dest[width*j];
			const int sz = sizeof(float) * 4;

			uchar* msk = randomMask.ptr<uchar>(j);//
			for (int i = 0; i < wstep; i += 4)
			{
				if (msk[i])//
				{
					memcpy(ptch, s0, sz);
					memcpy(ptch + 4, s0 + width, sz);

					fDCT2x2_2pack_thresh_keep00_iDCT2x2_2pack((float*)patch.data, (float*)patch.data, thresh);

					//add data
					__m128 mp1 = _mm_loadu_ps(ptch);
					__m128 sp1 = _mm_loadu_ps(d0);
					_mm_storeu_ps(d0, _mm_add_ps(sp1, mp1));
					mp1 = _mm_loadu_ps(ptch + 4);
					sp1 = _mm_loadu_ps(d0 + width);
					_mm_storeu_ps(d0 + width, _mm_add_ps(sp1, mp1));

					memcpy(ptch, s0 + 1, sz);
					memcpy(ptch + 4, s0 + width + 1, sz);

					fDCT2x2_2pack_thresh_keep00_iDCT2x2_2pack((float*)patch.data, (float*)patch.data, thresh);

					//add data
					mp1 = _mm_loadu_ps(ptch);
					sp1 = _mm_loadu_ps(d0 + 1);
					_mm_storeu_ps(d0 + 1, _mm_add_ps(sp1, mp1));
					mp1 = _mm_loadu_ps(ptch + 4);
					sp1 = _mm_loadu_ps(d0 + width + 1);
					_mm_storeu_ps(d0 + width + 1, _mm_add_ps(sp1, mp1));
				}
				s0 += 4;
				d0 += 4;
			}
		}
	}
};

class RRDHTThresholdingInvorker16x16 : public cv::ParallelLoopBody
{
private:
	float* src;
	float* dest;
	float thresh;
	int width;
	int height;
	cv::Mat& randomMask;//for randomized sampling

public:
	int EvenOddFull;
	RRDHTThresholdingInvorker16x16(float *sim, float* dim, Mat& rmask, float Th, int w, int h) : src(sim), dest(dim), width(w), height(h), thresh(Th), randomMask(rmask)
	{
		EvenOddFull = 0;
	}
	virtual void operator() (const Range& range) const
	{
		const int size1 = width * height;
		const int hstep = height - 16;
		const int wstep = width - 16;

		int start, end;
		if (EvenOddFull < 0)
		{
			start = range.start;
			end = range.end;
		}
		else if (EvenOddFull == 0)
		{
			start = range.start;
			end = range.start + (range.end - range.start) / 2;
		}
		if (EvenOddFull == 1)
		{
			start = range.start + (range.end - range.start) / 2;
			end = range.end;
		}
		for (int j = start; j != end; j++)
		{
			Mat patch(Size(16, 16), CV_32F);
			float* ptch = patch.ptr<float>(0);

			float* s0 = &src[width*j];
			float* d0 = &dest[width*j];
			const int sz = sizeof(float) * 16;

			uchar* msk = randomMask.ptr<uchar>(j);//
			for (int i = 0; i < wstep; i++)
			{
				if (msk[i])//
				{
					for (int n = 0; n < 16; n++)
						memcpy(ptch + 16 * n, s0 + n*width, sz);

					Hadamard2D16x16andThreshandIDHT(ptch, thresh);

					//add data
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
						s += 4;
						d += 4;
					}
				}
				s0++;
				d0++;
			}
		}
	}
};

class RRDHTThresholdingInvorker8x8 : public cv::ParallelLoopBody
{
private:
	float* src;
	float* dest;
	float thresh;
	int width;
	int height;
	cv::Mat& randomMask;//for randomized sampling

public:
	int EvenOddFull;
	RRDHTThresholdingInvorker8x8(float *sim, float* dim, Mat& rmask, float Th, int w, int h) : src(sim), dest(dim), width(w), height(h), thresh(Th), randomMask(rmask)
	{
		EvenOddFull = 0;
	}
	virtual void operator() (const Range& range) const
	{
		const int size1 = width * height;
		const int hstep = height - 8;
		const int wstep = width - 8;
		const int w1 = 1 * width;
		const int w2 = 2 * width;
		const int w3 = 3 * width;
		const int w4 = 4 * width;
		const int w5 = 5 * width;
		const int w6 = 6 * width;
		const int w7 = 7 * width;

		int start, end;
		if (EvenOddFull < 0)
		{
			start = range.start;
			end = range.end;
		}
		else if (EvenOddFull == 0)
		{
			start = range.start;
			end = range.start + (range.end - range.start) / 2;
		}
		if (EvenOddFull == 1)
		{
			start = range.start + (range.end - range.start) / 2;
			end = range.end;
		}
		for (int j = start; j != end; j++)
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
					memcpy(ptch, s0, sz);
					memcpy(ptch + 8, s0 + w1, sz);
					memcpy(ptch + 16, s0 + w2, sz);
					memcpy(ptch + 24, s0 + w3, sz);
					memcpy(ptch + 32, s0 + w4, sz);
					memcpy(ptch + 40, s0 + w5, sz);
					memcpy(ptch + 48, s0 + w6, sz);
					memcpy(ptch + 56, s0 + w7, sz);

					Hadamard2D8x8andThreshandIDHT(ptch, thresh);


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

class RRDHTThresholdingInvorker4x4 : public cv::ParallelLoopBody
{
private:
	float* src;
	float* dest;
	float thresh;
	int width;
	int height;
	cv::Mat& randomMask;//for randomized sampling

public:
	int EvenOddFull;
	RRDHTThresholdingInvorker4x4(float *sim, float* dim, Mat& rmask, float Th, int w, int h) : src(sim), dest(dim), width(w), height(h), thresh(Th), randomMask(rmask)
	{
		EvenOddFull = 0;
	}
	virtual void operator() (const Range& range) const
	{
		const int size1 = width * height;
		const int hstep = height - 4;
		const int wstep = width - 4;
		const int w1 = 1 * width;
		const int w2 = 2 * width;
		const int w3 = 3 * width;

		int start, end;
		if (EvenOddFull < 0)
		{
			start = range.start;
			end = range.end;
		}
		else if (EvenOddFull == 0)
		{
			start = range.start;
			end = range.start + (range.end - range.start) / 2;
		}
		if (EvenOddFull == 1)
		{
			start = range.start + (range.end - range.start) / 2;
			end = range.end;
		}
		for (int j = start; j != end; j++)
		{
			Mat patch(Size(4, 4), CV_32F);
			float* ptch = patch.ptr<float>(0);

			float* s0 = &src[width*j];
			float* d0 = &dest[width*j];
			const int sz = sizeof(float) * 4;

			uchar* msk = randomMask.ptr<uchar>(j);
			for (int i = 0; i < wstep; i++)
			{
				if (msk[i])
				{
					memcpy(ptch, s0, sz);
					memcpy(ptch + 4, s0 + w1, sz);
					memcpy(ptch + 8, s0 + w2, sz);
					memcpy(ptch + 12, s0 + w3, sz);

					Hadamard2D4x4andThreshandIDHT(ptch, thresh);

					//add data
					for (int jp = 0; jp < 4; jp++)
					{
						float* s = patch.ptr<float>(jp);
						float* d = &d0[(jp)*width];
						__m128 mp1 = _mm_load_ps(s);
						__m128 sp1 = _mm_loadu_ps(d);
						_mm_storeu_ps(d, _mm_add_ps(sp1, mp1));
					}
				}
				s0++;
				d0++;
			}
		}
	}
};

void RRDXTDenoise::body(float *src, float* dest, float Th)
{
	int numThreads = getNumThreads();

	if (basis == BASIS::DCT)
	{
		if (isSSE)
		{
			if (patch_size.width == 4)
			{
				RRDCTThresholdingInvorker4x4 invork(src, dest, samplingMap, Th, size.width, size.height);
				parallel_for_(Range(0, size.height - patch_size.height), invork, numThreads);
				invork.EvenOddFull = 1;
				parallel_for_(Range(0, size.height - patch_size.height), invork, numThreads);
			}
			else if (patch_size.width == 8)
			{
				RRDCTThresholdingInvorker8x8 invork(src, dest, samplingMap, Th, size.width, size.height);
				parallel_for_(Range(0, size.height - patch_size.height), invork, numThreads);
				invork.EvenOddFull = 1;
				parallel_for_(Range(0, size.height - patch_size.height), invork, numThreads);
			}
			else if (patch_size.width == 16)
			{
				RRDCTThresholdingInvorker16x16 invork(src, dest, samplingMap, Th, size.width, size.height);
				parallel_for_(Range(0, size.height - patch_size.height), invork, numThreads);
				invork.EvenOddFull = 1;
				parallel_for_(Range(0, size.height - patch_size.height), invork, numThreads);
			}
			else
			{
				RRDCTThresholdingInvorker invork(src, dest, samplingMap, Th, size.width, size.height, patch_size);
				parallel_for_(Range(0, size.height - patch_size.height), invork, numThreads);
				invork.EvenOddFull = 1;
				parallel_for_(Range(0, size.height - patch_size.height), invork, numThreads);
			}
		}
		else
		{
			RRDCTThresholdingInvorker invork(src, dest, samplingMap, Th, size.width, size.height, patch_size);
			parallel_for_(Range(0, size.height - patch_size.height), invork, numThreads);
			invork.EvenOddFull = 1;
			parallel_for_(Range(0, size.height - patch_size.height), invork, numThreads);
		}
	}
	else if (basis == BASIS::DHT)
	{
		if (isSSE)
		{
			if (patch_size.width == 4)
			{
				RRDHTThresholdingInvorker4x4 invork(src, dest, samplingMap, Th, size.width, size.height);
				parallel_for_(Range(0, size.height - patch_size.height), invork, numThreads);
				invork.EvenOddFull = 1;
				parallel_for_(Range(0, size.height - patch_size.height), invork, numThreads);
			}
			else if (patch_size.width == 8)
			{
				RRDHTThresholdingInvorker8x8 invork(src, dest, samplingMap, Th, size.width, size.height);
				parallel_for_(Range(0, size.height - patch_size.height), invork, numThreads);
				invork.EvenOddFull = 1;
				parallel_for_(Range(0, size.height - patch_size.height), invork, numThreads);
			}
			else if (patch_size.width == 16)
			{
				RRDHTThresholdingInvorker16x16 invork(src, dest, samplingMap, Th, size.width, size.height);
				parallel_for_(Range(0, size.height - patch_size.height), invork, numThreads);
				invork.EvenOddFull = 1;
				parallel_for_(Range(0, size.height - patch_size.height), invork, numThreads);
			}
			else
			{
				cout << "supported only 4x4 8x8 16x16" << endl;
			}
		}
		else
		{
			cout << "not supported for non SSE DHT" << endl;
		}
	}
	else if (basis == BASIS::DWT)
	{
		cout << "not supported" << endl;
	}
}

/*
// copy over block
class RandomizedRDCTDenoise8x8_Invoker : public cv::ParallelLoopBody
{
private:
float th;
float* src;
float* dst;
Mat& lut_im;
const int wstep;
const int hstep;
const int width;
const int height;

const int sliceHeightReal;
const Size clip_size;
const Size patch_size;
public:
RandomizedRDCTDenoise8x8_Invoker(float* sim, float* dest, Mat& lutim, float thresh, Size clipSize, Size patchSize, int image_width, int image_height, int w_step, int h_step, int slice_height_real) :
src(sim), dst(dest), lut_im(lutim), th(thresh), clip_size(clipSize), patch_size(patchSize), width(image_width), height(image_height), wstep(w_step), hstep(h_step), sliceHeightReal(slice_height_real)
{
}

virtual void operator() (const Range& range) const
{
const int w1 = 1 * width;
const int w2 = 2 * width;
const int w3 = 3 * width;
const int w4 = 4 * width;
const int w5 = 5 * width;
const int w6 = 6 * width;
const int w7 = 7 * width;

int data_size = sizeof(float)*patch_size.width;

for (int k = range.start; k != range.end; k++)
{
Mat patch(patch_size, CV_32F);
Mat sclip = Mat(clip_size, CV_32F, src + width*k*(sliceHeightReal));
Mat dc = Mat::zeros(clip_size, CV_32F);//data map
Mat cc = Mat::zeros(clip_size, CV_32F);//count map

float* s0 = sclip.ptr<float>(0);
float* d0 = dc.ptr<float>(0);
float* c0 = cc.ptr<float>(0);
for (int j = 0; j<hstep; j++)
{
float* ptch = patch.ptr<float>(0);
float* s = &s0[width*j];
float* d = &d0[width*j];
float* c = &c0[width*j];
uchar* pt = lut_im.ptr<uchar>(j + k*sliceHeightReal);

for (int i = 0; i<wstep; i++)
{
if (pt[i])
{
memcpy(ptch, s, data_size);
memcpy(ptch + 8, s + w1, data_size);
memcpy(ptch + 16, s + w2, data_size);
memcpy(ptch + 24, s + w3, data_size);
memcpy(ptch + 32, s + w4, data_size);
memcpy(ptch + 40, s + w5, data_size);
memcpy(ptch + 48, s + w6, data_size);
memcpy(ptch + 56, s + w7, data_size);

fDCT8x8_threshold_keep00_iDCT8x8(ptch, th);

const __m128 mones = _mm_set1_ps(1.0);
for (int t = 0; t < patch_size.width; t++)
{
float* s2 = patch.ptr<float>(t);
float* d2 = &d[t*width];
float* c2 = &c[t*width];
__m128 mp1 = _mm_load_ps(s2);
__m128 sp1 = _mm_loadu_ps(d2);
__m128 cp1 = _mm_loadu_ps(c2);

_mm_storeu_ps(d2, _mm_add_ps(sp1, mp1));
_mm_storeu_ps(c2, _mm_add_ps(cp1, mones));

s2 += 4;
d2 += 4;
c2 += 4;

mp1 = _mm_load_ps(s2);
sp1 = _mm_loadu_ps(d2);
cp1 = _mm_loadu_ps(c2);
_mm_storeu_ps(d2, _mm_add_ps(sp1, mp1));
_mm_storeu_ps(c2, _mm_add_ps(cp1, mones));
}
}
s++;
d++;
c++;
}
}

divide(dc, cc, dc);

//Mat temp_clip = cc(Rect(patch_size.width,patch_size.height,imwidth-2*patch_size.width,hThread)).clone();
//calcMatStatic(temp_clip);

Mat sum_clip = Mat(Size(width, sliceHeightReal), CV_32F, dst + width*sliceHeightReal*k);
dc(Rect(0, patch_size.height, width, sliceHeightReal)).copyTo(sum_clip);
}
}
};

*/
class RandomQueue
{
public:

	deque<Point> dq1, dq2;

	bool empty();
	int size();
	void push(Point pt);
	Point pop();
};

// Random Queue
bool RandomQueue::empty()
{
	return dq1.empty();
}
int RandomQueue::size()
{
	return (int)dq1.size();
}
void RandomQueue::push(Point pt)
{
	dq1.push_front(pt);
}
Point RandomQueue::pop()
{
	int n = rand() % dq1.size();

	Point pt;
	while (n--)
	{
		dq2.push_front(dq1.front());
		dq1.pop_front();
	}

	pt = dq1.front();
	dq1.pop_front();

	while (!dq2.empty())
	{
		dq1.push_front(dq2.front());
		dq2.pop_front();
	}
	return pt;
}

// Grid
class Grid
{
public:

	int width;
	int height;
	double cellsize;
	deque<Point> *bpt;

	Grid(double _cellsize, int _width, int _height);
	~Grid();
	Point imageToGrid(Point pt);
	void set(Point pt);
};

Grid::Grid(double _cellsize, int _width, int _height)
{
	cellsize = _cellsize;

	width = (int)(_width / cellsize) + 1;
	height = (int)(_height / cellsize) + 1;

	bpt = new deque<Point>[width*height];
}
Grid::~Grid()
{
	delete[] bpt;
}
Point Grid::imageToGrid(Point pt)
{
	int gx = (int)(pt.x / cellsize);
	int gy = (int)(pt.y / cellsize);

	return Point(gx, gy);
}

void Grid::set(Point pt)
{
	Point gpt = imageToGrid(pt);

	bpt[gpt.x + width*gpt.y].push_front(pt);
}

Point generateRandomPointAround(Point pt, const double mind, RNG& rng)
{
	float r1 = rng.uniform(0.f, 1.f);
	float r2 = rng.uniform(0.f, 1.f);

	double radius = mind*(1 + r1);
	double angle = 2 * CV_PI*r2;

	int x = (int)(pt.x + radius*cos(angle));
	int y = (int)(pt.y + radius*sin(angle));

	return Point(x, y);
}

bool inKernel(Point pt, int width, int height)
{
	if (pt.x < 0 || pt.x >= width)
		return false;
	if (pt.y < 0 || pt.y >= height)
		return false;
	return true;
}

inline float Distance(Point& pt1, Point& pt2)
{
	return (float)sqrt((pt1.x - pt2.x)*(pt1.x - pt2.x) + (pt1.y - pt2.y)*(pt1.y - pt2.y));
}

bool inNeibourhood(Grid& grid, Point pt, const double mind, const double cellsize)
{
	Point gpt = grid.imageToGrid(pt);

	for (int y = -2; y <= 2; y++)
	{
		if (y + gpt.y >= 0 && y + gpt.y < grid.height)
		{
			for (int x = -2; x <= 2; x++)
			{
				if (x + gpt.x >= 0 && x + gpt.x < grid.width)
				{
					Point ppt = Point(x + gpt.x, y + gpt.y);
					int w = grid.width;
					int num = (int)grid.bpt[ppt.x + w*ppt.y].size();

					if (num > 0)
					{
						int i = 0;

						while (i < num)
						{
							Point bpt_ = grid.bpt[ppt.x + w*ppt.y].at(i);
							float dist = Distance(bpt_, pt);
							if (dist < mind)
								return false;
							i++;
						}
					}
				}
			}
		}
	}
	return true;
}

void setPoissonDisk(Mat& kernel, const double mind, RNG& rng)
{
	CV_Assert(kernel.type() == CV_8U);
	kernel.setTo(0);
	int width = kernel.cols;
	int height = kernel.rows;

	double cellsize = mind / sqrt((double)2);

	Grid grid(cellsize, width, height);
	RandomQueue proc;

	Point first(rng.uniform(0, width), rng.uniform(0, height));

	proc.push(first);
	kernel.at<uchar>(first) = 255;
	grid.set(first);

	while (!proc.empty())
	{
		Point pt = proc.pop();
		for (int i = 0; i < 30; i++)
		{
			Point newpt = generateRandomPointAround(pt, mind, rng);

			if (inKernel(newpt, width, height) && inNeibourhood(grid, newpt, mind, cellsize))
			{
				proc.push(newpt);
				kernel.at<uchar>(newpt) = 255;
				grid.set(newpt);
			}
		}
	}
}

void setLattice(Mat& dest, int d, RNG& rng)
{
	d = max(1, d);
	int hd = d / 2;
	for (int j = d / 2; j < dest.rows - d / 2; j += d)
	{
		uchar* m = dest.ptr(j);
		for (int i = d / 2; i < dest.cols - d / 2; i += d)
		{
			int x = rng.uniform(-hd, hd);
			int y = dest.cols*rng.uniform(-hd, hd);
			m[y + x + i] = 1;
		}
	}
}

void RRDXTDenoise::setSamplingMap(Mat& samplingMap, SAMPLING samplingType, int d)
{
	Size s = Size(size.width + patch_size.width, size.height + patch_size.height);
	switch (samplingType)
	{
	default:
	case FULL:
		samplingMap = Mat::ones(s, CV_8U);
		break;

	case LATTICE:
		samplingMap = Mat::zeros(s, CV_8U);
		setLattice(samplingMap, d, rng);
		break;

	case POISSONDISK:
		samplingMap = Mat::zeros(s, CV_8U);
		setPoissonDisk(samplingMap, d, rng);
		break;
	}
}

void RRDXTDenoise::generateSamplingMaps(Size size, Size psize, int number_of_LUT, int d, SAMPLING sampleType)
{
	init(size, 3, psize);

	samplingMapLUTs.clear();
	samplingMapLUTs.resize(number_of_LUT);

	for (int i = 0; i < number_of_LUT; i++)
	{
		setSamplingMap(samplingMapLUTs[i], sampleType, d);
		//cp::showMatInfo(samplingMapLUTs[i]);
	}
}

void RRDXTDenoise::getSamplingFromLUT(Mat& samplingMap)
{
	if ((int)samplingMapLUTs.size() == 0) generateSamplingMaps(size, patch_size, 20, 0, SAMPLING::FULL);
	samplingMapLUTs[rng.uniform(0, (int)samplingMapLUTs.size())].copyTo(samplingMap);
}

void RRDXTDenoise::div(float* inplace0, float* inplace1, float* inplace2, float* count, const int size1)
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

void RRDXTDenoise::div(float* inplace0, float* inplace1, float* inplace2, float* inplace3, float* count, const int size1)
{
	float* s0 = inplace0;
	float* s1 = inplace1;
	float* s2 = inplace2;
	float* s3 = inplace3;
	float* c = count;

	for (int i = 0; i < size1; i += 4)
	{
		__m128 md0 = _mm_load_ps(s0);
		__m128 md1 = _mm_load_ps(s1);
		__m128 md2 = _mm_load_ps(s2);
		__m128 md3 = _mm_load_ps(s3);
		__m128 mdiv = _mm_rcp_ps(_mm_load_ps(c));
		_mm_store_ps(s0, _mm_mul_ps(md0, mdiv));
		_mm_store_ps(s1, _mm_mul_ps(md1, mdiv));
		_mm_store_ps(s2, _mm_mul_ps(md2, mdiv));
		_mm_store_ps(s3, _mm_mul_ps(md3, mdiv));

		s0 += 4;
		s1 += 4;
		s2 += 4;
		s3 += 4;
		c += 4;
	}
}

void redundantColorTransformFwd(Mat& src, Mat& dest)
{
	dest.create(Size(src.cols, src.rows * 4), CV_32F);

	float* s = src.ptr<float>(0);
	float* d = dest.ptr<float>(0);

	const int size = src.cols*src.rows;
	const int step0 = 0 * size;
	const int step1 = 1 * size;
	const int step2 = 2 * size;
	const int step3 = 3 * size;
	for (int i = 0; i < src.size().area(); i++)
	{
		d[i + step0] = s[3 * i + 2] + s[3 * i + 1] + s[3 * i + 0];
		d[i + step1] = s[3 * i + 2] - s[3 * i + 1];
		d[i + step2] = s[3 * i + 1] - s[3 * i + 0];
		d[i + step3] = -s[3 * i + 2] + s[3 * i + 0];
	}
}

void redundantColorTransformInv(Mat& src, Mat& dest)
{
	if (dest.empty())dest.create(Size(src.cols, src.rows), CV_32FC3);

	const int size = src.cols*src.rows / 4;
	const int step0 = 0 * size;
	const int step1 = 1 * size;
	const int step2 = 2 * size;
	const int step3 = 3 * size;

	float* s = src.ptr<float>(0);
	float* d = dest.ptr<float>(0);
	for (int i = 0; i < src.size().area() / 4; i++)
	{
		d[3 * i + 2] = 0.33333f*(s[i + step0] + s[i + step1] - s[i + step3]);
		d[3 * i + 1] = 0.33333f*(s[i + step0] - s[i + step1] + s[i + step2]);
		d[3 * i + 0] = 0.33333f*(s[i + step0] - s[i + step2] + s[i + step3]);
	}
}

void RRDXTDenoise::colorredundunt(Mat& src_, Mat& dest, float sigma, Size psize, BASIS transform_basis)
{
	Mat src;
	if (src_.depth() != CV_32F)src_.convertTo(src, CV_MAKETYPE(CV_32F, src_.channels()));
	else src = src_;

	basis = transform_basis;
	if (src.size() != size || src.channels() != channel || psize != patch_size)
	{
		init(src.size(), src.channels(), psize);
	}

	int w = src.cols + 2 * patch_size.width;
	w = ((4 - w % 4) % 4);

	copyMakeBorder(src, im, psize.height, patch_size.height, patch_size.width, patch_size.width + w, cv::BORDER_REPLICATE);

	const int width = im.cols;
	const int height = im.rows;
	const int size1 = width*height;
	float* ipixels;
	float* opixels;

	// Threshold
	float Th = getThreshold(sigma);
	Mat cmap = Mat::zeros(im.size(), CV_32F);
	{
		if (channel == 3)
		{
			redundantColorTransformFwd(im, buff);
		}
		else
		{
			buff = im.clone();
		}

		sum = Mat::zeros(buff.size(), CV_32F);
		ipixels = buff.ptr<float>(0);
		opixels = sum.ptr<float>(0);
	}

	getSamplingFromLUT(samplingMap);
	{
		if (channel == 3)
		{
			body(ipixels + 0 * size1, opixels + 0 * size1, Th);
			body(ipixels + 1 * size1, opixels + 1 * size1, Th);
			body(ipixels + 2 * size1, opixels + 2 * size1, Th);
			body(ipixels + 3 * size1, opixels + 3 * size1, Th);
		}
		else
		{
			body(ipixels, opixels, Th);
		}
		computeRandomizedCoutMap(cmap, samplingMap, patch_size);
	}
	div(opixels, opixels + size1, opixels + 2 * size1, opixels + 3 * size1, cmap.ptr<float>(0), size1);

	{
		// inverse 3-point DCT transform in the color dimension
		if (channel == 3)
		{
			redundantColorTransformInv(sum, im);
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

void RRDXTDenoise::operator()(Mat& src_, Mat& dest, float sigma, Size psize, BASIS transform_basis)
{
	Mat src;
	if (src_.depth() != CV_32F)src_.convertTo(src, CV_MAKETYPE(CV_32F, src_.channels()));
	else src = src_;

	basis = transform_basis;
	if (src.size() != size || src.channels() != channel || psize != patch_size)
	{
		init(src.size(), src.channels(), psize);
	}

	int w = src.cols + 2 * patch_size.width;
	w = ((4 - w % 4) % 4);

	copyMakeBorder(src, im, psize.height, patch_size.height, patch_size.width, patch_size.width + w, cv::BORDER_REPLICATE);

	const int width = im.cols;
	const int height = im.rows;
	const int size1 = width*height;
	float* ipixels;
	float* opixels;

	// Threshold
	float Th = getThreshold(sigma);
	Mat cmap = Mat::zeros(im.size(), CV_32F);
	{
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

	getSamplingFromLUT(samplingMap);
	{
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
		computeRandomizedCoutMap(cmap, samplingMap, patch_size);
	}
	if (channel == 3) div(opixels, opixels + size1, opixels + 2 * size1, cmap.ptr<float>(0), size1);
	else divide(sum, cmap, sum);

	{
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

RRDXTDenoise::RRDXTDenoise()
{
	rng(cv::getCPUTickCount()); 
}

RRDXTDenoise::RRDXTDenoise(cv::Size size, int color, cv::Size patch_size_) :RedundantDXTDenoise(size, color, patch_size_)
{
	rng(cv::getCPUTickCount());
	generateSamplingMaps(size, patch_size_, 20, 0, SAMPLING::FULL);
}