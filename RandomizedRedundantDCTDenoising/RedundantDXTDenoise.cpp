#include "RedundantDXTDenoise.h"

#include <nmmintrin.h> //SSE4.2
#define  _USE_MATH_DEFINES
#include <stdio.h>
#include <math.h>
#include <string.h>

using namespace std;
using namespace cv;

void Hadamard2D4x4(float* src);
void Hadamard2D4x4andThresh(float* src, float thresh);
void Hadamard2D4x4andThreshandIDHT(float* src, float thresh);
void Hadamard2D8x8andThresh(float* src, float thresh);
void Hadamard2D8x8(float* src);
void Hadamard2D8x8andThreshandIDHT(float* src, float thresh);

void Hadamard2D16x16andThreshandIDHT(float* src, float thresh);
void Hadamard2D16x16(float* src);


#define _KEEP_00_COEF_

//info: code
//http://d.hatena.ne.jp/shiku_otomiya/20100902/p1 (in japanese)

//paper LLM89
//C. Loeffler, A. Ligtenberg, and G. S. Moschytz, 
//"Practical fast 1-D DCT algorithms with 11 multiplications,"
//Proc. Int'l. Conf. on Acoustics, Speech, and Signal Processing (ICASSP89), pp. 988-991, 1989.



void print_m128(__m128 src);

//////////////////////////////////////////////////////////////////////////////////////
//transpose simd//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

void transpose4x4(float* src)
{
	__m128 m0 = _mm_load_ps(src);
	__m128 m1 = _mm_load_ps(src + 4);
	__m128 m2 = _mm_load_ps(src + 8);
	__m128 m3 = _mm_load_ps(src + 12);
	_MM_TRANSPOSE4_PS(m0, m1, m2, m3);
	_mm_store_ps(src, m0);
	_mm_store_ps(src + 4, m1);
	_mm_store_ps(src + 8, m2);
	_mm_store_ps(src + 12, m3);
}

void transpose4x4(float* src, float* dest)
{
	__m128 m0 = _mm_load_ps(src);
	__m128 m1 = _mm_load_ps(src + 4);
	__m128 m2 = _mm_load_ps(src + 8);
	__m128 m3 = _mm_load_ps(src + 12);
	_MM_TRANSPOSE4_PS(m0, m1, m2, m3);
	_mm_store_ps(dest, m0);
	_mm_store_ps(dest + 4, m1);
	_mm_store_ps(dest + 8, m2);
	_mm_store_ps(dest + 12, m3);
}

void transpose8x8(const float* src, float* dest)
{
	__m128 m0 = _mm_load_ps(src);
	__m128 m1 = _mm_load_ps(src + 8);
	__m128 m2 = _mm_load_ps(src + 16);
	__m128 m3 = _mm_load_ps(src + 24);
	_MM_TRANSPOSE4_PS(m0, m1, m2, m3);
	_mm_store_ps(dest, m0);
	_mm_store_ps(dest + 8, m1);
	_mm_store_ps(dest + 16, m2);
	_mm_store_ps(dest + 24, m3);

	m0 = _mm_load_ps(src + 4);
	m1 = _mm_load_ps(src + 12);
	m2 = _mm_load_ps(src + 20);
	m3 = _mm_load_ps(src + 28);
	_MM_TRANSPOSE4_PS(m0, m1, m2, m3);
	_mm_store_ps(dest + 32, m0);
	_mm_store_ps(dest + 40, m1);
	_mm_store_ps(dest + 48, m2);
	_mm_store_ps(dest + 56, m3);

	m0 = _mm_load_ps(src + 32);
	m1 = _mm_load_ps(src + 40);
	m2 = _mm_load_ps(src + 48);
	m3 = _mm_load_ps(src + 56);
	_MM_TRANSPOSE4_PS(m0, m1, m2, m3);
	_mm_store_ps(dest + 4, m0);
	_mm_store_ps(dest + 12, m1);
	_mm_store_ps(dest + 20, m2);
	_mm_store_ps(dest + 28, m3);

	m0 = _mm_load_ps(src + 36);
	m1 = _mm_load_ps(src + 44);
	m2 = _mm_load_ps(src + 52);
	m3 = _mm_load_ps(src + 60);
	_MM_TRANSPOSE4_PS(m0, m1, m2, m3);
	_mm_store_ps(dest + 36, m0);
	_mm_store_ps(dest + 44, m1);
	_mm_store_ps(dest + 52, m2);
	_mm_store_ps(dest + 60, m3);
}

void transpose8x8(float* src)
{
	__declspec(align(16)) float temp[16];
	__m128 m0 = _mm_load_ps(src);
	__m128 m1 = _mm_load_ps(src + 8);
	__m128 m2 = _mm_load_ps(src + 16);
	__m128 m3 = _mm_load_ps(src + 24);
	_MM_TRANSPOSE4_PS(m0, m1, m2, m3);
	_mm_store_ps(src, m0);
	_mm_store_ps(src + 8, m1);
	_mm_store_ps(src + 16, m2);
	_mm_store_ps(src + 24, m3);


	m0 = _mm_load_ps(src + 4);
	m1 = _mm_load_ps(src + 12);
	m2 = _mm_load_ps(src + 20);
	m3 = _mm_load_ps(src + 28);
	_MM_TRANSPOSE4_PS(m0, m1, m2, m3);
	/*_mm_store_ps(dest+32,m0);
	_mm_store_ps(dest+40,m1);
	_mm_store_ps(dest+48,m2);
	_mm_store_ps(dest+56,m3);*/
	_mm_store_ps(temp, m0);
	_mm_store_ps(temp + 4, m1);
	_mm_store_ps(temp + 8, m2);
	_mm_store_ps(temp + 12, m3);

	m0 = _mm_load_ps(src + 32);
	m1 = _mm_load_ps(src + 40);
	m2 = _mm_load_ps(src + 48);
	m3 = _mm_load_ps(src + 56);
	_MM_TRANSPOSE4_PS(m0, m1, m2, m3);
	_mm_store_ps(src + 4, m0);
	_mm_store_ps(src + 12, m1);
	_mm_store_ps(src + 20, m2);
	_mm_store_ps(src + 28, m3);

	memcpy(src + 32, temp, sizeof(float) * 4);
	memcpy(src + 40, temp + 4, sizeof(float) * 4);
	memcpy(src + 48, temp + 8, sizeof(float) * 4);
	memcpy(src + 56, temp + 12, sizeof(float) * 4);

	m0 = _mm_load_ps(src + 36);
	m1 = _mm_load_ps(src + 44);
	m2 = _mm_load_ps(src + 52);
	m3 = _mm_load_ps(src + 60);
	_MM_TRANSPOSE4_PS(m0, m1, m2, m3);
	_mm_store_ps(src + 36, m0);
	_mm_store_ps(src + 44, m1);
	_mm_store_ps(src + 52, m2);
	_mm_store_ps(src + 60, m3);
}

void transpose16x16(float* src)
{
	__declspec(align(16)) float temp[64];
	__declspec(align(16)) float tmp[64];
	int sz = sizeof(float) * 8;
	for (int i = 0; i < 8; i++)
	{
		memcpy(temp + 8 * i, src + 16 * i, sz);
	}
	transpose8x8(temp);
	for (int i = 0; i < 8; i++)
	{
		memcpy(src + 16 * i, temp + 8 * i, sz);
	}

	for (int i = 0; i < 8; i++)
	{
		memcpy(tmp + 8 * i, src + 16 * i + 8, sz);
		memcpy(temp + 8 * i, src + 16 * (i + 8), sz);
	}
	transpose8x8(tmp);
	transpose8x8(temp);
	for (int i = 0; i < 8; i++)
	{
		memcpy(src + 16 * i + 8, temp + 8 * i, sz);
		memcpy(src + 16 * (i + 8), tmp + 8 * i, sz);
	}

	for (int i = 0; i < 8; i++)
	{
		memcpy(temp + 8 * i, src + 16 * (i + 8) + 8, sz);
	}
	transpose8x8(temp);
	for (int i = 0; i < 8; i++)
	{
		memcpy(src + 16 * (i + 8) + 8, temp + 8 * i, sz);
	}
}


///////////////////////////////////////////////////////////////////////////////////////////////////
//DCT simd functions///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
void dct4x4_1d_llm_fwd_sse(float* s, float* d);//8add, 4 mul
void dct4x4_1d_llm_inv_sse(float* s, float* d);

void dct4x4_llm_sse(float* a, float* b, float* temp, int flag);
void fDCT4x4_32f_and_threshold_and_iDCT4x4_32f(float* s, float threshold);
int  fDCT4x4_32f_and_threshold_and_iDCT4x4_nonzero_32f(float* s, float threshold);

void fDCT8x8_32f(const float* s, float* d, float* temp);
void iDCT8x8_32f(const float* s, float* d, float* temp);

void fDCT8x8_32f_and_threshold(const float* s, float* d, float threshold, float* temp);

void fDCT8x8_32f_and_threshold_and_iDCT8x8_32f(float* s, float threshold);
int  fDCT8x8_32f_and_threshold_and_iDCT8x8_nonzero_32f(float* s, float threshold);

void iDCT8x8_32f(float* s);
void fDCT8x8_32f_and_threshold(float* s, float threshold);

void fDCT2x2_2pack_32f_and_thresh_and_iDCT2x2_2pack(float* src, float* dest, float thresh);


void fDCT2D8x4_and_threshold_keep00_32f(const float* x, float* y, float thresh)
{
	const int __declspec(align(16)) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
	const __m128 mth = _mm_set1_ps(thresh);
	const __m128 zeros = _mm_setzero_ps();

	__m128 c0 = _mm_load_ps(x);
	__m128 c1 = _mm_load_ps(x + 56);
	__m128 t0 = _mm_add_ps(c0, c1);
	__m128 t7 = _mm_sub_ps(c0, c1);

	c1 = _mm_load_ps(x + 48);
	c0 = _mm_load_ps(x + 8);
	__m128 t1 = _mm_add_ps(c0, c1);
	__m128 t6 = _mm_sub_ps(c0, c1);

	c1 = _mm_load_ps(x + 40);
	c0 = _mm_load_ps(x + 16);
	__m128 t2 = _mm_add_ps(c0, c1);
	__m128 t5 = _mm_sub_ps(c0, c1);

	c0 = _mm_load_ps(x + 24);
	c1 = _mm_load_ps(x + 32);
	__m128 t3 = _mm_add_ps(c0, c1);
	__m128 t4 = _mm_sub_ps(c0, c1);

	/*
	c1 = x[0]; c2 = x[7]; t0 = c1 + c2; t7 = c1 - c2;
	c1 = x[1]; c2 = x[6]; t1 = c1 + c2; t6 = c1 - c2;
	c1 = x[2]; c2 = x[5]; t2 = c1 + c2; t5 = c1 - c2;
	c1 = x[3]; c2 = x[4]; t3 = c1 + c2; t4 = c1 - c2;
	*/

	c0 = _mm_add_ps(t0, t3);
	__m128 c3 = _mm_sub_ps(t0, t3);
	c1 = _mm_add_ps(t1, t2);
	__m128 c2 = _mm_sub_ps(t1, t2);

	/*
	c0 = t0 + t3; c3 = t0 - t3;
	c1 = t1 + t2; c2 = t1 - t2;
	*/

	const __m128 invsqrt2h = _mm_set_ps1(0.353554f);

	__m128 v = _mm_mul_ps(_mm_add_ps(c0, c1), invsqrt2h);
	__m128 msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	// keep 00 coef.
	__m128 v2 = _mm_blendv_ps(zeros, v, msk);
	v2 = _mm_blend_ps(v2, v, 1);
	_mm_store_ps(y, v2);

	v = _mm_mul_ps(_mm_sub_ps(c0, c1), invsqrt2h);
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(y + 32, v);

	/*y[0] = c0 + c1;
	y[4] = c0 - c1;*/

	__m128 w0 = _mm_set_ps1(0.541196f);
	__m128 w1 = _mm_set_ps1(1.306563f);
	v = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(w0, c2), _mm_mul_ps(w1, c3)), invsqrt2h);
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(y + 16, v);

	v = _mm_mul_ps(_mm_sub_ps(_mm_mul_ps(w0, c3), _mm_mul_ps(w1, c2)), invsqrt2h);
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(y + 48, v);
	/*
	y[2] = c2 * r[6] + c3 * r[2];
	y[6] = c3 * r[6] - c2 * r[2];
	*/

	w0 = _mm_set_ps1(1.175876f);
	w1 = _mm_set_ps1(0.785695f);
	c3 = _mm_add_ps(_mm_mul_ps(w0, t4), _mm_mul_ps(w1, t7));
	c0 = _mm_sub_ps(_mm_mul_ps(w0, t7), _mm_mul_ps(w1, t4));
	/*
	c3 = t4 * r[3] + t7 * r[5];
	c0 = t7 * r[3] - t4 * r[5];
	*/

	w0 = _mm_set_ps1(1.387040f);
	w1 = _mm_set_ps1(0.275899f);
	c2 = _mm_add_ps(_mm_mul_ps(w0, t5), _mm_mul_ps(w1, t6));
	c1 = _mm_sub_ps(_mm_mul_ps(w0, t6), _mm_mul_ps(w1, t5));
	/*
	c2 = t5 * r[1] + t6 * r[7];
	c1 = t6 * r[1] - t5 * r[7];
	*/

	v = _mm_mul_ps(_mm_sub_ps(c0, c2), invsqrt2h);
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);

	_mm_store_ps(y + 24, v);

	v = _mm_mul_ps(_mm_sub_ps(c3, c1), invsqrt2h);
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(y + 40, v);
	//y[5] = c3 - c1; y[3] = c0 - c2;

	const __m128 invsqrt2 = _mm_set_ps1(0.707107f);
	c0 = _mm_mul_ps(_mm_add_ps(c0, c2), invsqrt2);
	c3 = _mm_mul_ps(_mm_add_ps(c3, c1), invsqrt2);
	//c0 = (c0 + c2) * invsqrt2;
	//c3 = (c3 + c1) * invsqrt2;

	v = _mm_mul_ps(_mm_add_ps(c0, c3), invsqrt2h);
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(y + 8, v);

	v = _mm_mul_ps(_mm_sub_ps(c0, c3), invsqrt2h);
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);

	_mm_store_ps(y + 56, v);
	//y[1] = c0 + c3; y[7] = c0 - c3;

	/*for(i = 0;i < 8;i++)
	{
	y[i] *= invsqrt2h;
	}*/
}
void fDCT2D8x4_and_threshold_32f(const float* x, float* y, float thresh)
{
	const int __declspec(align(16)) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
	const __m128 mth = _mm_set1_ps(thresh);
	const __m128 zeros = _mm_setzero_ps();

	__m128 c0 = _mm_load_ps(x);
	__m128 c1 = _mm_load_ps(x + 56);
	__m128 t0 = _mm_add_ps(c0, c1);
	__m128 t7 = _mm_sub_ps(c0, c1);

	c1 = _mm_load_ps(x + 48);
	c0 = _mm_load_ps(x + 8);
	__m128 t1 = _mm_add_ps(c0, c1);
	__m128 t6 = _mm_sub_ps(c0, c1);

	c1 = _mm_load_ps(x + 40);
	c0 = _mm_load_ps(x + 16);
	__m128 t2 = _mm_add_ps(c0, c1);
	__m128 t5 = _mm_sub_ps(c0, c1);

	c0 = _mm_load_ps(x + 24);
	c1 = _mm_load_ps(x + 32);
	__m128 t3 = _mm_add_ps(c0, c1);
	__m128 t4 = _mm_sub_ps(c0, c1);

	/*
	c1 = x[0]; c2 = x[7]; t0 = c1 + c2; t7 = c1 - c2;
	c1 = x[1]; c2 = x[6]; t1 = c1 + c2; t6 = c1 - c2;
	c1 = x[2]; c2 = x[5]; t2 = c1 + c2; t5 = c1 - c2;
	c1 = x[3]; c2 = x[4]; t3 = c1 + c2; t4 = c1 - c2;
	*/

	c0 = _mm_add_ps(t0, t3);
	__m128 c3 = _mm_sub_ps(t0, t3);
	c1 = _mm_add_ps(t1, t2);
	__m128 c2 = _mm_sub_ps(t1, t2);

	/*
	c0 = t0 + t3; c3 = t0 - t3;
	c1 = t1 + t2; c2 = t1 - t2;
	*/

	const __m128 invsqrt2h = _mm_set_ps1(0.353554f);

	__m128 v = _mm_mul_ps(_mm_add_ps(c0, c1), invsqrt2h);
	__m128 msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);

	_mm_store_ps(y, v);

	v = _mm_mul_ps(_mm_sub_ps(c0, c1), invsqrt2h);
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(y + 32, v);

	/*y[0] = c0 + c1;
	y[4] = c0 - c1;*/

	__m128 w0 = _mm_set_ps1(0.541196f);
	__m128 w1 = _mm_set_ps1(1.306563f);
	v = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(w0, c2), _mm_mul_ps(w1, c3)), invsqrt2h);
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(y + 16, v);

	v = _mm_mul_ps(_mm_sub_ps(_mm_mul_ps(w0, c3), _mm_mul_ps(w1, c2)), invsqrt2h);
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(y + 48, v);
	/*
	y[2] = c2 * r[6] + c3 * r[2];
	y[6] = c3 * r[6] - c2 * r[2];
	*/

	w0 = _mm_set_ps1(1.175876f);
	w1 = _mm_set_ps1(0.785695f);
	c3 = _mm_add_ps(_mm_mul_ps(w0, t4), _mm_mul_ps(w1, t7));
	c0 = _mm_sub_ps(_mm_mul_ps(w0, t7), _mm_mul_ps(w1, t4));
	/*
	c3 = t4 * r[3] + t7 * r[5];
	c0 = t7 * r[3] - t4 * r[5];
	*/

	w0 = _mm_set_ps1(1.387040f);
	w1 = _mm_set_ps1(0.275899f);
	c2 = _mm_add_ps(_mm_mul_ps(w0, t5), _mm_mul_ps(w1, t6));
	c1 = _mm_sub_ps(_mm_mul_ps(w0, t6), _mm_mul_ps(w1, t5));
	/*
	c2 = t5 * r[1] + t6 * r[7];
	c1 = t6 * r[1] - t5 * r[7];
	*/

	v = _mm_mul_ps(_mm_sub_ps(c0, c2), invsqrt2h);
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);

	_mm_store_ps(y + 24, v);

	v = _mm_mul_ps(_mm_sub_ps(c3, c1), invsqrt2h);
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(y + 40, v);
	//y[5] = c3 - c1; y[3] = c0 - c2;

	const __m128 invsqrt2 = _mm_set_ps1(0.707107f);
	c0 = _mm_mul_ps(_mm_add_ps(c0, c2), invsqrt2);
	c3 = _mm_mul_ps(_mm_add_ps(c3, c1), invsqrt2);
	//c0 = (c0 + c2) * invsqrt2;
	//c3 = (c3 + c1) * invsqrt2;

	v = _mm_mul_ps(_mm_add_ps(c0, c3), invsqrt2h);
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);

	_mm_store_ps(y + 8, v);

	v = _mm_mul_ps(_mm_sub_ps(c0, c3), invsqrt2h);
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);

	_mm_store_ps(y + 56, v);
	//y[1] = c0 + c3; y[7] = c0 - c3;

	/*for(i = 0;i < 8;i++)
	{
	y[i] *= invsqrt2h;
	}*/
}


void fDCT2D8x4noscale_32f(const float* x, float* y)
{
	__m128 c0 = _mm_load_ps(x);
	__m128 c1 = _mm_load_ps(x + 56);
	__m128 t0 = _mm_add_ps(c0, c1);
	__m128 t7 = _mm_sub_ps(c0, c1);

	c1 = _mm_load_ps(x + 48);
	c0 = _mm_load_ps(x + 8);
	__m128 t1 = _mm_add_ps(c0, c1);
	__m128 t6 = _mm_sub_ps(c0, c1);

	c1 = _mm_load_ps(x + 40);
	c0 = _mm_load_ps(x + 16);
	__m128 t2 = _mm_add_ps(c0, c1);
	__m128 t5 = _mm_sub_ps(c0, c1);

	c0 = _mm_load_ps(x + 24);
	c1 = _mm_load_ps(x + 32);
	__m128 t3 = _mm_add_ps(c0, c1);
	__m128 t4 = _mm_sub_ps(c0, c1);

	/*
	c1 = x[0]; c2 = x[7]; t0 = c1 + c2; t7 = c1 - c2;
	c1 = x[1]; c2 = x[6]; t1 = c1 + c2; t6 = c1 - c2;
	c1 = x[2]; c2 = x[5]; t2 = c1 + c2; t5 = c1 - c2;
	c1 = x[3]; c2 = x[4]; t3 = c1 + c2; t4 = c1 - c2;
	*/

	c0 = _mm_add_ps(t0, t3);
	__m128 c3 = _mm_sub_ps(t0, t3);
	c1 = _mm_add_ps(t1, t2);
	__m128 c2 = _mm_sub_ps(t1, t2);

	/*
	c0 = t0 + t3; c3 = t0 - t3;
	c1 = t1 + t2; c2 = t1 - t2;
	*/


	_mm_store_ps(y, _mm_add_ps(c0, c1));
	_mm_store_ps(y + 32, _mm_sub_ps(c0, c1));

	/*y[0] = c0 + c1;
	y[4] = c0 - c1;*/

	__m128 w0 = _mm_set_ps1(0.541196f);
	__m128 w1 = _mm_set_ps1(1.306563f);
	_mm_store_ps(y + 16, _mm_add_ps(_mm_mul_ps(w0, c2), _mm_mul_ps(w1, c3)));
	_mm_store_ps(y + 48, _mm_sub_ps(_mm_mul_ps(w0, c3), _mm_mul_ps(w1, c2)));
	/*
	y[2] = c2 * r[6] + c3 * r[2];
	y[6] = c3 * r[6] - c2 * r[2];
	*/

	w0 = _mm_set_ps1(1.175876f);
	w1 = _mm_set_ps1(0.785695f);
	c3 = _mm_add_ps(_mm_mul_ps(w0, t4), _mm_mul_ps(w1, t7));
	c0 = _mm_sub_ps(_mm_mul_ps(w0, t7), _mm_mul_ps(w1, t4));
	/*
	c3 = t4 * r[3] + t7 * r[5];
	c0 = t7 * r[3] - t4 * r[5];
	*/

	w0 = _mm_set_ps1(1.387040f);
	w1 = _mm_set_ps1(0.275899f);
	c2 = _mm_add_ps(_mm_mul_ps(w0, t5), _mm_mul_ps(w1, t6));
	c1 = _mm_sub_ps(_mm_mul_ps(w0, t6), _mm_mul_ps(w1, t5));
	/*
	c2 = t5 * r[1] + t6 * r[7];
	c1 = t6 * r[1] - t5 * r[7];
	*/

	_mm_store_ps(y + 24, _mm_sub_ps(c0, c2));
	_mm_store_ps(y + 40, _mm_sub_ps(c3, c1));
	//y[5] = c3 - c1; y[3] = c0 - c2;

	const __m128 invsqrt2 = _mm_set_ps1(0.707107f);
	c0 = _mm_mul_ps(_mm_add_ps(c0, c2), invsqrt2);
	c3 = _mm_mul_ps(_mm_add_ps(c3, c1), invsqrt2);
	//c0 = (c0 + c2) * invsqrt2;
	//c3 = (c3 + c1) * invsqrt2;

	_mm_store_ps(y + 8, _mm_add_ps(c0, c3));
	_mm_store_ps(y + 56, _mm_sub_ps(c0, c3));
	//y[1] = c0 + c3; y[7] = c0 - c3;

	/*for(i = 0;i < 8;i++)
	{
	y[i] *= invsqrt2h;
	}*/
}
void fDCT2D8x4_32f(const float* x, float* y)
{
	__m128 c0 = _mm_load_ps(x);
	__m128 c1 = _mm_load_ps(x + 56);
	__m128 t0 = _mm_add_ps(c0, c1);
	__m128 t7 = _mm_sub_ps(c0, c1);

	c1 = _mm_load_ps(x + 48);
	c0 = _mm_load_ps(x + 8);
	__m128 t1 = _mm_add_ps(c0, c1);
	__m128 t6 = _mm_sub_ps(c0, c1);

	c1 = _mm_load_ps(x + 40);
	c0 = _mm_load_ps(x + 16);
	__m128 t2 = _mm_add_ps(c0, c1);
	__m128 t5 = _mm_sub_ps(c0, c1);

	c0 = _mm_load_ps(x + 24);
	c1 = _mm_load_ps(x + 32);
	__m128 t3 = _mm_add_ps(c0, c1);
	__m128 t4 = _mm_sub_ps(c0, c1);

	/*
	c1 = x[0]; c2 = x[7]; t0 = c1 + c2; t7 = c1 - c2;
	c1 = x[1]; c2 = x[6]; t1 = c1 + c2; t6 = c1 - c2;
	c1 = x[2]; c2 = x[5]; t2 = c1 + c2; t5 = c1 - c2;
	c1 = x[3]; c2 = x[4]; t3 = c1 + c2; t4 = c1 - c2;
	*/

	c0 = _mm_add_ps(t0, t3);
	__m128 c3 = _mm_sub_ps(t0, t3);
	c1 = _mm_add_ps(t1, t2);
	__m128 c2 = _mm_sub_ps(t1, t2);

	/*
	c0 = t0 + t3; c3 = t0 - t3;
	c1 = t1 + t2; c2 = t1 - t2;
	*/

	const __m128 invsqrt2h = _mm_set_ps1(0.353554f);
	_mm_store_ps(y, _mm_mul_ps(_mm_add_ps(c0, c1), invsqrt2h));
	_mm_store_ps(y + 32, _mm_mul_ps(_mm_sub_ps(c0, c1), invsqrt2h));

	/*y[0] = c0 + c1;
	y[4] = c0 - c1;*/

	__m128 w0 = _mm_set_ps1(0.541196f);
	__m128 w1 = _mm_set_ps1(1.306563f);
	_mm_store_ps(y + 16, _mm_mul_ps(_mm_add_ps(_mm_mul_ps(w0, c2), _mm_mul_ps(w1, c3)), invsqrt2h));
	_mm_store_ps(y + 48, _mm_mul_ps(_mm_sub_ps(_mm_mul_ps(w0, c3), _mm_mul_ps(w1, c2)), invsqrt2h));
	/*
	y[2] = c2 * r[6] + c3 * r[2];
	y[6] = c3 * r[6] - c2 * r[2];
	*/

	w0 = _mm_set_ps1(1.175876f);
	w1 = _mm_set_ps1(0.785695f);
	c3 = _mm_add_ps(_mm_mul_ps(w0, t4), _mm_mul_ps(w1, t7));
	c0 = _mm_sub_ps(_mm_mul_ps(w0, t7), _mm_mul_ps(w1, t4));
	/*
	c3 = t4 * r[3] + t7 * r[5];
	c0 = t7 * r[3] - t4 * r[5];
	*/

	w0 = _mm_set_ps1(1.387040f);
	w1 = _mm_set_ps1(0.275899f);
	c2 = _mm_add_ps(_mm_mul_ps(w0, t5), _mm_mul_ps(w1, t6));
	c1 = _mm_sub_ps(_mm_mul_ps(w0, t6), _mm_mul_ps(w1, t5));
	/*
	c2 = t5 * r[1] + t6 * r[7];
	c1 = t6 * r[1] - t5 * r[7];
	*/

	_mm_store_ps(y + 24, _mm_mul_ps(_mm_sub_ps(c0, c2), invsqrt2h));
	_mm_store_ps(y + 40, _mm_mul_ps(_mm_sub_ps(c3, c1), invsqrt2h));
	//y[5] = c3 - c1; y[3] = c0 - c2;

	const __m128 invsqrt2 = _mm_set_ps1(0.707107f);
	c0 = _mm_mul_ps(_mm_add_ps(c0, c2), invsqrt2);
	c3 = _mm_mul_ps(_mm_add_ps(c3, c1), invsqrt2);
	//c0 = (c0 + c2) * invsqrt2;
	//c3 = (c3 + c1) * invsqrt2;

	_mm_store_ps(y + 8, _mm_mul_ps(_mm_add_ps(c0, c3), invsqrt2h));
	_mm_store_ps(y + 56, _mm_mul_ps(_mm_sub_ps(c0, c3), invsqrt2h));
	//y[1] = c0 + c3; y[7] = c0 - c3;

	/*for(i = 0;i < 8;i++)
	{
	y[i] *= invsqrt2h;
	}*/
}

void fDCT8x8_32f_and_threshold(const float* s, float* d, float threshold, float* temp)
{
	transpose8x8(s, temp);

	/*for (int j = 0; j < 8; j ++)
	{
	for (int i = 0; i < 8; i ++)
	{
	temp[8*i+j] =s[8*j+i];
	}
	}*/

	fDCT2D8x4_32f(temp, d);
	fDCT2D8x4_32f(temp + 4, d + 4);

	transpose8x8(d, temp);
	/*for (int j = 0; j < 8; j ++)
	{
	for (int i = 0; i < 8; i ++)
	{
	temp[8*i+j] =d[8*j+i];
	}
	}*/
	fDCT2D8x4_and_threshold_32f(temp, d, threshold);
	fDCT2D8x4_and_threshold_32f(temp + 4, d + 4, threshold);

}
void fDCT8x8_32f(const float* s, float* d, float* temp)
{
	//for (int j = 0; j < 8; j ++)
	//{
	//	for (int i = 0; i < 8; i ++)
	//	{
	//		temp[8*i+j] =s[8*j+i];
	//	}
	//}
	transpose8x8(s, temp);

	fDCT2D8x4_32f(temp, d);
	fDCT2D8x4_32f(temp + 4, d + 4);

	//for (int j = 0; j < 8; j ++)
	//{
	//	for (int i = 0; i < 8; i ++)
	//	{
	//		temp[8*i+j] =d[8*j+i];
	//	}
	//}
	transpose8x8(d, temp);
	fDCT2D8x4_32f(temp, d);
	fDCT2D8x4_32f(temp + 4, d + 4);
}

void fDCT1Dllm_32f(const float* x, float* y)
{
	float t0, t1, t2, t3, t4, t5, t6, t7; float c0, c1, c2, c3; float r[8]; int i;

	for (i = 0; i < 8; i++){ r[i] = (float)(cos((double)i / 16.0 * M_PI) * M_SQRT2); }
	const float invsqrt2 = 0.707107f;//(float)(1.0f / M_SQRT2);
	const float invsqrt2h = 0.353554f;//invsqrt2*0.5f;

	c1 = x[0]; c2 = x[7]; t0 = c1 + c2; t7 = c1 - c2;
	c1 = x[1]; c2 = x[6]; t1 = c1 + c2; t6 = c1 - c2;
	c1 = x[2]; c2 = x[5]; t2 = c1 + c2; t5 = c1 - c2;
	c1 = x[3]; c2 = x[4]; t3 = c1 + c2; t4 = c1 - c2;

	c0 = t0 + t3; c3 = t0 - t3;
	c1 = t1 + t2; c2 = t1 - t2;

	y[0] = c0 + c1;
	y[4] = c0 - c1;
	y[2] = c2 * r[6] + c3 * r[2];
	y[6] = c3 * r[6] - c2 * r[2];

	c3 = t4 * r[3] + t7 * r[5];
	c0 = t7 * r[3] - t4 * r[5];
	c2 = t5 * r[1] + t6 * r[7];
	c1 = t6 * r[1] - t5 * r[7];

	y[5] = c3 - c1; y[3] = c0 - c2;
	c0 = (c0 + c2) * invsqrt2;
	c3 = (c3 + c1) * invsqrt2;
	y[1] = c0 + c3; y[7] = c0 - c3;

	for (i = 0; i < 8; i++)
	{
		y[i] *= invsqrt2h;
	}
}

void fDCT2Dllm_32f(const float* s, float* d, float* temp)
{
	for (int j = 0; j < 8; j++)
	{
		fDCT1Dllm_32f(s + j * 8, temp + j * 8);
	}

	for (int j = 0; j < 8; j++)
	{
		for (int i = 0; i < 8; i++)
		{
			d[8 * i + j] = temp[8 * j + i];
		}
	}
	for (int j = 0; j < 8; j++)
	{
		fDCT1Dllm_32f(d + j * 8, temp + j * 8);
	}

	for (int j = 0; j < 8; j++)
	{
		for (int i = 0; i < 8; i++)
		{
			d[8 * i + j] = temp[8 * j + i];
		}
	}
}

void iDCT1Dllm_32f(const float* y, float* x)
{
	float a0, a1, a2, a3, b0, b1, b2, b3; float z0, z1, z2, z3, z4; float r[8]; int i;

	for (i = 0; i < 8; i++){ r[i] = (float)(cos((double)i / 16.0 * M_PI) * M_SQRT2); }

	z0 = y[1] + y[7]; z1 = y[3] + y[5]; z2 = y[3] + y[7]; z3 = y[1] + y[5];
	z4 = (z0 + z1) * r[3];

	z0 = z0 * (-r[3] + r[7]);
	z1 = z1 * (-r[3] - r[1]);
	z2 = z2 * (-r[3] - r[5]) + z4;
	z3 = z3 * (-r[3] + r[5]) + z4;

	b3 = y[7] * (-r[1] + r[3] + r[5] - r[7]) + z0 + z2;
	b2 = y[5] * (r[1] + r[3] - r[5] + r[7]) + z1 + z3;
	b1 = y[3] * (r[1] + r[3] + r[5] - r[7]) + z1 + z2;
	b0 = y[1] * (r[1] + r[3] - r[5] - r[7]) + z0 + z3;

	z4 = (y[2] + y[6]) * r[6];
	z0 = y[0] + y[4]; z1 = y[0] - y[4];
	z2 = z4 - y[6] * (r[2] + r[6]);
	z3 = z4 + y[2] * (r[2] - r[6]);
	a0 = z0 + z3; a3 = z0 - z3;
	a1 = z1 + z2; a2 = z1 - z2;

	x[0] = a0 + b0; x[7] = a0 - b0;
	x[1] = a1 + b1; x[6] = a1 - b1;
	x[2] = a2 + b2; x[5] = a2 - b2;
	x[3] = a3 + b3; x[4] = a3 - b3;

	for (i = 0; i < 8; i++){ x[i] *= 0.353554f; }
}

void iDCT2Dllm_32f(const float* s, float* d, float* temp)
{
	for (int j = 0; j < 8; j++)
	{
		iDCT1Dllm_32f(s + j * 8, temp + j * 8);
	}

	for (int j = 0; j < 8; j++)
	{
		for (int i = 0; i < 8; i++)
		{
			d[8 * i + j] = temp[8 * j + i];
		}
	}
	for (int j = 0; j < 8; j++)
	{
		iDCT1Dllm_32f(d + j * 8, temp + j * 8);
	}

	for (int j = 0; j < 8; j++)
	{
		for (int i = 0; i < 8; i++)
		{
			d[8 * i + j] = temp[8 * j + i];
		}
	}
}

void iDCT2D8x4_32f(const float* y, float* x)
{
	/*
	float a0,a1,a2,a3,b0,b1,b2,b3; float z0,z1,z2,z3,z4; float r[8]; int i;
	for(i = 0;i < 8;i++){ r[i] = (float)(cos((double)i / 16.0 * M_PI) * M_SQRT2); }
	*/
	/*
	0: 1.414214
	1: 1.387040
	2: 1.306563
	3:
	4: 1.000000
	5: 0.785695
	6:
	7: 0.275899
	*/
	__m128 my1 = _mm_load_ps(y + 8);
	__m128 my7 = _mm_load_ps(y + 56);
	__m128 mz0 = _mm_add_ps(my1, my7);

	__m128 my3 = _mm_load_ps(y + 24);
	__m128 mz2 = _mm_add_ps(my3, my7);
	__m128 my5 = _mm_load_ps(y + 40);
	__m128 mz1 = _mm_add_ps(my3, my5);
	__m128 mz3 = _mm_add_ps(my1, my5);

	__m128 w = _mm_set1_ps(1.175876f);
	__m128 mz4 = _mm_mul_ps(_mm_add_ps(mz0, mz1), w);
	//z0 = y[1] + y[7]; z1 = y[3] + y[5]; z2 = y[3] + y[7]; z3 = y[1] + y[5];
	//z4 = (z0 + z1) * r[3];

	w = _mm_set1_ps(-1.961571f);
	mz2 = _mm_add_ps(_mm_mul_ps(mz2, w), mz4);
	w = _mm_set1_ps(-0.390181f);
	mz3 = _mm_add_ps(_mm_mul_ps(mz3, w), mz4);
	w = _mm_set1_ps(-0.899976f);
	mz0 = _mm_mul_ps(mz0, w);
	w = _mm_set1_ps(-2.562915f);
	mz1 = _mm_mul_ps(mz1, w);


	/*
	-0.899976
	-2.562915
	-1.961571
	-0.390181
	z0 = z0 * (-r[3] + r[7]);
	z1 = z1 * (-r[3] - r[1]);
	z2 = z2 * (-r[3] - r[5]) + z4;
	z3 = z3 * (-r[3] + r[5]) + z4;*/

	w = _mm_set1_ps(0.298631f);
	__m128 mb3 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(my7, w), mz0), mz2);
	w = _mm_set1_ps(2.053120f);
	__m128 mb2 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(my5, w), mz1), mz3);
	w = _mm_set1_ps(3.072711f);
	__m128 mb1 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(my3, w), mz1), mz2);
	w = _mm_set1_ps(1.501321f);
	__m128 mb0 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(my1, w), mz0), mz3);
	/*
	0.298631
	2.053120
	3.072711
	1.501321
	b3 = y[7] * (-r[1] + r[3] + r[5] - r[7]) + z0 + z2;
	b2 = y[5] * ( r[1] + r[3] - r[5] + r[7]) + z1 + z3;
	b1 = y[3] * ( r[1] + r[3] + r[5] - r[7]) + z1 + z2;
	b0 = y[1] * ( r[1] + r[3] - r[5] - r[7]) + z0 + z3;
	*/

	__m128 my2 = _mm_load_ps(y + 16);
	__m128 my6 = _mm_load_ps(y + 48);
	w = _mm_set1_ps(0.541196f);
	mz4 = _mm_mul_ps(_mm_add_ps(my2, my6), w);
	__m128 my0 = _mm_load_ps(y);
	__m128 my4 = _mm_load_ps(y + 32);
	mz0 = _mm_add_ps(my0, my4);
	mz1 = _mm_sub_ps(my0, my4);


	w = _mm_set1_ps(-1.847759f);
	mz2 = _mm_add_ps(mz4, _mm_mul_ps(my6, w));
	w = _mm_set1_ps(0.765367f);
	mz3 = _mm_add_ps(mz4, _mm_mul_ps(my2, w));

	my0 = _mm_add_ps(mz0, mz3);
	my3 = _mm_sub_ps(mz0, mz3);
	my1 = _mm_add_ps(mz1, mz2);
	my2 = _mm_sub_ps(mz1, mz2);
	/*
	1.847759
	0.765367
	z4 = (y[2] + y[6]) * r[6];
	z0 = y[0] + y[4]; z1 = y[0] - y[4];
	z2 = z4 - y[6] * (r[2] + r[6]);
	z3 = z4 + y[2] * (r[2] - r[6]);
	a0 = z0 + z3; a3 = z0 - z3;
	a1 = z1 + z2; a2 = z1 - z2;
	*/

	w = _mm_set1_ps(0.353554f);
	_mm_store_ps(x, _mm_mul_ps(w, _mm_add_ps(my0, mb0)));
	_mm_store_ps(x + 56, _mm_mul_ps(w, _mm_sub_ps(my0, mb0)));
	_mm_store_ps(x + 8, _mm_mul_ps(w, _mm_add_ps(my1, mb1)));
	_mm_store_ps(x + 48, _mm_mul_ps(w, _mm_sub_ps(my1, mb1)));
	_mm_store_ps(x + 16, _mm_mul_ps(w, _mm_add_ps(my2, mb2)));
	_mm_store_ps(x + 40, _mm_mul_ps(w, _mm_sub_ps(my2, mb2)));
	_mm_store_ps(x + 24, _mm_mul_ps(w, _mm_add_ps(my3, mb3)));
	_mm_store_ps(x + 32, _mm_mul_ps(w, _mm_sub_ps(my3, mb3)));
	/*
	x[0] = a0 + b0; x[7] = a0 - b0;
	x[1] = a1 + b1; x[6] = a1 - b1;
	x[2] = a2 + b2; x[5] = a2 - b2;
	x[3] = a3 + b3; x[4] = a3 - b3;
	for(i = 0;i < 8;i++){ x[i] *= 0.353554f; }
	*/
}



void iDCT8x8_32f(const float* s, float* d, float* temp)
{
	transpose8x8((float*)s, temp);
	//for (int j = 0; j < 8; j ++)
	//{
	//	for (int i = 0; i < 8; i ++)
	//	{
	//		temp[8*i+j] =s[8*j+i];
	//	}
	//}
	iDCT2D8x4_32f(temp, d);
	iDCT2D8x4_32f(temp + 4, d + 4);

	transpose8x8(d, temp);
	/*for (int j = 0; j < 8; j ++)
	{
	for (int i = 0; i < 8; i ++)
	{
	temp[8*i+j] =d[8*j+i];
	}
	}*/
	iDCT2D8x4_32f(temp, d);
	iDCT2D8x4_32f(temp + 4, d + 4);
}



#ifdef UNDERCONSTRUCTION_____
//internal simd using sse3
void LLMDCTOpt(const float* x, float* y)
{
	float t4, t5, t6, t7; float c0, c1, c2, c3;
	float* r = dct_tbl;

	const float invsqrt2 = 0.707107f;//(float)(1.0f / M_SQRT2);
	const float invsqrt2h = 0.353554f;//invsqrt2*0.5f;

	{
		__m128 mc1 = _mm_load_ps(x);
		__m128 mc2 = _mm_loadr_ps(x + 4);

		__m128 mt1 = _mm_add_ps(mc1, mc2);
		__m128 mt2 = _mm_sub_ps(mc1, mc2);//rev

		mc1 = _mm_addsub_ps(_mm_shuffle_ps(mt1, mt1, _MM_SHUFFLE(1, 1, 0, 0)), _mm_shuffle_ps(mt1, mt1, _MM_SHUFFLE(2, 2, 3, 3)));
		mc1 = _mm_shuffle_ps(mc1, mc1, _MM_SHUFFLE(0, 2, 3, 1));

		_mm_store_ps(y, mc1);
		_mm_store_ps(y + 4, mt2);

	}
	c0 = y[0];
	c1 = y[1];
	c2 = y[2];
	c3 = y[3];
	/*c3=y[0];
	c0=y[1];
	c2=y[2];
	c1=y[3];*/

	t7 = y[4];
	t6 = y[5];
	t5 = y[6];
	t4 = y[7];

	y[0] = c0 + c1;
	y[4] = c0 - c1;
	y[2] = c2 * r[6] + c3 * r[2];
	y[6] = c3 * r[6] - c2 * r[2];

	c3 = t4 * r[3] + t7 * r[5];
	c0 = t7 * r[3] - t4 * r[5];
	c2 = t5 * r[1] + t6 * r[7];
	c1 = t6 * r[1] - t5 * r[7];

	y[5] = c3 - c1; y[3] = c0 - c2;
	c0 = (c0 + c2) * invsqrt2;
	c3 = (c3 + c1) * invsqrt2;
	y[1] = c0 + c3; y[7] = c0 - c3;

	const __m128 invsqh = _mm_set_ps1(invsqrt2h);
	__m128 my = _mm_load_ps(y);
	_mm_store_ps(y, _mm_mul_ps(my, invsqh));

	my = _mm_load_ps(y + 4);
	_mm_store_ps(y + 4, _mm_mul_ps(my, invsqh));
}
#endif

void fDCT8x8_32f_and_threshold_and_iDCT8x8_32f(float* s, float threshold)
{
	fDCT2D8x4_32f(s, s);
	fDCT2D8x4_32f(s + 4, s + 4);
	transpose8x8(s);
#ifdef _KEEP_00_COEF_
	fDCT2D8x4_and_threshold_keep00_32f(s, s, threshold);
#else
	fDCT2D8x4_and_threshold_32f(s, s, threshold);
#endif
	fDCT2D8x4_and_threshold_32f(s + 4, s + 4, threshold);
	//ommiting transform
	//transpose8x8(s);
	//transpose8x8(s);
	iDCT2D8x4_32f(s, s);
	iDCT2D8x4_32f(s + 4, s + 4);
	transpose8x8(s);
	iDCT2D8x4_32f(s, s);
	iDCT2D8x4_32f(s + 4, s + 4);

	return;
}


inline int getNonzero(float* s, int size)
{
	int ret = 0;
	for (int i = 0; i < size; i++)
	{
		if (s[i] != 0.f)ret++;
	}

	return ret;
}

int fDCT8x8_32f_and_threshold_and_iDCT8x8_nonzero_32f(float* s, float threshold)
{
	fDCT2D8x4_32f(s, s);
	fDCT2D8x4_32f(s + 4, s + 4);
	transpose8x8(s);
#ifdef _KEEP_00_COEF_
	fDCT2D8x4_and_threshold_keep00_32f(s, s, threshold);
#else
	fDCT2D8x4_and_threshold_32f(s, s, threshold);
#endif
	fDCT2D8x4_and_threshold_32f(s + 4, s + 4, threshold);
	int ret = getNonzero(s, 64);
	//ommiting transform
	//transpose8x8(s);
	//transpose8x8(s);
	iDCT2D8x4_32f(s, s);
	iDCT2D8x4_32f(s + 4, s + 4);
	transpose8x8(s);
	iDCT2D8x4_32f(s, s);
	iDCT2D8x4_32f(s + 4, s + 4);

	return ret;
}


//2x2

void dct1d2_32f(float* src, float* dest)
{
	dest[0] = 0.7071067812f*(src[0] + src[1]);
	dest[1] = 0.7071067812f*(src[0] - src[1]);
}

void fDCT2x2_2pack_32f_and_thresh_and_iDCT2x2_2pack(float* src, float* dest, float thresh)
{
	__m128 ms0 = _mm_load_ps(src);
	__m128 ms1 = _mm_load_ps(src + 4);
	const __m128 mm = _mm_set1_ps(0.5f);
	__m128 a = _mm_add_ps(ms0, ms1);
	__m128 b = _mm_sub_ps(ms0, ms1);

	__m128 t1 = _mm_unpacklo_ps(a, b);
	__m128 t2 = _mm_unpackhi_ps(a, b);
	ms0 = _mm_shuffle_ps(t1, t2, _MM_SHUFFLE(1, 0, 1, 0));
	ms1 = _mm_shuffle_ps(t1, t2, _MM_SHUFFLE(3, 2, 3, 2));

	a = _mm_mul_ps(mm, _mm_add_ps(ms0, ms1));
	b = _mm_mul_ps(mm, _mm_sub_ps(ms0, ms1));

	const int __declspec(align(16)) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
	const __m128 mth = _mm_set1_ps(thresh);

	__m128 msk = _mm_cmpgt_ps(_mm_and_ps(a, *(const __m128*)v32f_absmask), mth);
	ms0 = _mm_blendv_ps(_mm_setzero_ps(), a, msk);
#ifdef _KEEP_00_COEF_
	ms0 = _mm_blend_ps(ms0, a, 1);
#endif
	msk = _mm_cmpgt_ps(_mm_and_ps(b, *(const __m128*)v32f_absmask), mth);
	ms1 = _mm_blendv_ps(_mm_setzero_ps(), b, msk);

	a = _mm_add_ps(ms0, ms1);
	b = _mm_sub_ps(ms0, ms1);

	t1 = _mm_unpacklo_ps(a, b);
	t2 = _mm_unpackhi_ps(a, b);
	ms0 = _mm_shuffle_ps(t1, t2, _MM_SHUFFLE(1, 0, 1, 0));
	ms1 = _mm_shuffle_ps(t1, t2, _MM_SHUFFLE(3, 2, 3, 2));

	a = _mm_mul_ps(mm, _mm_add_ps(ms0, ms1));
	b = _mm_mul_ps(mm, _mm_sub_ps(ms0, ms1));

	_mm_store_ps(dest, a);
	_mm_store_ps(dest + 4, b);
}

void DCT2x2_32f(float* src, float* dest, float* temp)
{
	dct1d2_32f(src, temp);
	dct1d2_32f(src + 2, temp + 2);
	float v = temp[1];
	temp[1] = temp[2];
	temp[2] = v;

	dct1d2_32f(temp, dest);
	dct1d2_32f(temp + 2, dest + 2);

	v = dest[1];
	dest[1] = dest[2];
	dest[2] = v;
}

#define fDCT2x2_32f DCT2x2_32f
#define iDCT2x2_32f DCT2x2_32f

void dct1d2_32f_and_thresh(float* src, float* dest, float thresh)
{
	float v = 0.7071068f*(src[0] + src[1]);
	dest[0] = (abs(v) < thresh) ? 0.f : v;
	v = 0.7071068f*(src[0] - src[1]);
	dest[1] = (abs(v) < thresh) ? 0 : v;
}
void fDCT2x2_32f_and_threshold(float* src, float* dest, float* temp, float thresh)
{
	dct1d2_32f(src, temp);
	dct1d2_32f(src + 2, temp + 2);
	float v = temp[1];
	temp[1] = temp[2];
	temp[2] = v;

	dct1d2_32f_and_thresh(temp, dest, thresh);
	dct1d2_32f_and_thresh(temp + 2, dest + 2, thresh);
	//dct1d2_32f(temp,dest);
	//dct1d2_32f(temp+2,dest+2);

	v = dest[1];
	dest[1] = dest[2];
	dest[2] = v;
}

void dct4x4_1d_llm_fwd_sse(float* s, float* d)//8add, 4 mul
{
	const __m128 c2 = _mm_set1_ps(1.30656f);//cos(CV_PI*2/16.0)*sqrt(2);
	const __m128 c6 = _mm_set1_ps(0.541196f);//cos(CV_PI*6/16.0)*sqrt(2);

	__m128 s0 = _mm_load_ps(s); s += 4;
	__m128 s1 = _mm_load_ps(s); s += 4;
	__m128 s2 = _mm_load_ps(s); s += 4;
	__m128 s3 = _mm_load_ps(s);

	__m128 p03 = _mm_add_ps(s0, s3);
	__m128 p12 = _mm_add_ps(s1, s2);
	__m128 m03 = _mm_sub_ps(s0, s3);
	__m128 m12 = _mm_sub_ps(s1, s2);

	_mm_store_ps(d, _mm_add_ps(p03, p12));
	_mm_store_ps(d + 4, _mm_add_ps(_mm_mul_ps(c2, m03), _mm_mul_ps(c6, m12)));
	_mm_store_ps(d + 8, _mm_sub_ps(p03, p12));
	_mm_store_ps(d + 12, _mm_sub_ps(_mm_mul_ps(c6, m03), _mm_mul_ps(c2, m12)));
}

void dct4x4_1d_llm_fwd_sse_and_transpose(float* s, float* d)//8add, 4 mul
{
	const __m128 c2 = _mm_set1_ps(1.30656f);//cos(CV_PI*2/16.0)*sqrt(2);
	const __m128 c6 = _mm_set1_ps(0.541196f);//cos(CV_PI*6/16.0)*sqrt(2);

	__m128 s0 = _mm_load_ps(s); s += 4;
	__m128 s1 = _mm_load_ps(s); s += 4;
	__m128 s2 = _mm_load_ps(s); s += 4;
	__m128 s3 = _mm_load_ps(s);

	__m128 p03 = _mm_add_ps(s0, s3);
	__m128 p12 = _mm_add_ps(s1, s2);
	__m128 m03 = _mm_sub_ps(s0, s3);
	__m128 m12 = _mm_sub_ps(s1, s2);

	s0 = _mm_add_ps(p03, p12);
	s1 = _mm_add_ps(_mm_mul_ps(c2, m03), _mm_mul_ps(c6, m12));
	s2 = _mm_sub_ps(p03, p12);
	s3 = _mm_sub_ps(_mm_mul_ps(c6, m03), _mm_mul_ps(c2, m12));
	_MM_TRANSPOSE4_PS(s0, s1, s2, s3);
	_mm_store_ps(d, s0);
	_mm_store_ps(d + 4, s1);
	_mm_store_ps(d + 8, s2);
	_mm_store_ps(d + 12, s3);
}

void dct4x4_1d_llm_inv_sse(float* s, float* d)
{
	const __m128 c2 = _mm_set1_ps(1.30656f);//cos(CV_PI*2/16.0)*sqrt(2);
	const __m128 c6 = _mm_set1_ps(0.541196f);//cos(CV_PI*6/16.0)*sqrt(2);

	__m128 s0 = _mm_load_ps(s); s += 4;
	__m128 s1 = _mm_load_ps(s); s += 4;
	__m128 s2 = _mm_load_ps(s); s += 4;
	__m128 s3 = _mm_load_ps(s);

	__m128 t10 = _mm_add_ps(s0, s2);
	__m128 t12 = _mm_sub_ps(s0, s2);

	__m128 t0 = _mm_add_ps(_mm_mul_ps(c2, s1), _mm_mul_ps(c6, s3));
	__m128 t2 = _mm_sub_ps(_mm_mul_ps(c6, s1), _mm_mul_ps(c2, s3));

	_mm_store_ps(d, _mm_add_ps(t10, t0));
	_mm_store_ps(d + 4, _mm_add_ps(t12, t2));
	_mm_store_ps(d + 8, _mm_sub_ps(t12, t2));
	_mm_store_ps(d + 12, _mm_sub_ps(t10, t0));
}

void dct4x4_1d_llm_inv_sse_and_transpose(float* s, float* d)
{
	const __m128 c2 = _mm_set1_ps(1.30656f);//cos(CV_PI*2/16.0)*sqrt(2);
	const __m128 c6 = _mm_set1_ps(0.541196f);//cos(CV_PI*6/16.0)*sqrt(2);

	__m128 s0 = _mm_load_ps(s); s += 4;
	__m128 s1 = _mm_load_ps(s); s += 4;
	__m128 s2 = _mm_load_ps(s); s += 4;
	__m128 s3 = _mm_load_ps(s);

	__m128 t10 = _mm_add_ps(s0, s2);
	__m128 t12 = _mm_sub_ps(s0, s2);

	__m128 t0 = _mm_add_ps(_mm_mul_ps(c2, s1), _mm_mul_ps(c6, s3));
	__m128 t2 = _mm_sub_ps(_mm_mul_ps(c6, s1), _mm_mul_ps(c2, s3));

	s0 = _mm_add_ps(t10, t0);
	s1 = _mm_add_ps(t12, t2);
	s2 = _mm_sub_ps(t12, t2);
	s3 = _mm_sub_ps(t10, t0);
	_MM_TRANSPOSE4_PS(s0, s1, s2, s3);
	_mm_store_ps(d, s0);
	_mm_store_ps(d + 4, s1);
	_mm_store_ps(d + 8, s2);
	_mm_store_ps(d + 12, s3);
}

void dct4x4_llm_sse(float* a, float* b, float* temp, int flag)
{
	if (flag == 0)
	{
		dct4x4_1d_llm_fwd_sse(a, temp);
		transpose4x4(temp);
		dct4x4_1d_llm_fwd_sse(temp, b);
		transpose4x4(b);
		__m128 c = _mm_set1_ps(0.250f);
		_mm_store_ps(b, _mm_mul_ps(_mm_load_ps(b), c));
		_mm_store_ps(b + 4, _mm_mul_ps(_mm_load_ps(b + 4), c));
		_mm_store_ps(b + 8, _mm_mul_ps(_mm_load_ps(b + 8), c));
		_mm_store_ps(b + 12, _mm_mul_ps(_mm_load_ps(b + 12), c));
	}
	else
	{
		dct4x4_1d_llm_inv_sse(a, temp);
		transpose4x4(temp);
		dct4x4_1d_llm_inv_sse(temp, b);
		transpose4x4(b);
		__m128 c = _mm_set1_ps(0.250f);
		_mm_store_ps(b, _mm_mul_ps(_mm_load_ps(b), c));
		_mm_store_ps(b + 4, _mm_mul_ps(_mm_load_ps(b + 4), c));
		_mm_store_ps(b + 8, _mm_mul_ps(_mm_load_ps(b + 8), c));
		_mm_store_ps(b + 12, _mm_mul_ps(_mm_load_ps(b + 12), c));
	}
}
void fDCT2D4x4_and_threshold_keep00_32f(float* s, float* d, float thresh)
{
	const int __declspec(align(16)) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
	const __m128 mth = _mm_set1_ps(thresh);
	const __m128 zeros = _mm_setzero_ps();
	const __m128 c2 = _mm_set1_ps(1.30656f);//cos(CV_PI*2/16.0)*sqrt(2);
	const __m128 c6 = _mm_set1_ps(0.541196f);//cos(CV_PI*6/16.0)*sqrt(2);

	__m128 s0 = _mm_load_ps(s); s += 4;
	__m128 s1 = _mm_load_ps(s); s += 4;
	__m128 s2 = _mm_load_ps(s); s += 4;
	__m128 s3 = _mm_load_ps(s);

	__m128 p03 = _mm_add_ps(s0, s3);
	__m128 p12 = _mm_add_ps(s1, s2);
	__m128 m03 = _mm_sub_ps(s0, s3);
	__m128 m12 = _mm_sub_ps(s1, s2);

	__m128 v = _mm_add_ps(p03, p12);
	__m128 msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	// keep 00 coef.
	__m128 v2 = _mm_blendv_ps(zeros, v, msk);
	v2 = _mm_blend_ps(v2, v, 1);
	_mm_store_ps(d, v2);

	v = _mm_add_ps(_mm_mul_ps(c2, m03), _mm_mul_ps(c6, m12));
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(d + 4, v);

	v = _mm_sub_ps(p03, p12);
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(d + 8, v);

	v = _mm_sub_ps(_mm_mul_ps(c6, m03), _mm_mul_ps(c2, m12));
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(d + 12, v);
}

void fDCT2D4x4_and_threshold_32f(float* s, float* d, float thresh)
{
	const int __declspec(align(16)) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
	const __m128 mth = _mm_set1_ps(thresh);
	const __m128 zeros = _mm_setzero_ps();
	const __m128 c2 = _mm_set1_ps(1.30656f);//cos(CV_PI*2/16.0)*sqrt(2);
	const __m128 c6 = _mm_set1_ps(0.541196f);//cos(CV_PI*6/16.0)*sqrt(2);

	__m128 s0 = _mm_load_ps(s); s += 4;
	__m128 s1 = _mm_load_ps(s); s += 4;
	__m128 s2 = _mm_load_ps(s); s += 4;
	__m128 s3 = _mm_load_ps(s);

	__m128 p03 = _mm_add_ps(s0, s3);
	__m128 p12 = _mm_add_ps(s1, s2);
	__m128 m03 = _mm_sub_ps(s0, s3);
	__m128 m12 = _mm_sub_ps(s1, s2);

	__m128 v = _mm_add_ps(p03, p12);
	__m128 msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(d, v);

	v = _mm_add_ps(_mm_mul_ps(c2, m03), _mm_mul_ps(c6, m12));
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(d + 4, v);

	v = _mm_sub_ps(p03, p12);
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(d + 8, v);

	v = _mm_sub_ps(_mm_mul_ps(c6, m03), _mm_mul_ps(c2, m12));
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(d + 12, v);
}

void fDCT4x4_32f_and_threshold_and_iDCT4x4_32f(float* s, float threshold)
{
	dct4x4_1d_llm_fwd_sse_and_transpose(s, s);
#ifdef _KEEP_00_COEF_
	fDCT2D4x4_and_threshold_keep00_32f(s, s, 4 * threshold);
#else
	fDCT2D8x4_and_threshold_32f(s, s, 4 * threshold);
#endif
	//ommiting transform
	//transpose4x4(s);
	dct4x4_1d_llm_inv_sse_and_transpose(s, s);//transpose4x4(s);
	dct4x4_1d_llm_inv_sse(s, s);
	//ommiting transform
	//transpose4x4(s);

	__m128 c = _mm_set1_ps(0.06250f);
	_mm_store_ps(s, _mm_mul_ps(_mm_load_ps(s), c));
	_mm_store_ps(s + 4, _mm_mul_ps(_mm_load_ps(s + 4), c));
	_mm_store_ps(s + 8, _mm_mul_ps(_mm_load_ps(s + 8), c));
	_mm_store_ps(s + 12, _mm_mul_ps(_mm_load_ps(s + 12), c));
}

int fDCT4x4_32f_and_threshold_and_iDCT4x4_nonzero_32f(float* s, float threshold)
{
	dct4x4_1d_llm_fwd_sse_and_transpose(s, s);
#ifdef _KEEP_00_COEF_
	fDCT2D4x4_and_threshold_keep00_32f(s, s, 4 * threshold);
#else
	fDCT2D8x4_and_threshold_32f(s, s, 4 * threshold);
#endif
	//ommiting transform
	//transpose4x4(s);
	int v = getNonzero(s, 16);
	dct4x4_1d_llm_inv_sse_and_transpose(s, s);//transpose4x4(s);
	dct4x4_1d_llm_inv_sse(s, s);
	//ommiting transform
	//transpose4x4(s);

	__m128 c = _mm_set1_ps(0.06250f);
	_mm_store_ps(s, _mm_mul_ps(_mm_load_ps(s), c));
	_mm_store_ps(s + 4, _mm_mul_ps(_mm_load_ps(s + 4), c));
	_mm_store_ps(s + 8, _mm_mul_ps(_mm_load_ps(s + 8), c));
	_mm_store_ps(s + 12, _mm_mul_ps(_mm_load_ps(s + 12), c));
	return v;
}

//////////////////////////////////////////////////////////////////////////////////////
//Hadamard simd//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
void Hadamard1D4(float *val)
{
	__m128 xmm0, xmm1, xmm2, xmm3;
	__declspec(align(16)) float sign[2][4] = { { 1.0f, -1.0f, 1.0f, -1.0f }, { 1.0f, 1.0f, -1.0f, -1.0f } };

	xmm2 = _mm_load_ps(sign[0]);
	xmm3 = _mm_load_ps(sign[1]);

	xmm1 = xmm0 = _mm_load_ps(val); //x1 x2 x3 x4
	xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
	xmm1 = _mm_mul_ps(xmm1, xmm2); //x1 x2 x3 x4 => x1 -x2 x3 -x4
	xmm1 = xmm0 = _mm_add_ps(xmm0, xmm1);

	xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
	xmm1 = _mm_mul_ps(xmm1, xmm3); //x1 x2 x3 x4 => x1 x2 -x3 -x4
	xmm0 = _mm_add_ps(xmm0, xmm1);
	_mm_store_ps(val, xmm0);
};

void Hadamard1D8(float *val)
{
	__m128 xmm0, xmm1, xmm2;

	__declspec(align(16)) float sign0[4] = { 1.0f, -1.0f, 1.0f, -1.0f };
	__declspec(align(16)) float sign1[4] = { 1.0f, 1.0f, -1.0f, -1.0f };

	xmm2 = _mm_load_ps(sign0);

	xmm1 = xmm0 = _mm_load_ps(val); //x1 x2 x3 x4
	xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
	xmm1 = _mm_mul_ps(xmm1, xmm2); //x1 x2 x3 x4 => x1 -x2 x3 -x4
	__m128 mmaddvalue0 = _mm_add_ps(xmm0, xmm1);


	xmm1 = xmm0 = _mm_load_ps(val + 4); //x1 x2 x3 x4
	xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
	xmm1 = _mm_mul_ps(xmm1, xmm2); //x1 x2 x3 x4 => x1 -x2 x3 -x4
	__m128 mmaddvalue1 = _mm_add_ps(xmm0, xmm1);

	xmm2 = _mm_load_ps(sign1);

	xmm1 = xmm0 = mmaddvalue0; //x1 x2 x3 x4
	xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
	xmm1 = _mm_mul_ps(xmm1, xmm2); //x1 x2 x3 x4 => x1 x2 -x3 -x4
	mmaddvalue0 = _mm_add_ps(xmm0, xmm1);

	xmm1 = xmm0 = mmaddvalue1; //x1 x2 x3 x4
	xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
	xmm1 = _mm_mul_ps(xmm1, xmm2); //x1 x2 x3 x4 => x1 x2 -x3 -x4
	mmaddvalue1 = _mm_add_ps(xmm0, xmm1);

	xmm0 = xmm1 = mmaddvalue0; //x[p+1] x[p+2] x[p+3] x[p+4]
	xmm2 = mmaddvalue1; //x[p+1+m] x[p+2+m] x[p+3+m] x[p+4+m]
	xmm0 = _mm_add_ps(xmm0, xmm2);
	_mm_store_ps(val, xmm0);
	xmm1 = _mm_sub_ps(xmm1, xmm2);
	_mm_store_ps(val + 4, xmm1);
};

void Hadamard1D16(float *val)
{
	__m128 xmm0, xmm1, xmm2;
	__m128 mmadd0, mmadd1, mmadd2, mmadd3;
	__declspec(align(16)) float sign[2][4] = { { 1.0f, -1.0f, 1.0f, -1.0f }, { 1.0f, 1.0f, -1.0f, -1.0f } };

	xmm2 = _mm_load_ps(sign[0]);


	xmm1 = xmm0 = _mm_load_ps(val); //x1 x2 x3 x4
	xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
	xmm1 = _mm_mul_ps(xmm1, xmm2); //x1 x2 x3 x4 => x1 -x2 x3 -x4
	mmadd0 = _mm_add_ps(xmm0, xmm1);


	xmm1 = xmm0 = _mm_load_ps(val + 4); //x1 x2 x3 x4
	xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
	xmm1 = _mm_mul_ps(xmm1, xmm2); //x1 x2 x3 x4 => x1 -x2 x3 -x4
	mmadd1 = _mm_add_ps(xmm0, xmm1);


	xmm1 = xmm0 = _mm_load_ps(val + 8); //x1 x2 x3 x4
	xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
	xmm1 = _mm_mul_ps(xmm1, xmm2); //x1 x2 x3 x4 => x1 -x2 x3 -x4
	mmadd2 = _mm_add_ps(xmm0, xmm1);


	xmm1 = xmm0 = _mm_load_ps(val + 12); //x1 x2 x3 x4
	xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
	xmm1 = _mm_mul_ps(xmm1, xmm2); //x1 x2 x3 x4 => x1 -x2 x3 -x4
	mmadd3 = _mm_add_ps(xmm0, xmm1);


	////////////
	xmm2 = _mm_load_ps(sign[1]);

	xmm1 = xmm0 = mmadd0; //x1 x2 x3 x4
	xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
	xmm1 = _mm_mul_ps(xmm1, xmm2); //x1 x2 x3 x4 => x1 x2 -x3 -x4
	mmadd0 = _mm_add_ps(xmm0, xmm1);


	xmm1 = xmm0 = mmadd1;
	xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
	xmm1 = _mm_mul_ps(xmm1, xmm2); //x1 x2 x3 x4 => x1 x2 -x3 -x4
	mmadd1 = _mm_add_ps(xmm0, xmm1);


	xmm1 = xmm0 = mmadd2;
	xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
	xmm1 = _mm_mul_ps(xmm1, xmm2); //x1 x2 x3 x4 => x1 x2 -x3 -x4
	mmadd2 = _mm_add_ps(xmm0, xmm1);


	xmm1 = xmm0 = mmadd3;
	xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
	xmm1 = _mm_mul_ps(xmm1, xmm2); //x1 x2 x3 x4 => x1 x2 -x3 -x4
	mmadd3 = _mm_add_ps(xmm0, xmm1);

	////////////

	xmm0 = xmm1 = mmadd0; //x[p+1] x[p+2] x[p+3] x[p+4]
	xmm2 = mmadd1; //x[p+1+m] x[p+2+m] x[p+3+m] x[p+4+m]
	mmadd0 = _mm_add_ps(xmm0, xmm2);
	mmadd1 = _mm_sub_ps(xmm1, xmm2);

	xmm0 = xmm1 = mmadd2;
	xmm2 = mmadd3;
	mmadd2 = _mm_add_ps(xmm0, xmm2);
	mmadd3 = _mm_sub_ps(xmm1, xmm2);

	_mm_store_ps(val, _mm_add_ps(mmadd0, mmadd2));
	_mm_store_ps(val + 4, _mm_add_ps(mmadd1, mmadd3));
	_mm_store_ps(val + 8, _mm_sub_ps(mmadd0, mmadd2));
	_mm_store_ps(val + 12, _mm_sub_ps(mmadd1, mmadd3));
};


void Hadamard1D16x16(float *val)
{
	__declspec(align(16)) float sign[2][4] = { { 1.0f, -1.0f, 1.0f, -1.0f }, { 1.0f, 1.0f, -1.0f, -1.0f } };
	const __m128 sgn0 = _mm_load_ps(sign[0]);
	const __m128 sgn1 = _mm_load_ps(sign[1]);

	for (int i = 0; i < 16; i++)
	{
		__m128 xmm0, xmm1, xmm2;
		__m128 mmadd0, mmadd1, mmadd2, mmadd3;


		xmm1 = xmm0 = _mm_load_ps(val); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
		xmm1 = _mm_mul_ps(xmm1, sgn0); //x1 x2 x3 x4 => x1 -x2 x3 -x4
		mmadd0 = _mm_add_ps(xmm0, xmm1);


		xmm1 = xmm0 = _mm_load_ps(val + 4); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
		xmm1 = _mm_mul_ps(xmm1, sgn0); //x1 x2 x3 x4 => x1 -x2 x3 -x4
		mmadd1 = _mm_add_ps(xmm0, xmm1);


		xmm1 = xmm0 = _mm_load_ps(val + 8); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
		xmm1 = _mm_mul_ps(xmm1, sgn0); //x1 x2 x3 x4 => x1 -x2 x3 -x4
		mmadd2 = _mm_add_ps(xmm0, xmm1);


		xmm1 = xmm0 = _mm_load_ps(val + 12); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
		xmm1 = _mm_mul_ps(xmm1, sgn0); //x1 x2 x3 x4 => x1 -x2 x3 -x4
		mmadd3 = _mm_add_ps(xmm0, xmm1);


		////////////

		xmm1 = xmm0 = mmadd0; //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
		xmm1 = _mm_mul_ps(xmm1, sgn1); //x1 x2 x3 x4 => x1 x2 -x3 -x4
		mmadd0 = _mm_add_ps(xmm0, xmm1);


		xmm1 = xmm0 = mmadd1;
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
		xmm1 = _mm_mul_ps(xmm1, sgn1); //x1 x2 x3 x4 => x1 x2 -x3 -x4
		mmadd1 = _mm_add_ps(xmm0, xmm1);


		xmm1 = xmm0 = mmadd2;
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
		xmm1 = _mm_mul_ps(xmm1, sgn1); //x1 x2 x3 x4 => x1 x2 -x3 -x4
		mmadd2 = _mm_add_ps(xmm0, xmm1);


		xmm1 = xmm0 = mmadd3;
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
		xmm1 = _mm_mul_ps(xmm1, sgn1); //x1 x2 x3 x4 => x1 x2 -x3 -x4
		mmadd3 = _mm_add_ps(xmm0, xmm1);

		////////////

		xmm0 = xmm1 = mmadd0; //x[p+1] x[p+2] x[p+3] x[p+4]
		xmm2 = mmadd1; //x[p+1+m] x[p+2+m] x[p+3+m] x[p+4+m]
		mmadd0 = _mm_add_ps(xmm0, xmm2);
		mmadd1 = _mm_sub_ps(xmm1, xmm2);

		xmm0 = xmm1 = mmadd2;
		xmm2 = mmadd3;
		mmadd2 = _mm_add_ps(xmm0, xmm2);
		mmadd3 = _mm_sub_ps(xmm1, xmm2);

		_mm_store_ps(val, _mm_add_ps(mmadd0, mmadd2));
		_mm_store_ps(val + 4, _mm_add_ps(mmadd1, mmadd3));
		_mm_store_ps(val + 8, _mm_sub_ps(mmadd0, mmadd2));
		_mm_store_ps(val + 12, _mm_sub_ps(mmadd1, mmadd3));
		val += 16;
	}
};

void Hadamard1Dn(float *val, size_t n)
{
	size_t i, j, k;
	__m128 xmm0, xmm1, xmm2;
	float *addvalue, *subvalue;
	__declspec(align(16)) float sign[2][4] = { { 1.0f, -1.0f, 1.0f, -1.0f }, { 1.0f, 1.0f, -1.0f, -1.0f } };

	xmm2 = _mm_load_ps(sign[0]);
	for (i = 0, addvalue = val; i < n; i += 4, addvalue++)
	{
		xmm1 = xmm0 = _mm_load_ps(addvalue); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
		xmm1 = _mm_mul_ps(xmm1, xmm2); //x1 x2 x3 x4 => x1 -x2 x3 -x4
		xmm0 = _mm_add_ps(xmm0, xmm1);
		_mm_store_ps(addvalue, xmm0);
	}

	xmm2 = _mm_load_ps(sign[1]);
	for (i = 0, addvalue = val; i < n; i += 4, addvalue++)
	{
		xmm1 = xmm0 = _mm_load_ps(addvalue); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
		xmm1 = _mm_mul_ps(xmm1, xmm2); //x1 x2 x3 x4 => x1 x2 -x3 -x4
		xmm0 = _mm_add_ps(xmm0, xmm1);
		_mm_store_ps(addvalue, xmm0);
	}

	for (i = 4; i < n; i <<= 1)
	{
		for (j = 0; j < n; j += i * 2)
		{
			addvalue = (val + j + 0);
			subvalue = (val + j + i);
			for (k = 0; k < i; k += 4, addvalue++, subvalue++)
			{
				xmm0 = xmm1 = _mm_load_ps(addvalue); //x[p+1] x[p+2] x[p+3] x[p+4]
				xmm2 = _mm_load_ps(subvalue); //x[p+1+m] x[p+2+m] x[p+3+m] x[p+4+m]
				xmm0 = _mm_add_ps(xmm0, xmm2);
				xmm1 = _mm_sub_ps(xmm1, xmm2);
				_mm_store_ps(addvalue, xmm0);
				_mm_store_ps(subvalue, xmm1);
			}
		}
	}
};

void divval(float* src, int size, float div)
{
	const __m128 h = _mm_set1_ps(div);
	for (int i = 0; i < size; i += 4)
	{
		_mm_store_ps(src + i, _mm_mul_ps(_mm_load_ps(src + i), h));
	}
}



void divvalandthresh(float* src, int size, float thresh, float div)
{
	const __m128 h = _mm_set1_ps(div);
	const int __declspec(align(16)) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
	const __m128 mth = _mm_set1_ps(thresh);
	const __m128 zeros = _mm_setzero_ps();

	for (int i = 0; i < size; i += 4)
	{
		__m128 v = _mm_mul_ps(_mm_load_ps(src + i), h);
		__m128 msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
		v = _mm_blendv_ps(zeros, v, msk);
		_mm_store_ps(src + i, v);
	}
}

void Hadamard2D16x16(float* src)
{
	for (int i = 0; i < 16; i++)
		Hadamard1D16(src + 16 * i);


	transpose16x16(src);

	for (int i = 0; i < 16; i++)
		Hadamard1D16(src + 16 * i);

	transpose16x16(src);
	divval(src, 256, 0.0625f);
}

void Hadamard2D16x16andThreshandIDHT(float* src, float thresh)
{
	Hadamard1D16x16(src);
	transpose16x16(src);
	Hadamard1D16x16(src);
#ifdef _KEEP_00_COEF_
	float f0 = src[0] * 0.0625f;
#endif
	divvalandthresh(src, 256, thresh, 0.0625f);
#ifdef _KEEP_00_COEF_
	src[0] = f0;
#endif
	Hadamard1D16x16(src);
	transpose16x16(src);
	Hadamard1D16x16(src);

	divval(src, 256, 0.0625f);
}

void Hadamard2D4x4(float* src)
{
	Hadamard1D4(src);
	Hadamard1D4(src + 4);
	Hadamard1D4(src + 8);
	Hadamard1D4(src + 12);
	divval(src, 16, 0.5f);
	transpose4x4(src);
	Hadamard1D4(src);
	Hadamard1D4(src + 4);
	Hadamard1D4(src + 8);
	Hadamard1D4(src + 12);
	transpose4x4(src);
	divval(src, 16, 0.5f);
}

void Hadamard2D4x4andThresh(float* src, float thresh)
{
	Hadamard1D4(src);
	Hadamard1D4(src + 4);
	Hadamard1D4(src + 8);
	Hadamard1D4(src + 12);
	divval(src, 16, 0.5f);
	transpose4x4(src);
	Hadamard1D4(src);
	Hadamard1D4(src + 4);
	Hadamard1D4(src + 8);
	Hadamard1D4(src + 12);
	transpose4x4(src);
	divvalandthresh(src, 16, thresh, 0.5f);
}

void Hadamard2D4x4andThreshandIDHT(float* src, float thresh)
{
	const __m128 h = _mm_set1_ps(0.25f);
	const int __declspec(align(16)) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
	const __m128 mth = _mm_set1_ps(thresh);
	const __m128 zeros = _mm_setzero_ps();

	__declspec(align(16)) float sign[2][4] = { { 1.0f, -1.0f, 1.0f, -1.0f }, { 1.0f, 1.0f, -1.0f, -1.0f } };
	float* val = src;

	for (int i = 0; i < 4; i++)
	{
		__m128 xmm0, xmm1, xmm2, xmm3;

		xmm2 = _mm_load_ps(sign[0]);
		xmm3 = _mm_load_ps(sign[1]);

		xmm1 = xmm0 = _mm_load_ps(val); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
		xmm1 = _mm_mul_ps(xmm1, xmm2); //x1 x2 x3 x4 => x1 -x2 x3 -x4
		xmm1 = xmm0 = _mm_add_ps(xmm0, xmm1);

		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
		xmm1 = _mm_mul_ps(xmm1, xmm3); //x1 x2 x3 x4 => x1 x2 -x3 -x4
		xmm0 = _mm_add_ps(xmm0, xmm1);
		_mm_store_ps(val, xmm0);
		val += 4;
	};
	{
		__m128 m0 = _mm_load_ps(src);
		__m128 m1 = _mm_load_ps(src + 4);
		__m128 m2 = _mm_load_ps(src + 8);
		__m128 m3 = _mm_load_ps(src + 12);
		_MM_TRANSPOSE4_PS(m0, m1, m2, m3);
		_mm_store_ps(src, m0);
		_mm_store_ps(src + 4, m1);
		_mm_store_ps(src + 8, m2);
		_mm_store_ps(src + 12, m3);
	}
	val = src;
	{
		__m128 xmm0, xmm1, xmm2, xmm3;

		xmm2 = _mm_load_ps(sign[0]);
		xmm3 = _mm_load_ps(sign[1]);

		xmm1 = xmm0 = _mm_load_ps(val); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
		xmm1 = _mm_mul_ps(xmm1, xmm2); //x1 x2 x3 x4 => x1 -x2 x3 -x4
		xmm1 = xmm0 = _mm_add_ps(xmm0, xmm1);

		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
		xmm1 = _mm_mul_ps(xmm1, xmm3); //x1 x2 x3 x4 => x1 x2 -x3 -x4
		xmm0 = _mm_add_ps(xmm0, xmm1);

		__m128 v = _mm_mul_ps(xmm0, h);
		__m128 msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
#ifdef _KEEP_00_COEF_
		__m128 v2 = _mm_blendv_ps(zeros, v, msk);
		v2 = _mm_blend_ps(v2, v, 1);
		_mm_store_ps(val, v2);
#else
		v = _mm_blendv_ps(zeros, v, msk);
		_mm_store_ps(val, v);
#endif		
		val += 4;
	}
	for (int i = 1; i < 4; i++)
	{
		__m128 xmm0, xmm1, xmm2, xmm3;

		xmm2 = _mm_load_ps(sign[0]);
		xmm3 = _mm_load_ps(sign[1]);

		xmm1 = xmm0 = _mm_load_ps(val); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
		xmm1 = _mm_mul_ps(xmm1, xmm2); //x1 x2 x3 x4 => x1 -x2 x3 -x4
		xmm1 = xmm0 = _mm_add_ps(xmm0, xmm1);

		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
		xmm1 = _mm_mul_ps(xmm1, xmm3); //x1 x2 x3 x4 => x1 x2 -x3 -x4
		xmm0 = _mm_add_ps(xmm0, xmm1);

		__m128 v = _mm_mul_ps(xmm0, h);
		__m128 msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);

		v = _mm_blendv_ps(zeros, v, msk);
		_mm_store_ps(val, v);

		val += 4;
	}

	val = src;
	for (int i = 0; i < 4; i++)
	{
		__m128 xmm0, xmm1, xmm2, xmm3;

		xmm2 = _mm_load_ps(sign[0]);
		xmm3 = _mm_load_ps(sign[1]);

		xmm1 = xmm0 = _mm_load_ps(val); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
		xmm1 = _mm_mul_ps(xmm1, xmm2); //x1 x2 x3 x4 => x1 -x2 x3 -x4
		xmm1 = xmm0 = _mm_add_ps(xmm0, xmm1);

		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
		xmm1 = _mm_mul_ps(xmm1, xmm3); //x1 x2 x3 x4 => x1 x2 -x3 -x4
		xmm0 = _mm_add_ps(xmm0, xmm1);
		_mm_store_ps(val, xmm0);
		val += 4;
	}
	{
		__m128 m0 = _mm_load_ps(src);
		__m128 m1 = _mm_load_ps(src + 4);
		__m128 m2 = _mm_load_ps(src + 8);
		__m128 m3 = _mm_load_ps(src + 12);
		_MM_TRANSPOSE4_PS(m0, m1, m2, m3);
		_mm_store_ps(src, m0);
		_mm_store_ps(src + 4, m1);
		_mm_store_ps(src + 8, m2);
		_mm_store_ps(src + 12, m3);
	}
	val = src;
	for (int i = 0; i < 4; i++)
	{
		__m128 xmm0, xmm1, xmm2, xmm3;

		xmm2 = _mm_load_ps(sign[0]);
		xmm3 = _mm_load_ps(sign[1]);

		xmm1 = xmm0 = _mm_load_ps(val); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
		xmm1 = _mm_mul_ps(xmm1, xmm2); //x1 x2 x3 x4 => x1 -x2 x3 -x4
		xmm1 = xmm0 = _mm_add_ps(xmm0, xmm1);

		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
		xmm1 = _mm_mul_ps(xmm1, xmm3); //x1 x2 x3 x4 => x1 x2 -x3 -x4
		xmm0 = _mm_add_ps(xmm0, xmm1);
		_mm_store_ps(val, _mm_mul_ps(xmm0, h));

		val += 4;
	};

}

void Hadamard2D8x8(float* src)
{
	Hadamard1D8(src);
	Hadamard1D8(src + 8);
	Hadamard1D8(src + 16);
	Hadamard1D8(src + 24);
	Hadamard1D8(src + 32);
	Hadamard1D8(src + 40);
	Hadamard1D8(src + 48);
	Hadamard1D8(src + 56);
	divval(src, 64, 0.5f);
	transpose8x8(src);
	Hadamard1D8(src);
	Hadamard1D8(src + 8);
	Hadamard1D8(src + 16);
	Hadamard1D8(src + 24);
	Hadamard1D8(src + 32);
	Hadamard1D8(src + 40);
	Hadamard1D8(src + 48);
	Hadamard1D8(src + 56);
	transpose8x8(src);
	divval(src, 64, 0.5f);
}

void Hadamard2D8x8andThresh(float* src, float thresh)
{
	Hadamard1D8(src);
	Hadamard1D8(src + 8);
	Hadamard1D8(src + 16);
	Hadamard1D8(src + 24);
	Hadamard1D8(src + 32);
	Hadamard1D8(src + 40);
	Hadamard1D8(src + 48);
	Hadamard1D8(src + 56);
	divval(src, 64, 0.5f);
	transpose8x8(src);
	Hadamard1D8(src);
	Hadamard1D8(src + 8);
	Hadamard1D8(src + 16);
	Hadamard1D8(src + 24);
	Hadamard1D8(src + 32);
	Hadamard1D8(src + 40);
	Hadamard1D8(src + 48);
	Hadamard1D8(src + 56);
	transpose8x8(src);
	divvalandthresh(src, 64, thresh, 0.25f);
}


void Hadamard2D8x8i(float *vall)
{
	float* val = vall;
	__declspec(align(16)) float sign0[4] = { 1.0f, -1.0f, 1.0f, -1.0f };
	__declspec(align(16)) float sign1[4] = { 1.0f, 1.0f, -1.0f, -1.0f };

	const __m128 sgn0 = _mm_load_ps(sign0);
	const __m128 sgn1 = _mm_load_ps(sign1);
	__m128 xmm0, xmm1;
	for (int i = 0; i < 8; i++)
	{
		xmm1 = xmm0 = _mm_load_ps(val); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
		xmm1 = _mm_mul_ps(xmm1, sgn0); //x1 x2 x3 x4 => x1 -x2 x3 -x4
		__m128 mmaddvalue0 = _mm_add_ps(xmm0, xmm1);


		xmm1 = xmm0 = _mm_load_ps(val + 4); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
		xmm1 = _mm_mul_ps(xmm1, sgn0); //x1 x2 x3 x4 => x1 -x2 x3 -x4
		__m128 mmaddvalue1 = _mm_add_ps(xmm0, xmm1);

		xmm1 = xmm0 = mmaddvalue0; //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
		xmm1 = _mm_mul_ps(xmm1, sgn1); //x1 x2 x3 x4 => x1 x2 -x3 -x4
		mmaddvalue0 = _mm_add_ps(xmm0, xmm1);

		xmm1 = xmm0 = mmaddvalue1; //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
		xmm1 = _mm_mul_ps(xmm1, sgn1); //x1 x2 x3 x4 => x1 x2 -x3 -x4
		mmaddvalue1 = _mm_add_ps(xmm0, xmm1);

		xmm0 = xmm1 = mmaddvalue0; //x[p+1] x[p+2] x[p+3] x[p+4]
		//x[p+1+m] x[p+2+m] x[p+3+m] x[p+4+m]
		xmm0 = _mm_add_ps(xmm0, mmaddvalue1);
		_mm_store_ps(val, xmm0);
		xmm1 = _mm_sub_ps(xmm1, mmaddvalue1);
		_mm_store_ps(val + 4, xmm1);
		val += 8;
	}

	float* src = vall;
	{
		__declspec(align(16)) float temp[16];
		__m128 m0 = _mm_load_ps(src);
		__m128 m1 = _mm_load_ps(src + 8);
		__m128 m2 = _mm_load_ps(src + 16);
		__m128 m3 = _mm_load_ps(src + 24);
		_MM_TRANSPOSE4_PS(m0, m1, m2, m3);
		_mm_store_ps(src, m0);
		_mm_store_ps(src + 8, m1);
		_mm_store_ps(src + 16, m2);
		_mm_store_ps(src + 24, m3);


		m0 = _mm_load_ps(src + 4);
		m1 = _mm_load_ps(src + 12);
		m2 = _mm_load_ps(src + 20);
		m3 = _mm_load_ps(src + 28);
		_MM_TRANSPOSE4_PS(m0, m1, m2, m3);
		/*_mm_store_ps(dest+32,m0);
		_mm_store_ps(dest+40,m1);
		_mm_store_ps(dest+48,m2);
		_mm_store_ps(dest+56,m3);*/
		_mm_store_ps(temp, m0);
		_mm_store_ps(temp + 4, m1);
		_mm_store_ps(temp + 8, m2);
		_mm_store_ps(temp + 12, m3);

		m0 = _mm_load_ps(src + 32);
		m1 = _mm_load_ps(src + 40);
		m2 = _mm_load_ps(src + 48);
		m3 = _mm_load_ps(src + 56);
		_MM_TRANSPOSE4_PS(m0, m1, m2, m3);
		_mm_store_ps(src + 4, m0);
		_mm_store_ps(src + 12, m1);
		_mm_store_ps(src + 20, m2);
		_mm_store_ps(src + 28, m3);

		memcpy(src + 32, temp, sizeof(float) * 4);
		memcpy(src + 40, temp + 4, sizeof(float) * 4);
		memcpy(src + 48, temp + 8, sizeof(float) * 4);
		memcpy(src + 56, temp + 12, sizeof(float) * 4);


		m0 = _mm_load_ps(src + 36);
		m1 = _mm_load_ps(src + 44);
		m2 = _mm_load_ps(src + 52);
		m3 = _mm_load_ps(src + 60);
		_MM_TRANSPOSE4_PS(m0, m1, m2, m3);
		_mm_store_ps(src + 36, m0);
		_mm_store_ps(src + 44, m1);
		_mm_store_ps(src + 52, m2);
		_mm_store_ps(src + 60, m3);
	}
	val = vall;
	const __m128 h = _mm_set1_ps(0.125f);
	for (int i = 0; i < 8; i++)
	{
		xmm1 = xmm0 = _mm_load_ps(val); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
		xmm1 = _mm_mul_ps(xmm1, sgn0); //x1 x2 x3 x4 => x1 -x2 x3 -x4
		__m128 mmaddvalue0 = _mm_add_ps(xmm0, xmm1);


		xmm1 = xmm0 = _mm_load_ps(val + 4); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
		xmm1 = _mm_mul_ps(xmm1, sgn0); //x1 x2 x3 x4 => x1 -x2 x3 -x4
		__m128 mmaddvalue1 = _mm_add_ps(xmm0, xmm1);


		xmm1 = xmm0 = mmaddvalue0; //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
		xmm1 = _mm_mul_ps(xmm1, sgn1); //x1 x2 x3 x4 => x1 x2 -x3 -x4
		mmaddvalue0 = _mm_add_ps(xmm0, xmm1);

		xmm1 = xmm0 = mmaddvalue1; //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
		xmm1 = _mm_mul_ps(xmm1, sgn1); //x1 x2 x3 x4 => x1 x2 -x3 -x4
		mmaddvalue1 = _mm_add_ps(xmm0, xmm1);

		xmm0 = xmm1 = mmaddvalue0; //x[p+1] x[p+2] x[p+3] x[p+4]
		//x[p+1+m] x[p+2+m] x[p+3+m] x[p+4+m]
		xmm0 = _mm_add_ps(xmm0, mmaddvalue1);
		_mm_store_ps(val, _mm_mul_ps(h, xmm0));
		xmm1 = _mm_sub_ps(xmm1, mmaddvalue1);
		_mm_store_ps(val + 4, _mm_mul_ps(h, xmm1));
		val += 8;
	}
};

void Hadamard2D8x8i_and_thresh(float *vall, float thresh)
{
	float* val = vall;
	__declspec(align(16)) float sign0[4] = { 1.0f, -1.0f, 1.0f, -1.0f };
	__declspec(align(16)) float sign1[4] = { 1.0f, 1.0f, -1.0f, -1.0f };

	const __m128 sgn0 = _mm_load_ps(sign0);
	const __m128 sgn1 = _mm_load_ps(sign1);
	__m128 xmm0, xmm1;
	for (int i = 0; i < 8; i++)
	{
		xmm1 = xmm0 = _mm_load_ps(val); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
		xmm1 = _mm_mul_ps(xmm1, sgn0); //x1 x2 x3 x4 => x1 -x2 x3 -x4
		__m128 mmaddvalue0 = _mm_add_ps(xmm0, xmm1);


		xmm1 = xmm0 = _mm_load_ps(val + 4); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
		xmm1 = _mm_mul_ps(xmm1, sgn0); //x1 x2 x3 x4 => x1 -x2 x3 -x4
		__m128 mmaddvalue1 = _mm_add_ps(xmm0, xmm1);


		xmm1 = xmm0 = mmaddvalue0; //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
		xmm1 = _mm_mul_ps(xmm1, sgn1); //x1 x2 x3 x4 => x1 x2 -x3 -x4
		mmaddvalue0 = _mm_add_ps(xmm0, xmm1);

		xmm1 = xmm0 = mmaddvalue1; //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
		xmm1 = _mm_mul_ps(xmm1, sgn1); //x1 x2 x3 x4 => x1 x2 -x3 -x4
		mmaddvalue1 = _mm_add_ps(xmm0, xmm1);

		xmm0 = xmm1 = mmaddvalue0; //x[p+1] x[p+2] x[p+3] x[p+4]
		//x[p+1+m] x[p+2+m] x[p+3+m] x[p+4+m]
		xmm0 = _mm_add_ps(xmm0, mmaddvalue1);
		_mm_store_ps(val, xmm0);
		xmm1 = _mm_sub_ps(xmm1, mmaddvalue1);
		_mm_store_ps(val + 4, xmm1);
		val += 8;
	}

	float* src = vall;
	{
		__declspec(align(16)) float temp[16];
		__m128 m0 = _mm_load_ps(src);
		__m128 m1 = _mm_load_ps(src + 8);
		__m128 m2 = _mm_load_ps(src + 16);
		__m128 m3 = _mm_load_ps(src + 24);
		_MM_TRANSPOSE4_PS(m0, m1, m2, m3);
		_mm_store_ps(src, m0);
		_mm_store_ps(src + 8, m1);
		_mm_store_ps(src + 16, m2);
		_mm_store_ps(src + 24, m3);


		m0 = _mm_load_ps(src + 4);
		m1 = _mm_load_ps(src + 12);
		m2 = _mm_load_ps(src + 20);
		m3 = _mm_load_ps(src + 28);
		_MM_TRANSPOSE4_PS(m0, m1, m2, m3);
		/*_mm_store_ps(dest+32,m0);
		_mm_store_ps(dest+40,m1);
		_mm_store_ps(dest+48,m2);
		_mm_store_ps(dest+56,m3);*/
		_mm_store_ps(temp, m0);
		_mm_store_ps(temp + 4, m1);
		_mm_store_ps(temp + 8, m2);
		_mm_store_ps(temp + 12, m3);

		m0 = _mm_load_ps(src + 32);
		m1 = _mm_load_ps(src + 40);
		m2 = _mm_load_ps(src + 48);
		m3 = _mm_load_ps(src + 56);
		_MM_TRANSPOSE4_PS(m0, m1, m2, m3);
		_mm_store_ps(src + 4, m0);
		_mm_store_ps(src + 12, m1);
		_mm_store_ps(src + 20, m2);
		_mm_store_ps(src + 28, m3);

		memcpy(src + 32, temp, sizeof(float) * 4);
		memcpy(src + 40, temp + 4, sizeof(float) * 4);
		memcpy(src + 48, temp + 8, sizeof(float) * 4);
		memcpy(src + 56, temp + 12, sizeof(float) * 4);


		m0 = _mm_load_ps(src + 36);
		m1 = _mm_load_ps(src + 44);
		m2 = _mm_load_ps(src + 52);
		m3 = _mm_load_ps(src + 60);
		_MM_TRANSPOSE4_PS(m0, m1, m2, m3);
		_mm_store_ps(src + 36, m0);
		_mm_store_ps(src + 44, m1);
		_mm_store_ps(src + 52, m2);
		_mm_store_ps(src + 60, m3);
	}
	val = vall;

	const __m128 h = _mm_set1_ps(0.125f);
	const int __declspec(align(16)) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
	const __m128 mth = _mm_set1_ps(thresh);
	const __m128 zeros = _mm_setzero_ps();

	{
		xmm1 = xmm0 = _mm_load_ps(val); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
		xmm1 = _mm_mul_ps(xmm1, sgn0); //x1 x2 x3 x4 => x1 -x2 x3 -x4
		__m128 mmaddvalue0 = _mm_add_ps(xmm0, xmm1);


		xmm1 = xmm0 = _mm_load_ps(val + 4); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
		xmm1 = _mm_mul_ps(xmm1, sgn0); //x1 x2 x3 x4 => x1 -x2 x3 -x4
		__m128 mmaddvalue1 = _mm_add_ps(xmm0, xmm1);


		xmm1 = xmm0 = mmaddvalue0; //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
		xmm1 = _mm_mul_ps(xmm1, sgn1); //x1 x2 x3 x4 => x1 x2 -x3 -x4
		mmaddvalue0 = _mm_add_ps(xmm0, xmm1);

		xmm1 = xmm0 = mmaddvalue1; //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
		xmm1 = _mm_mul_ps(xmm1, sgn1); //x1 x2 x3 x4 => x1 x2 -x3 -x4
		mmaddvalue1 = _mm_add_ps(xmm0, xmm1);

		xmm0 = xmm1 = mmaddvalue0; //x[p+1] x[p+2] x[p+3] x[p+4]
		//x[p+1+m] x[p+2+m] x[p+3+m] x[p+4+m]
		xmm0 = _mm_add_ps(xmm0, mmaddvalue1);
		__m128 v = _mm_mul_ps(xmm0, h);
		__m128 msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
#ifdef _KEEP_00_COEF_
		__m128 v2 = _mm_blendv_ps(zeros, v, msk);
		v2 = _mm_blend_ps(v2, v, 1);
		_mm_store_ps(val, v2);
#else
		v = _mm_blendv_ps(zeros, v, msk);
		_mm_store_ps(val, v);
#endif


		xmm1 = _mm_sub_ps(xmm1, mmaddvalue1);
		v = _mm_mul_ps(xmm1, h);
		msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
		v = _mm_blendv_ps(zeros, v, msk);
		_mm_store_ps(val + 4, v);
		val += 8;
	}
	for (int i = 1; i < 8; i++)
	{
		xmm1 = xmm0 = _mm_load_ps(val); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
		xmm1 = _mm_mul_ps(xmm1, sgn0); //x1 x2 x3 x4 => x1 -x2 x3 -x4
		__m128 mmaddvalue0 = _mm_add_ps(xmm0, xmm1);


		xmm1 = xmm0 = _mm_load_ps(val + 4); //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0xb1); //x1 x2 x3 x4 => x2 x1 x4 x3
		xmm1 = _mm_mul_ps(xmm1, sgn0); //x1 x2 x3 x4 => x1 -x2 x3 -x4
		__m128 mmaddvalue1 = _mm_add_ps(xmm0, xmm1);


		xmm1 = xmm0 = mmaddvalue0; //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
		xmm1 = _mm_mul_ps(xmm1, sgn1); //x1 x2 x3 x4 => x1 x2 -x3 -x4
		mmaddvalue0 = _mm_add_ps(xmm0, xmm1);

		xmm1 = xmm0 = mmaddvalue1; //x1 x2 x3 x4
		xmm0 = _mm_shuffle_ps(xmm0, xmm1, 0x4e); //x1 x2 x3 x4 => x3 x4 x1 x2
		xmm1 = _mm_mul_ps(xmm1, sgn1); //x1 x2 x3 x4 => x1 x2 -x3 -x4
		mmaddvalue1 = _mm_add_ps(xmm0, xmm1);

		xmm0 = xmm1 = mmaddvalue0; //x[p+1] x[p+2] x[p+3] x[p+4]
		//x[p+1+m] x[p+2+m] x[p+3+m] x[p+4+m]
		xmm0 = _mm_add_ps(xmm0, mmaddvalue1);
		__m128 v = _mm_mul_ps(xmm0, h);
		__m128 msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
		v = _mm_blendv_ps(zeros, v, msk);
		_mm_store_ps(val, v);

		xmm1 = _mm_sub_ps(xmm1, mmaddvalue1);
		v = _mm_mul_ps(xmm1, h);
		msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
		v = _mm_blendv_ps(zeros, v, msk);
		_mm_store_ps(val + 4, v);
		val += 8;
	}
};
void Hadamard2D8x8andThreshandIDHT(float* src, float thresh)
{
	Hadamard2D8x8i_and_thresh(src, thresh);
	Hadamard2D8x8i(src);
}
//////////////////////////////////////////////////////////////////////////////////////
//TBB for DCT TBB DCT tbbdct//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
void fDCT16x16_threshold_keep00_iDCT16x16(const float* src, float* dest, float th);
class DenoiseDCTShrinkageInvorker16x16 : public cv::ParallelLoopBody
{
private:
	float* src;
	float* dest;
	float thresh;
	int width;
	int height;

public:

	DenoiseDCTShrinkageInvorker16x16(float *sim, float* dim, float Th, int w, int h) : src(sim), dest(dim), width(w), height(h), thresh(Th)
	{
		;
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
			Mat mask(Size(16, 16), CV_8U);

			float* ptch = patch.ptr<float>(0);

			float* s0 = &src[width*j];
			float* d0 = &dest[width*j];
			const int sz = sizeof(float) * 16;
			for (int i = 0; i < wstep; i++)
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

				//fDCT8x8_32f_and_threshold_and_iDCT8x8_32f(ptch, thresh);
				
				//

				fDCT16x16_threshold_keep00_iDCT16x16(patch.ptr<float>(0), patch.ptr<float>(0), thresh);
				

				//fDCT16x16(patch.ptr<float>(0), patch.ptr<float>(0));
				//dct(patch,patch);
				//float f0 = *(float*)patch.data;
				//compare(abs(patch), thresh, mask, CMP_LT);
				//patch.setTo(0.f, mask);
				//*(float*)patch.data = f0;
				
				//dct(patch,patch,DCT_INVERSE);
				
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
				}
				s0++;
				d0++;
			}
		}
	}
};

class DenoiseDCTShrinkageInvorker8x8 : public cv::ParallelLoopBody
{
private:
	float* src;
	float* dest;
	float thresh;
	int width;
	int height;

public:

	DenoiseDCTShrinkageInvorker8x8(float *sim, float* dim, float Th, int w, int h) : src(sim), dest(dim), width(w), height(h), thresh(Th)
	{
		;
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
		int j;

		for (j = range.start; j != range.end; j++)
		{
			Mat patch(Size(8, 8), CV_32F);
			float* ptch = patch.ptr<float>(0);

			float* s0 = &src[width*j];
			float* d0 = &dest[width*j];
			const int sz = sizeof(float) * 8;
			for (int i = 0; i < wstep; i++)
			{
				memcpy(ptch, s0, sz);
				memcpy(ptch + 8, s0 + w1, sz);
				memcpy(ptch + 16, s0 + w2, sz);
				memcpy(ptch + 24, s0 + w3, sz);
				memcpy(ptch + 32, s0 + w4, sz);
				memcpy(ptch + 40, s0 + w5, sz);
				memcpy(ptch + 48, s0 + w6, sz);
				memcpy(ptch + 56, s0 + w7, sz);

				fDCT8x8_32f_and_threshold_and_iDCT8x8_32f(ptch, thresh);

				/*fDCT8x8_32f(patch.ptr<float>(0),patch.ptr<float>(0), buff.ptr<float>(0));
				float f0=*(float*)patch.data;
				compare(abs(patch),thresh,mask,CMP_LT);
				patch.setTo(0.f,mask);
				*(float*)patch.data = f0;
				iDCT8x8_32f(patch.ptr<float>(0),patch.ptr<float>(0), buff.ptr<float>(0));*/
				//dct(patch,patch,DCT_INVERSE);


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
				s0++;
				d0++;
			}
		}
	}
};

class DenoiseDCTShrinkageInvorker4x4 : public cv::ParallelLoopBody
{
private:
	float* src;
	float* dest;
	float thresh;
	int width;
	int height;

public:
	DenoiseDCTShrinkageInvorker4x4(float *sim, float* dim, float Th, int w, int h) : src(sim), dest(dim), width(w), height(h), thresh(Th)
	{
		;
	}
	virtual void operator() (const Range& range) const
	{
		const int size1 = width * height;
		const int hstep = height - 4 + 1;
		const int wstep = width - 4 + 1;
		const int w1 = 1 * width;
		const int w2 = 2 * width;
		const int w3 = 3 * width;

		int j;
		Mat buff(Size(4, 4), CV_32F); Mat mask;
		for (j = range.start; j != range.end; j++)
		{
			Mat patch(Size(4, 4), CV_32F);
			float* ptch = patch.ptr<float>(0);

			float* s0 = &src[width*j];
			float* d0 = &dest[width*j];
			const int sz = sizeof(float) * 4;
			for (int i = 0; i < wstep; i++)
			{
				memcpy(ptch, s0, sz);
				memcpy(ptch + 4, s0 + w1, sz);
				memcpy(ptch + 8, s0 + w2, sz);
				memcpy(ptch + 12, s0 + w3, sz);

				fDCT4x4_32f_and_threshold_and_iDCT4x4_32f(patch.ptr<float>(0), thresh);

				//#define _HARD_THRESHOLDING_
				//#ifdef _HARD_THRESHOLDING_
				//				dct4x4_llm_sse(patch.ptr<float>(0),patch.ptr<float>(0), buff.ptr<float>(0),0);
				//				//dct(patch,patch);
				//#ifdef _KEEP_00_COEF_
				//				float f0=*(float*)patch.data;
				//#endif
				//				compare(abs(patch),thresh,mask,CMP_LT);
				//				patch.setTo(0.f,mask);
				//
				//#ifdef _KEEP_00_COEF_
				//				*(float*)patch.data = f0;
				//#endif
				//				dct4x4_llm_sse(patch.ptr<float>(0),patch.ptr<float>(0), buff.ptr<float>(0),DCT_INVERSE);
				//				//dct(patch,patch,DCT_INVERSE);
				//
				//#else
				//				dct(patch,patch);
				//				Mat dst;
				//				max(abs(patch)-25*thresh,0.f,dst);
				//				compare(patch,0,mask,CMP_LT);
				//				Mat(-1*dst).copyTo(dst,mask);
				//				dst.at<float>(0,0)=patch.at<float>(0,0);
				//				dct(dst,patch,DCT_INVERSE);
				//#endif

				//add data
				for (int jp = 0; jp < 4; jp++)
				{
					float* s = patch.ptr<float>(jp);
					float* d = &d0[(jp)*width];
					__m128 mp1 = _mm_load_ps(s);
					__m128 sp1 = _mm_loadu_ps(d);

					_mm_storeu_ps(d, _mm_add_ps(sp1, mp1));
				}
				s0++;
				d0++;
			}
		}
	}
};

class DenoiseDCTShrinkageInvorker2x2 : public cv::ParallelLoopBody
{
private:

	float* src;
	float* dest;
	float thresh;
	int width;
	int height;

public:
	DenoiseDCTShrinkageInvorker2x2(float *sim, float* dim, float Th, int w, int h) : src(sim), dest(dim), width(w), height(h), thresh(Th)
	{
		;
	}
	virtual void operator() (const Range& range) const
	{
		//2x2 patch
		const int size1 = width * height;
		const int hstep = height - 2 + 1;
		const int wstep = width - 2 + 1;

		int j;

		for (j = range.start; j != range.end; j++)
		{
			Mat patch(Size(4, 2), CV_32F);
			float* ptch = patch.ptr<float>(0);
			float* s0 = &src[width*j];
			float* d0 = &dest[width*j];
			const int sz = sizeof(float) * 4;
			for (int i = 0; i < wstep; i += 4)
			{
				memcpy(ptch, s0, sz);
				memcpy(ptch + 4, s0 + width, sz);

				fDCT2x2_2pack_32f_and_thresh_and_iDCT2x2_2pack((float*)patch.data, (float*)patch.data, thresh);

				//add data
				__m128 mp1 = _mm_loadu_ps(ptch);
				__m128 sp1 = _mm_loadu_ps(d0);
				_mm_storeu_ps(d0, _mm_add_ps(sp1, mp1));
				mp1 = _mm_loadu_ps(ptch + 4);
				sp1 = _mm_loadu_ps(d0 + width);
				_mm_storeu_ps(d0 + width, _mm_add_ps(sp1, mp1));

				memcpy(ptch, s0 + 1, sz);
				memcpy(ptch + 4, s0 + width + 1, sz);

				fDCT2x2_2pack_32f_and_thresh_and_iDCT2x2_2pack((float*)patch.data, (float*)patch.data, thresh);

				//add data
				mp1 = _mm_loadu_ps(ptch);
				sp1 = _mm_loadu_ps(d0 + 1);
				_mm_storeu_ps(d0 + 1, _mm_add_ps(sp1, mp1));
				mp1 = _mm_loadu_ps(ptch + 4);
				sp1 = _mm_loadu_ps(d0 + width + 1);
				_mm_storeu_ps(d0 + width + 1, _mm_add_ps(sp1, mp1));

				s0 += 4;
				d0 += 4;
			}
		}
	}
};

class DenoiseDCTShrinkageInvorker : public cv::ParallelLoopBody
{
private:


	float* src;
	float* dest;
	float thresh;
	int width;
	int height;
	Size patch_size;

public:

	DenoiseDCTShrinkageInvorker(float *sim, float* dim, float Th, int w, int h, Size psize) : src(sim), dest(dim), width(w), height(h), patch_size(psize), thresh(Th)
	{
		;
	}
	virtual void operator() (const Range& range) const
	{
		int pwidth = patch_size.width;
		int pheight = patch_size.height;
		const int size1 = width * height;
		const int hstep = height - pheight + 1;
		const int wstep = width - pwidth + 1;

		int j;
		Mat d = Mat(Size(width, height), CV_32F, dest);
		for (j = range.start; j != range.end; j++)
		{
			Mat patch(patch_size, CV_32F);
			Mat mask;//(patch_size, CV_8U);

			float* ptch = patch.ptr<float>(0);

			float* s0 = &src[width*j];

			const int sz = sizeof(float)*patch_size.width;
			for (int i = 0; i < wstep; i++)
			{
				for (int k = 0; k < patch_size.height; k++)
				{
					memcpy(ptch + k*patch_size.width, s0 + k*width, sz);
				}

				//Mat show1; patch.convertTo(show1,CV_8U);imshow("patchb",show1); 

				dct(patch, patch);

#ifdef _KEEP_00_COEF_
				float f0 = *(float*)patch.data;
#endif

				compare(abs(patch), thresh, mask, CMP_LT);
				patch.setTo(0.f, mask);

#ifdef _KEEP_00_COEF_
				*(float*)patch.data = f0;
#endif

				dct(patch, patch, DCT_INVERSE);

				//Mat show; patch.convertTo(show,CV_8U);imshow("patch",show); waitKey(0);
				Mat r = d(Rect(i, j, patch_size.width, patch_size.height));
				r += patch;
				s0++;
			}
		}
	}
};

class ShearableDenoiseDCTShrinkageInvorker4x4 : public cv::ParallelLoopBody
{
private:
	float* src;
	float* dest;
	float thresh;
	int width;
	int height;
	int direct;


public:
	ShearableDenoiseDCTShrinkageInvorker4x4(float *sim, float* dim, float Th, int w, int h, int dr) : src(sim), dest(dim), width(w), height(h), thresh(Th), direct(dr)
	{
		;
	}

	virtual void operator() (const Range& range) const
	{
		const int size1 = width * height;
		const int hstep = height - 4 + 1;
		const int wstep = width - 4 + 1;
		const int w1 = 1 * width;
		const int w2 = 2 * width;
		const int w3 = 3 * width;

		int j;
		Mat buff(Size(4, 4), CV_32F); Mat mask;
		for (j = range.start; j != range.end; j++)
		{
			Mat patch(Size(4, 4), CV_32F);
			float* ptch = patch.ptr<float>(0);

			float* s0 = &src[width*j];
			float* d0 = &dest[width*j];
			const int sz = sizeof(float) * 4;
			for (int i = 0; i < wstep; i++)
			{
				if (direct == 0)
				{
					memcpy(ptch, s0, sz);
					memcpy(ptch + 4, s0 + w1, sz);
					memcpy(ptch + 8, s0 + w2, sz);
					memcpy(ptch + 12, s0 + w3, sz);
				}
				else if (direct == 1)
				{
					memcpy(ptch, s0, sz);
					memcpy(ptch + 4, s0 + w1, sz);
					memcpy(ptch + 8, s0 + w2 + 1, sz);
					memcpy(ptch + 12, s0 + w3 + 1, sz);
				}
				else if (direct == 2)
				{
					memcpy(ptch, s0, sz);
					memcpy(ptch + 4, s0 + w1 + 1, sz);
					memcpy(ptch + 8, s0 + w2 + 2, sz);
					memcpy(ptch + 12, s0 + w3 + 3, sz);
				}
				else if (direct == 3)
				{
					memcpy(ptch, s0, sz);
					memcpy(ptch + 4, s0 + w1 + 2, sz);
					memcpy(ptch + 8, s0 + w2 + 4, sz);
					memcpy(ptch + 12, s0 + w3 + 6, sz);

					/**(ptch  ) = *(s0     );
					*(ptch+1) = *(s0   +1);
					*(ptch+2) = *(s0-w1+2);
					*(ptch+3) = *(s0-w1+3);

					*(ptch+4) = *(s0+w1  );
					*(ptch+5) = *(s0+w1+1);
					*(ptch+6) = *(s0   +2);
					*(ptch+7) = *(s0   +3);

					*(ptch+ 8) = *(s0+w2  );
					*(ptch+ 9) = *(s0+w2+1);
					*(ptch+10) = *(s0+w1+2);
					*(ptch+11) = *(s0+w1+3);

					*(ptch+12) = *(s0+w3  );
					*(ptch+13) = *(s0+w3+1);
					*(ptch+14) = *(s0+w2+2);
					*(ptch+15) = *(s0+w2+3);*/

				}
				else if (direct == -3)
				{
					memcpy(ptch, s0, sz);
					memcpy(ptch + 4, s0 + w1 - 2, sz);
					memcpy(ptch + 8, s0 + w2 - 4, sz);
					memcpy(ptch + 12, s0 + w3 - 6, sz);
				}
				else if (direct == -1)
				{
					memcpy(ptch, s0, sz);
					memcpy(ptch + 4, s0 + w1, sz);
					memcpy(ptch + 8, s0 + w2 - 1, sz);
					memcpy(ptch + 12, s0 + w3 - 1, sz);
				}
				else if (direct == -2)
				{
					memcpy(ptch, s0, sz);
					memcpy(ptch + 4, s0 + w1 - 1, sz);
					memcpy(ptch + 8, s0 + w2 - 2, sz);
					memcpy(ptch + 12, s0 + w3 - 3, sz);
				}


				fDCT4x4_32f_and_threshold_and_iDCT4x4_32f(patch.ptr<float>(0), thresh);

				//#define _HARD_THRESHOLDING_
				//#ifdef _HARD_THRESHOLDING_
				//				dct4x4_llm_sse(patch.ptr<float>(0),patch.ptr<float>(0), buff.ptr<float>(0),0);
				//				//dct(patch,patch);
				//#ifdef _KEEP_00_COEF_
				//				float f0=*(float*)patch.data;
				//#endif
				//				compare(abs(patch),thresh,mask,CMP_LT);
				//				patch.setTo(0.f,mask);
				//
				//#ifdef _KEEP_00_COEF_
				//				*(float*)patch.data = f0;
				//#endif
				//				dct4x4_llm_sse(patch.ptr<float>(0),patch.ptr<float>(0), buff.ptr<float>(0),DCT_INVERSE);
				//				//dct(patch,patch,DCT_INVERSE);
				//
				//#else
				//				dct(patch,patch);
				//				Mat dst;
				//				max(abs(patch)-25*thresh,0.f,dst);
				//				compare(patch,0,mask,CMP_LT);
				//				Mat(-1*dst).copyTo(dst,mask);
				//				dst.at<float>(0,0)=patch.at<float>(0,0);
				//				dct(dst,patch,DCT_INVERSE);
				//#endif

				//add data

				if (direct == 0)
				{
					for (int jp = 0; jp < 4; jp++)
					{
						float* s = patch.ptr<float>(jp);
						float* d = &d0[(jp)*width];
						__m128 mp1 = _mm_load_ps(s);
						__m128 sp1 = _mm_loadu_ps(d);

						_mm_storeu_ps(d, _mm_add_ps(sp1, mp1));
					}
				}
				else if (direct == 1)
				{
					for (int jp = 0; jp < 4; jp++)
					{
						float* s = patch.ptr<float>(jp);
						float* d = &d0[(jp)*width + jp / 2];
						__m128 mp1 = _mm_load_ps(s);
						__m128 sp1 = _mm_loadu_ps(d);

						_mm_storeu_ps(d, _mm_add_ps(sp1, mp1));
					}
				}
				else if (direct == 2)
				{
					for (int jp = 0; jp < 4; jp++)
					{
						float* s = patch.ptr<float>(jp);
						float* d = &d0[(jp)*width + jp];
						__m128 mp1 = _mm_load_ps(s);
						__m128 sp1 = _mm_loadu_ps(d);

						_mm_storeu_ps(d, _mm_add_ps(sp1, mp1));
					}
				}
				else if (direct == 3)
				{
					for (int jp = 0; jp < 4; jp++)
					{
						float* s = patch.ptr<float>(jp);
						float* d = &d0[(jp)*width + 2 * jp];
						__m128 mp1 = _mm_load_ps(s);
						__m128 sp1 = _mm_loadu_ps(d);

						_mm_storeu_ps(d, _mm_add_ps(sp1, mp1));
					}
				}
				else if (direct == -3)
				{
					for (int jp = 0; jp < 4; jp++)
					{
						float* s = patch.ptr<float>(jp);
						float* d = &d0[(jp)*width - 2 * jp];
						__m128 mp1 = _mm_load_ps(s);
						__m128 sp1 = _mm_loadu_ps(d);

						_mm_storeu_ps(d, _mm_add_ps(sp1, mp1));
					}
				}
				/*else if(direct==3)
				{
				for (int jp = 0; jp < 4; jp ++)
				{
				float* s =patch.ptr<float>(jp);
				d0[(jp)*width+0] += s[0];
				d0[(jp)*width+1] += s[1];
				d0[(jp)*width+2] += s[2];
				d0[(jp)*width+3] += s[3];
				}
				}*/
				else if (direct == -1)
				{
					for (int jp = 0; jp < 4; jp++)
					{
						float* s = patch.ptr<float>(jp);
						float* d = &d0[(jp)*width - jp / 2];
						__m128 mp1 = _mm_load_ps(s);
						__m128 sp1 = _mm_loadu_ps(d);

						_mm_storeu_ps(d, _mm_add_ps(sp1, mp1));
					}
				}
				else if (direct == -2)
				{
					for (int jp = 0; jp < 4; jp++)
					{
						float* s = patch.ptr<float>(jp);
						float* d = &d0[(jp)*width - jp];
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

class DenoiseWeightedDCTShrinkageInvorker8x8 : public cv::ParallelLoopBody
{
private:
	float* src;
	float* dest;
	float* weight;
	float thresh;
	int width;
	int height;

public:

	DenoiseWeightedDCTShrinkageInvorker8x8(float *sim, float* dim, float* wmap, float Th, int w, int h) : src(sim), dest(dim), weight(wmap), width(w), height(h), thresh(Th)
	{
		;
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
		int j;

		for (j = range.start; j != range.end; j++)
		{
			Mat patch(Size(8, 8), CV_32F);
			float* ptch = patch.ptr<float>(0);

			float* s0 = &src[width*j];
			float* d0 = &dest[width*j];
			float* w0 = &weight[width*j];
			const int sz = sizeof(float) * 8;
			for (int i = 0; i < wstep; i++)
			{
				memcpy(ptch, s0, sz);
				memcpy(ptch + 8, s0 + w1, sz);
				memcpy(ptch + 16, s0 + w2, sz);
				memcpy(ptch + 24, s0 + w3, sz);
				memcpy(ptch + 32, s0 + w4, sz);
				memcpy(ptch + 40, s0 + w5, sz);
				memcpy(ptch + 48, s0 + w6, sz);
				memcpy(ptch + 56, s0 + w7, sz);

				int v = fDCT8x8_32f_and_threshold_and_iDCT8x8_nonzero_32f(ptch, thresh);

				/*fDCT8x8_32f(patch.ptr<float>(0),patch.ptr<float>(0), buff.ptr<float>(0));
				float f0=*(float*)patch.data;
				compare(abs(patch),thresh,mask,CMP_LT);
				patch.setTo(0.f,mask);
				*(float*)patch.data = f0;
				iDCT8x8_32f(patch.ptr<float>(0),patch.ptr<float>(0), buff.ptr<float>(0));*/
				//dct(patch,patch,DCT_INVERSE);


				const __m128 mw = _mm_set1_ps((float)(v));

				//add data
				for (int jp = 0; jp < 8; jp++)
				{
					float* s = patch.ptr<float>(jp);
					float* d = &d0[(jp)*width];
					float* w = &w0[(jp)*width];
					__m128 mp1 = _mm_load_ps(s);
					__m128 sp1 = _mm_loadu_ps(d);
					__m128 mw1 = _mm_loadu_ps(w);

					_mm_storeu_ps(w, _mm_add_ps(mw1, mw));
					_mm_storeu_ps(d, _mm_add_ps(sp1, _mm_mul_ps(mp1, mw)));

					s += 4;
					d += 4;
					w += 4;

					mp1 = _mm_load_ps(s);
					sp1 = _mm_loadu_ps(d);
					mw1 = _mm_loadu_ps(w);

					_mm_storeu_ps(w, _mm_add_ps(mw1, mw));
					_mm_storeu_ps(d, _mm_add_ps(sp1, _mm_mul_ps(mp1, mw)));
				}
				s0++;
				d0++;
				w0++;
			}
		}
	}
};

class DenoiseWeightedDCTShrinkageInvorker4x4 : public cv::ParallelLoopBody
{
private:
	float* src;
	float* dest;
	float* weight;
	float thresh;
	int width;
	int height;

public:
	DenoiseWeightedDCTShrinkageInvorker4x4(float *sim, float* dim, float* wmap, float Th, int w, int h) : src(sim), dest(dim), weight(wmap), width(w), height(h), thresh(Th)
	{
		;
	}
	virtual void operator() (const Range& range) const
	{
		const int size1 = width * height;
		const int hstep = height - 4 + 1;
		const int wstep = width - 4 + 1;
		const int w1 = 1 * width;
		const int w2 = 2 * width;
		const int w3 = 3 * width;

		int j;
		Mat buff(Size(4, 4), CV_32F); Mat mask;
		for (j = range.start; j != range.end; j++)
		{
			Mat patch(Size(4, 4), CV_32F);
			float* ptch = patch.ptr<float>(0);

			float* s0 = &src[width*j];
			float* d0 = &dest[width*j];
			float* w0 = &weight[width*j];
			const int sz = sizeof(float) * 4;
			for (int i = 0; i < wstep; i++)
			{
				memcpy(ptch, s0, sz);
				memcpy(ptch + 4, s0 + w1, sz);
				memcpy(ptch + 8, s0 + w2, sz);
				memcpy(ptch + 12, s0 + w3, sz);

				//Mat p2 = patch.clone();
				int v = fDCT4x4_32f_and_threshold_and_iDCT4x4_nonzero_32f(patch.ptr<float>(0), thresh);

				//float vv = norm(patch,p2,NORM_L1);
				//add data
				//const __m128 mw = _mm_set1_ps((float)(256.f-v*v));
				const __m128 mw = _mm_set1_ps((float)(v));
				//const __m128 mw = _mm_set1_ps((float)(vv));
				for (int jp = 0; jp < 4; jp++)
				{
					float* s = patch.ptr<float>(jp);
					float* d = &d0[(jp)*width];
					float* w = &w0[(jp)*width];

					__m128 mp1 = _mm_load_ps(s);
					__m128 mw1 = _mm_loadu_ps(w);
					__m128 sp1 = _mm_loadu_ps(d);

					_mm_storeu_ps(w, _mm_add_ps(mw1, mw));
					_mm_storeu_ps(d, _mm_add_ps(sp1, _mm_mul_ps(mw, mp1)));
				}
				s0++;
				d0++;
				w0++;
			}
		}
	}
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//TBB DHT tbbdht///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
class DenoiseDHTShrinkageInvorker16x16 : public cv::ParallelLoopBody
{
private:
	float* src;
	float* dest;
	float thresh;
	int width;
	int height;

public:

	DenoiseDHTShrinkageInvorker16x16(float *sim, float* dim, float Th, int w, int h) : src(sim), dest(dim), width(w), height(h), thresh(Th)
	{
		;
	}
	virtual void operator() (const Range& range) const
	{
		const int size1 = width * height;
		const int hstep = height - 16 + 1;
		const int wstep = width - 16 + 1;

		int j;
		for (j = range.start; j != range.end; j++)
		{
			Mat patch(Size(16, 16), CV_32F);
			float* ptch = patch.ptr<float>(0);

			float* s0 = &src[width*j];
			float* d0 = &dest[width*j];
			const int sz = sizeof(float) * 16;
			for (int i = 0; i < wstep; i++)
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
				s0++;
				d0++;
			}
		}
	}

};

class DenoiseDHTShrinkageInvorker8x8 : public cv::ParallelLoopBody
{
private:
	float* src;
	float* dest;
	float thresh;
	int width;
	int height;

public:

	DenoiseDHTShrinkageInvorker8x8(float *sim, float* dim, float Th, int w, int h) : src(sim), dest(dim), width(w), height(h), thresh(Th)
	{
		;
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
		int j;

		for (j = range.start; j != range.end; j++)
		{
			Mat patch(Size(8, 8), CV_32F);
			float* ptch = patch.ptr<float>(0);

			float* s0 = &src[width*j];
			float* d0 = &dest[width*j];
			const int sz = sizeof(float) * 8;
			for (int i = 0; i < wstep; i++)
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
				s0++;
				d0++;
			}
		}
	}

};

class DenoiseDHTShrinkageInvorker4x4 : public cv::ParallelLoopBody
{
private:
	float* src;
	float* dest;
	float thresh;
	int width;
	int height;

public:

	DenoiseDHTShrinkageInvorker4x4(float *sim, float* dim, float Th, int w, int h) : src(sim), dest(dim), width(w), height(h), thresh(Th)
	{
		;
	}
	virtual void operator() (const Range& range) const
	{
		const int size1 = width * height;
		const int hstep = height - 4 + 1;
		const int wstep = width - 4 + 1;
		const int w1 = 1 * width;
		const int w2 = 2 * width;
		const int w3 = 3 * width;
		int j;

		for (j = range.start; j != range.end; j++)
		{
			Mat patch(Size(4, 4), CV_32F);
			float* ptch = patch.ptr<float>(0);

			float* s0 = &src[width*j];
			float* d0 = &dest[width*j];
			const int sz = sizeof(float) * 4;
			for (int i = 0; i < wstep; i++)
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
				s0++;
				d0++;
			}
		}
	}
};


class DenoiseDHTShrinkageInvorker4x4S : public cv::ParallelLoopBody
{
private:
	float* src;
	float* dest;
	float thresh;
	int width;
	int height;

public:

	DenoiseDHTShrinkageInvorker4x4S(float *sim, float* dim, float Th, int w, int h) : src(sim), dest(dim), width(w), height(h), thresh(Th)
	{
		;
	}
	virtual void operator() (const Range& range) const
	{
		const int size1 = width * height;
		const int hstep = height - 4 + 1;
		const int wstep = width - 4 + 1;
		const int w1 = 1 * width;
		const int w2 = 2 * width;
		const int w3 = 3 * width;
		int j;

		for (j = range.start; j != range.end; j++)
		{
			Mat patch(Size(4, 4), CV_32F);
			float* ptch = patch.ptr<float>(0);

			float* s0 = &src[width*j];
			float* d0 = &dest[width*j];
			const int sz = sizeof(float) * 4;
			for (int i = 0; i < wstep; i++)
			{
				memcpy(ptch, s0, sz);
				memcpy(ptch + 4, s0 + w1 + 1, sz);
				memcpy(ptch + 8, s0 + w2 + 2, sz);
				memcpy(ptch + 12, s0 + w3 + 3, sz);

				Hadamard2D4x4andThreshandIDHT(ptch, thresh);

				//add data
				for (int jp = 0; jp < 4; jp++)
				{
					float* s = patch.ptr<float>(jp);
					float* d = &d0[(jp)*width];
					__m128 mp1 = _mm_load_ps(s);
					__m128 sp1 = _mm_loadu_ps(d + jp);

					_mm_storeu_ps(d + jp, _mm_add_ps(sp1, mp1));
				}
				s0++;
				d0++;
			}
		}
	}
};


class DenoiseDHTShrinkageInvorker4x4S2 : public cv::ParallelLoopBody
{
private:
	float* src;
	float* dest;
	float thresh;
	int width;
	int height;

public:

	DenoiseDHTShrinkageInvorker4x4S2(float *sim, float* dim, float Th, int w, int h) : src(sim), dest(dim), width(w), height(h), thresh(Th)
	{
		;
	}
	virtual void operator() (const Range& range) const
	{
		const int size1 = width * height;
		const int hstep = height - 4 + 1;
		const int wstep = width - 4 + 1;
		const int w1 = 1 * width;
		const int w2 = 2 * width;
		const int w3 = 3 * width;
		int j;

		for (j = range.start; j != range.end; j++)
		{
			Mat patch(Size(4, 4), CV_32F);
			float* ptch = patch.ptr<float>(0);

			float* s0 = &src[width*j];
			float* d0 = &dest[width*j];
			const int sz = sizeof(float) * 4;
			for (int i = 0; i < wstep; i++)
			{
				memcpy(ptch, s0, sz);
				memcpy(ptch + 4, s0 + w1 - 1, sz);
				memcpy(ptch + 8, s0 + w2 - 2, sz);
				memcpy(ptch + 12, s0 + w3 - 3, sz);


				Hadamard2D4x4andThreshandIDHT(ptch, thresh);

				//add data
				for (int jp = 0; jp < 4; jp++)
				{
					float* s = patch.ptr<float>(jp);
					float* d = &d0[(jp)*width];
					__m128 mp1 = _mm_load_ps(s);
					__m128 sp1 = _mm_loadu_ps(d - jp);

					_mm_storeu_ps(d - jp, _mm_add_ps(sp1, mp1));
				}
				s0++;
				d0++;
			}
		}
	}
};





void ivDWT8x8(float* src)
{
	float* s = src;
	for (int i = 0; i < 8; i++)
	{
		float s0 = s[0] + s[32];
		float s1 = s[0] - s[32];
		float s2 = s[8] + s[40];
		float s3 = s[8] - s[40];
		float s4 = s[16] + s[48];
		float s5 = s[16] - s[48];
		float s6 = s[24] + s[56];
		float s7 = s[24] - s[56];

		s[0] = s0; s[16] = s2;
		s[8] = s1; s[24] = s3;
		s[32] = s4; s[48] = s6;
		s[40] = s5; s[56] = s7;
		s++;
	}
}

void fvDWT8x8(float* src)
{
	float* s = src;
	for (int i = 0; i < 8; i++)
	{
		float v0 = (s[0] + s[8])*0.5f;
		float v4 = (s[0] - s[8])*0.5f;
		float v1 = (s[16] + s[24])*0.5f;
		float v5 = (s[16] - s[24])*0.5f;

		float v2 = (s[32] + s[40])*0.5f;
		float v6 = (s[32] - s[40])*0.5f;
		float v3 = (s[48] + s[56])*0.5f;
		float v7 = (s[48] - s[56])*0.5f;

		s[0] = v0; s[16] = v2;
		s[8] = v1; s[24] = v3;
		s[32] = v4; s[48] = v6;
		s[40] = v5; s[56] = v7;
		s++;
	}
}

void ihDWT8x8(float* src)
{
	float* s = src;
	for (int i = 0; i < 8; i++)
	{
		float s0 = s[0] + s[4];
		float s1 = s[0] - s[4];
		float s2 = s[1] + s[5];
		float s3 = s[1] - s[5];
		float s4 = s[2] + s[6];
		float s5 = s[2] - s[6];
		float s6 = s[3] + s[7];
		float s7 = s[3] - s[7];

		s[0] = s0; s[2] = s2;
		s[1] = s1; s[3] = s3;
		s[4] = s4; s[6] = s6;
		s[5] = s5; s[7] = s7;
		s += 8;
	}
}

void fhDWT8x8(float* src)
{
	float* s = src;
	for (int i = 0; i < 8; i++)
	{
		float v0 = (s[0] + s[1])*0.5f;
		float v4 = (s[0] - s[1])*0.5f;
		float v1 = (s[2] + s[3])*0.5f;
		float v5 = (s[2] - s[3])*0.5f;

		float v2 = (s[4] + s[5])*0.5f;
		float v6 = (s[4] - s[5])*0.5f;
		float v3 = (s[6] + s[7])*0.5f;
		float v7 = (s[6] - s[7])*0.5f;

		s[0] = v0; s[2] = v2;
		s[1] = v1; s[3] = v3;
		s[4] = v4; s[6] = v6;
		s[5] = v5; s[7] = v7;
		s += 8;
	}
}

void ihDWT4x4(float* src)
{
	float* s = src;
	for (int i = 0; i < 4; i++)
	{
		float s0 = s[0] + s[2];
		float s1 = s[0] - s[2];
		float s2 = s[1] + s[3];
		float s3 = s[1] - s[3];
		s[0] = s0; s[2] = s2;
		s[1] = s1; s[3] = s3;
		s += 4;
	}
}
void ivDWT4x4(float* src)
{
	float* s = src;
	for (int i = 0; i < 4; i++)
	{
		float s0 = s[0] + s[8];
		float s1 = s[0] - s[8];
		float s2 = s[4] + s[12];
		float s3 = s[4] - s[12];
		s[0] = s0; s[8] = s2;
		s[4] = s1; s[12] = s3;

		s++;
	}
}

void fhDWT4x4(float* src)
{
	float* s = src;
	for (int i = 0; i < 4; i++)
	{
		float v0 = (s[0] + s[1])*0.5f;
		float v2 = (s[0] - s[1])*0.5f;
		float v1 = (s[2] + s[3])*0.5f;
		float v3 = (s[2] - s[3])*0.5f;
		s[0] = v0; s[2] = v2;
		s[1] = v1; s[3] = v3;
		s += 4;
	}
}
void fvDWT4x4(float* src)
{
	float* s = src;
	for (int i = 0; i < 4; i++)
	{
		float v0 = (s[0] + s[4])*0.5f;
		float v2 = (s[0] - s[4])*0.5f;
		float v1 = (s[8] + s[12])*0.5f;
		float v3 = (s[8] - s[12])*0.5f;
		s[0] = v0; s[8] = v2;
		s[4] = v1; s[12] = v3;
		s++;
	}
}


void iDWT4x4(float* src)
{
	ivDWT4x4(src);
	ihDWT4x4(src);
	ivDWT4x4(src);
	ihDWT4x4(src);
}
void fDWT4x4(float* src)
{
	fhDWT4x4(src);
	fvDWT4x4(src);
	fhDWT4x4(src);
	fvDWT4x4(src);
}


void iDWT8x8(float* src)
{
	ivDWT8x8(src);
	ihDWT8x8(src);
	ivDWT8x8(src);
	ihDWT8x8(src);
	ivDWT8x8(src);
	ihDWT8x8(src);
}
void fDWT8x8(float* src)
{
	fhDWT8x8(src);
	fvDWT8x8(src);

	fhDWT8x8(src);
	fvDWT8x8(src);
	fhDWT8x8(src);
	fvDWT8x8(src);
}

void DWT2D4x4andThreshandIDWT(float* src, float thresh)
{

}

void printMat_float(Mat& src)
{
	for (int j = 0; j < src.rows; j++)
	{
		for (int i = 0; i < src.cols; i++)
		{
			cout << format("%5.2f ", src.at<float>(j, i));
		}
		cout << endl;
	}
	cout << endl;
}
class DenoiseDWTShrinkageInvorker8x8 : public cv::ParallelLoopBody
{
private:
	float* src;
	float* dest;
	float thresh;
	int width;
	int height;

public:

	DenoiseDWTShrinkageInvorker8x8(float *sim, float* dim, float Th, int w, int h) : src(sim), dest(dim), width(w), height(h), thresh(Th)
	{
		;
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
		int j;

		for (j = range.start; j != range.end; j++)
		{
			Mat patch(Size(8, 8), CV_32F);
			float* ptch = patch.ptr<float>(0);

			float* s0 = &src[width*j];
			float* d0 = &dest[width*j];
			const int sz = sizeof(float) * 8;
			for (int i = 0; i < wstep; i++)
			{
				memcpy(ptch, s0, sz);
				memcpy(ptch + 8, s0 + w1, sz);
				memcpy(ptch + 16, s0 + w2, sz);
				memcpy(ptch + 24, s0 + w3, sz);
				memcpy(ptch + 32, s0 + w4, sz);
				memcpy(ptch + 40, s0 + w5, sz);
				memcpy(ptch + 48, s0 + w6, sz);
				memcpy(ptch + 56, s0 + w7, sz);

				//if(j==200)
				{

					//printMat_float(patch);
					fDWT8x8(ptch);
					//printMat_float(patch);

					//float th = thresh*0.133;
					float th = 3.2f;

					for (int i = 1; i < 64; i++)
					{
						//ptch[i] = (abs(ptch[i])<th) ? 0.f: ptch[i];
						if (ptch[i] >= 0.f) ptch[i] = max(ptch[i] - th, 0.f);
						else ptch[i] = -max(-ptch[i] - th, 0.f);
					}
					//printMat_float(patch);

					iDWT8x8(ptch);
					//printMat_float(patch);
					//getchar();
				}

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
				s0++;
				d0++;
			}
		}
	}
};

class DenoiseDWTShrinkageInvorker4x4 : public cv::ParallelLoopBody
{
private:
	float* src;
	float* dest;
	float thresh;
	int width;
	int height;

public:

	DenoiseDWTShrinkageInvorker4x4(float *sim, float* dim, float Th, int w, int h) : src(sim), dest(dim), width(w), height(h), thresh(Th)
	{
		;
	}
	virtual void operator() (const Range& range) const
	{
		const int size1 = width * height;
		const int hstep = height - 4 + 1;
		const int wstep = width - 4 + 1;
		const int w1 = 1 * width;
		const int w2 = 2 * width;
		const int w3 = 3 * width;
		int j;

		for (j = range.start; j != range.end; j++)
		{
			Mat patch(Size(4, 4), CV_32F);
			float* ptch = patch.ptr<float>(0);

			float* s0 = &src[width*j];
			float* d0 = &dest[width*j];
			const int sz = sizeof(float) * 4;
			for (int i = 0; i < wstep; i++)
			{
				memcpy(ptch, s0, sz);
				memcpy(ptch + 4, s0 + w1, sz);
				memcpy(ptch + 8, s0 + w2, sz);
				memcpy(ptch + 12, s0 + w3, sz);

				fDWT4x4(ptch);
				float th = thresh*0.2f;
				for (int i = 1; i < 16; i++)
				{
					//ptch[i] = (abs(ptch[i])<th) ? 0.f: ptch[i];
					if (ptch[i] >= 0.f) ptch[i] = max(ptch[i] - th, 0.f);
					else ptch[i] = -max(-ptch[i] - th, 0.f);
				}
				iDWT4x4(ptch);


				//add data
				for (int jp = 0; jp < 4; jp++)
				{
					float* s = patch.ptr<float>(jp);
					float* d = &d0[(jp)*width];
					__m128 mp1 = _mm_load_ps(s);
					__m128 sp1 = _mm_loadu_ps(d);

					_mm_storeu_ps(d, _mm_add_ps(sp1, mp1));
				}
				s0++;
				d0++;
			}
		}
	}
};
///////////////////////////////////////////////////////////////////////////////////////////////////

void RedundantDXTDenoise::init(Size size_, int color_, Size patch_size_)
{
	int w = size_.width + 2 * patch_size_.width;
	w += ((4 - w % 4) % 4);

	channel = color_;

	size = Size(w, size_.height + 2 * patch_size_.height);
	patch_size = patch_size_;
}

RedundantDXTDenoise::RedundantDXTDenoise()
{
	isSSE = true;
}

RedundantDXTDenoise::RedundantDXTDenoise(Size size_, int color, Size patch_size_)
{
	isSSE = true;
	init(size_, color, patch_size_);
}

void RedundantDXTDenoise::div(float* inplace0, float* inplace1, float* inplace2, float* wmap0, float* wmap1, float* wmap2, const int size1)
{
	float* s0 = inplace0;
	float* s1 = inplace1;
	float* s2 = inplace2;

	float* w0 = wmap0;
	float* w1 = wmap1;
	float* w2 = wmap2;

	for (int i = 0; i < size1; i += 4)
	{
		__m128 md0 = _mm_load_ps(s0);
		__m128 md1 = _mm_load_ps(s1);
		__m128 md2 = _mm_load_ps(s2);

		__m128 mdiv0 = _mm_load_ps(w0);
		__m128 mdiv1 = _mm_load_ps(w1);
		__m128 mdiv2 = _mm_load_ps(w2);
		_mm_store_ps(s0, _mm_div_ps(md0, mdiv0));
		_mm_store_ps(s1, _mm_div_ps(md1, mdiv1));
		_mm_store_ps(s2, _mm_div_ps(md2, mdiv2));

		s0 += 4;
		s1 += 4;
		s2 += 4;

		w0 += 4;
		w1 += 4;
		w2 += 4;
	}
}

void RedundantDXTDenoise::div(float* inplace0, float* inplace1, float* inplace2, const int patch_area, const int size1)
{
	float* s0 = inplace0;
	float* s1 = inplace1;
	float* s2 = inplace2;
	const __m128 mdiv = _mm_set1_ps(1.f / (float)patch_area);
	for (int i = 0; i < size1; i += 4)
	{
		__m128 md0 = _mm_load_ps(s0);
		__m128 md1 = _mm_load_ps(s1);
		__m128 md2 = _mm_load_ps(s2);
		_mm_store_ps(s0, _mm_mul_ps(md0, mdiv));
		_mm_store_ps(s1, _mm_mul_ps(md1, mdiv));
		_mm_store_ps(s2, _mm_mul_ps(md2, mdiv));

		s0 += 4;
		s1 += 4;
		s2 += 4;
	}
}

void RedundantDXTDenoise::div(float* inplace0, float* wmap, const int size1)
{
	float* s0 = inplace0;
	float* w0 = wmap;
	for (int i = 0; i < size1; i += 4)
	{
		__m128 md0 = _mm_load_ps(s0);
		__m128 mdiv = _mm_load_ps(w0);
		_mm_store_ps(s0, _mm_div_ps(md0, mdiv));
		s0 += 4;
		w0 += 4;
	}
}

void RedundantDXTDenoise::div(float* inplace0, const int patch_area, const int size1)
{
	float* s0 = inplace0;
	const __m128 mdiv = _mm_set1_ps(1.f / (float)patch_area);
	for (int i = 0; i < size1; i += 4)
	{
		__m128 md0 = _mm_load_ps(s0);
		_mm_store_ps(s0, _mm_mul_ps(md0, mdiv));
		s0 += 4;
	}
}

void RedundantDXTDenoise::decorrelateColorInvert(float* src, float* dest, int width, int height)
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

void RedundantDXTDenoise::decorrelateColorForward(float* src, float* dest, int width, int height)
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

void RedundantDXTDenoise::body(float *src, float* dest, float* wmap, float Th)
{
	if (basis == BASIS::DHT)
	{
		if (isSSE)
		{
			if (patch_size.width == 2)
			{
				DenoiseDCTShrinkageInvorker2x2 invork(src, dest, Th, size.width, size.height);
				parallel_for_(Range(0, size.height - patch_size.height), invork);
			}
			else if (patch_size.width == 4)
			{
				DenoiseDCTShrinkageInvorker4x4 invork(src, dest, Th, size.width, size.height);
				parallel_for_(Range(0, size.height - patch_size.height), invork, 12.0);
			}
			else if (patch_size.width == 8)
			{
				DenoiseDCTShrinkageInvorker8x8 invork(src, dest, Th, size.width, size.height);
				parallel_for_(Range(0, size.height - patch_size.height), invork, 12.0);
			}
			else if (patch_size.width == 16)
			{
				DenoiseDCTShrinkageInvorker16x16 invork(src, dest, Th, size.width, size.height);
				parallel_for_(Range(0, size.height - patch_size.height), invork);
			}
			else
			{
				DenoiseDCTShrinkageInvorker invork(src, dest, Th, size.width, size.height, patch_size);
				parallel_for_(Range(0, size.height - patch_size.height), invork);
			}
		}
		else
		{
			DenoiseDCTShrinkageInvorker invork(src, dest, Th, size.width, size.height, patch_size);
			parallel_for_(Range(0, size.height - patch_size.height), invork);
		}
	}
	else if (basis == BASIS::DHT)
	{
		if (isSSE)
		{
			if (patch_size.width == 2)
			{
				//2x2 is same as DCT
				DenoiseDCTShrinkageInvorker2x2 invork(src, dest, Th, size.width, size.height);
				parallel_for_(Range(0, size.height - patch_size.height), invork);
			}
			else if (patch_size.width == 4)
			{
				ShearableDenoiseDCTShrinkageInvorker4x4 invork(src, dest, Th, size.width, size.height, 0);
				//DenoiseDHTShrinkageInvorker4x4 invork(src,dest,Th, size.width,size.height);			
				parallel_for_(Range(0, size.height - patch_size.height), invork, 12);
			}
			else if (patch_size.width == 8)
			{
				DenoiseDHTShrinkageInvorker8x8 invork(src, dest, Th, size.width, size.height);
				parallel_for_(Range(0, size.height - patch_size.height), invork);
			}
			else if (patch_size.width == 16)
			{
				DenoiseDHTShrinkageInvorker16x16 invork(src, dest, Th, size.width, size.height);
				parallel_for_(Range(0, size.height - patch_size.height), invork);
			}
			else
			{
				DenoiseDCTShrinkageInvorker invork(src, dest, Th, size.width, size.height, patch_size);
				parallel_for_(Range(0, size.height - patch_size.height), invork);
			}
		}
		else
		{
			DenoiseDCTShrinkageInvorker invork(src, dest, Th, size.width, size.height, patch_size);
			parallel_for_(Range(0, size.height - patch_size.height), invork);
		}
	}
	else if (basis == BASIS::DWT)
	{
		if (patch_size.width == 4)
		{
			DenoiseDWTShrinkageInvorker4x4 invork(src, dest, Th, size.width, size.height);
			//DenoiseDHTShrinkageInvorker4x4S invork(src,dest,Th, size.width,size.height);			
			parallel_for_(Range(0, size.height - patch_size.height), invork);
		}
		else if (patch_size.width == 8)
		{
			DenoiseDWTShrinkageInvorker8x8 invork(src, dest, Th, size.width, size.height);
			parallel_for_(Range(0, size.height - patch_size.height), invork);
		}
	}
}

void RedundantDXTDenoise::body(float *src, float* dest, float Th)
{
	if (basis == BASIS::DCT)
	{
		if (isSSE)
		{
			if (patch_size.width == 2)
			{
				DenoiseDCTShrinkageInvorker2x2 invork(src, dest, Th, size.width, size.height);
				parallel_for_(Range(0, size.height - patch_size.height), invork);
			}
			else if (patch_size.width == 4)
			{
				DenoiseDCTShrinkageInvorker4x4 invork(src, dest, Th, size.width, size.height);
				parallel_for_(Range(0, size.height - patch_size.height), invork, 12.0);
			}
			else if (patch_size.width == 8)
			{
				DenoiseDCTShrinkageInvorker8x8 invork(src, dest, Th, size.width, size.height);
				parallel_for_(Range(0, size.height - patch_size.height), invork, 12.0);
			}
			else if (patch_size.width == 16)
			{
				DenoiseDCTShrinkageInvorker16x16 invork(src, dest, Th, size.width, size.height);
				parallel_for_(Range(0, size.height - patch_size.height), invork);
			}
			else
			{
				DenoiseDCTShrinkageInvorker invork(src, dest, Th, size.width, size.height, patch_size);
				parallel_for_(Range(0, size.height - patch_size.height), invork);
			}
		}
		else
		{
			DenoiseDCTShrinkageInvorker invork(src, dest, Th, size.width, size.height, patch_size);
			parallel_for_(Range(0, size.height - patch_size.height), invork);
		}
	}
	else if (basis == BASIS::DHT)
	{
		if (isSSE)
		{
			if (patch_size.width == 2)
			{
				//2x2 is same as DCT
				DenoiseDCTShrinkageInvorker2x2 invork(src, dest, Th, size.width, size.height);
				parallel_for_(Range(0, size.height - patch_size.height), invork);
			}
			else if (patch_size.width == 4)
			{
				DenoiseDHTShrinkageInvorker4x4 invork(src,dest,Th, size.width,size.height);			
				parallel_for_(Range(0, size.height - patch_size.height), invork, 12);
			}
			else if (patch_size.width == 8)
			{
				DenoiseDHTShrinkageInvorker8x8 invork(src, dest, Th, size.width, size.height);
				parallel_for_(Range(0, size.height - patch_size.height), invork);
			}
			else if (patch_size.width == 16)
			{
				DenoiseDHTShrinkageInvorker16x16 invork(src, dest, Th, size.width, size.height);
				parallel_for_(Range(0, size.height - patch_size.height), invork);
			}
			else
			{
				DenoiseDCTShrinkageInvorker invork(src, dest, Th, size.width, size.height, patch_size);
				parallel_for_(Range(0, size.height - patch_size.height), invork);
			}
		}
		else
		{
			DenoiseDCTShrinkageInvorker invork(src, dest, Th, size.width, size.height, patch_size);
			parallel_for_(Range(0, size.height - patch_size.height), invork);
		}
	}
	else if (basis == BASIS::DWT)
	{
		if (patch_size.width == 4)
		{
			DenoiseDWTShrinkageInvorker4x4 invork(src, dest, Th, size.width, size.height);
			//DenoiseDHTShrinkageInvorker4x4S invork(src,dest,Th, size.width,size.height);			
			parallel_for_(Range(0, size.height - patch_size.height), invork);
		}
		else if (patch_size.width == 8)
		{
			DenoiseDWTShrinkageInvorker8x8 invork(src, dest, Th, size.width, size.height);
			parallel_for_(Range(0, size.height - patch_size.height), invork);
		}
	}
}

void RedundantDXTDenoise::body(float *src, float* dest, float Th, int dr)
{
	if (basis == BASIS::DHT)
	{
		if (isSSE)
		{
			if (patch_size.width == 2)
			{
				DenoiseDCTShrinkageInvorker2x2 invork(src, dest, Th, size.width, size.height);
				parallel_for_(Range(0, size.height - patch_size.height), invork);
			}
			else if (patch_size.width == 4)
			{
				ShearableDenoiseDCTShrinkageInvorker4x4 invork(src, dest, Th, size.width, size.height, dr);
				parallel_for_(Range(0, size.height - patch_size.height), invork, 12.0);
			}
			else if (patch_size.width == 8)
			{
				DenoiseDCTShrinkageInvorker8x8 invork(src, dest, Th, size.width, size.height);
				parallel_for_(Range(0, size.height - patch_size.height), invork);
			}
			else
			{
				DenoiseDCTShrinkageInvorker invork(src, dest, Th, size.width, size.height, patch_size);
				parallel_for_(Range(0, size.height - patch_size.height), invork);
			}
		}
		else
		{
			DenoiseDCTShrinkageInvorker invork(src, dest, Th, size.width, size.height, patch_size);
			parallel_for_(Range(0, size.height - patch_size.height), invork);
		}
	}
	else if (basis == BASIS::DHT)
	{
		if (isSSE)
		{
			if (patch_size.width == 2)
			{
				//2x2 is same as DCT
				DenoiseDCTShrinkageInvorker2x2 invork(src, dest, Th, size.width, size.height);
				parallel_for_(Range(0, size.height - patch_size.height), invork);
			}
			else if (patch_size.width == 4)
			{
				ShearableDenoiseDCTShrinkageInvorker4x4 invork(src, dest, Th, size.width, size.height, dr);
				//DenoiseDHTShrinkageInvorker4x4 invork(src,dest,Th, size.width,size.height);			
				parallel_for_(Range(0, size.height - patch_size.height), invork, 12);
			}
			else if (patch_size.width == 8)
			{
				DenoiseDHTShrinkageInvorker8x8 invork(src, dest, Th, size.width, size.height);
				parallel_for_(Range(0, size.height - patch_size.height), invork);
			}
			else if (patch_size.width == 16)
			{
				DenoiseDHTShrinkageInvorker16x16 invork(src, dest, Th, size.width, size.height);
				parallel_for_(Range(0, size.height - patch_size.height), invork);
			}
			else
			{
				DenoiseDCTShrinkageInvorker invork(src, dest, Th, size.width, size.height, patch_size);
				parallel_for_(Range(0, size.height - patch_size.height), invork);
			}
		}
		else
		{
			DenoiseDCTShrinkageInvorker invork(src, dest, Th, size.width, size.height, patch_size);
			parallel_for_(Range(0, size.height - patch_size.height), invork);
		}
	}
	else if (basis == BASIS::DWT)
	{
		if (patch_size.width == 4)
		{
			DenoiseDWTShrinkageInvorker4x4 invork(src, dest, Th, size.width, size.height);
			//DenoiseDHTShrinkageInvorker4x4S invork(src,dest,Th, size.width,size.height);			
			parallel_for_(Range(0, size.height - patch_size.height), invork);
		}
		else if (patch_size.width == 8)
		{
			DenoiseDWTShrinkageInvorker8x8 invork(src, dest, Th, size.width, size.height);
			parallel_for_(Range(0, size.height - patch_size.height), invork);
		}
	}
}
/*
void RedundantDXTDenoise::shearable(Mat& src_, Mat& dest, float sigma, Size psize, int transform_basis, int direct)
{
Mat src;
if (src_.depth() != CV_32F)src_.convertTo(src, CV_MAKETYPE(CV_32F, src_.channels()));
else src = src_;

basis = transform_basis;
if (src.size() != size || src.channels() != channel || psize != patch_size) init(src.size(), src.channels(), psize);

int w = src.cols + 4 * psize.width;
w = ((4 - w % 4) % 4);
Mat im; copyMakeBorder(src, im, psize.height, psize.height, 2 * psize.width, 2 * psize.width + w, cv::BORDER_REPLICATE);

const int width = im.cols;
const int height = im.rows;
const int size1 = width*height;
float* ipixels;
float* opixels;

// Threshold
float Th = 3 * sigma;

// DCT window size
{
#ifdef _CALCTIME_
CalcTime t("color");
#endif
if (channel == 3)
{
cvtColorOrder32F_BGR2BBBBGGGGRRRR(im, buff);
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

{
#ifdef _CALCTIME_
CalcTime t("body");
#endif
if (channel == 3)
{
Size s = size;
size = Size(size.width + 2 * patch_size.width, size.height);
body(ipixels, opixels, Th, direct);
body(ipixels + size1, opixels + size1, Th, direct);
body(ipixels + 2 * size1, opixels + 2 * size1, Th, direct);
size = s;
}
else
{
body(ipixels, opixels, Th, direct);
}
//body(ipixels, opixels,ipixels+size1, opixels+size1,ipixels+2*size1, opixels+2*size1,Th);

}

{
#ifdef _CALCTIME_
CalcTime t("div");
#endif

if (channel == 3)
{
float* d0 = &opixels[0];
float* d1 = &opixels[size1];
float* d2 = &opixels[2 * size1];
div(d0, d1, d2, patch_size.area(), size1);

}
else
{
float* d0 = &opixels[0];
div(d0, patch_size.area(), size1);
}
}

{
#ifdef _CALCTIME_
CalcTime t("inv color");
#endif
// inverse 3-point DCT transform in the color dimension
if (channel == 3)
{
decorrelateColorInvert(opixels, opixels, width, height);
cvtColorOrder32F_BBBBGGGGRRRR2BGR(sum, im);
}
else
{
im = sum;
}

Mat im2;
if (src_.depth() != CV_32F) im.convertTo(im2, src_.type());
else im2 = im;

Mat(im2(Rect(patch_size.width * 2, patch_size.height, src.cols, src.rows))).copyTo(dest);
}
}

void RedundantDXTDenoise::weighted(Mat& src_, Mat& dest, float sigma, Size psize, int transform_basis)
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

Mat weight0 = Mat::zeros(Size(width, height), CV_32F);
Mat weight1 = Mat::zeros(Size(width, height), CV_32F);
Mat weight2 = Mat::zeros(Size(width, height), CV_32F);
// DCT window size
{
#ifdef _CALCTIME_
CalcTime t("color");
#endif
if (channel == 3)
{
cvtColorOrder32F_BGR2BBBBGGGGRRRR(im, buff);
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

{
#ifdef _CALCTIME_
CalcTime t("body");
#endif
if (channel == 3)
{
float* w0 = weight0.ptr<float>(0);
float* w1 = weight1.ptr<float>(0);
float* w2 = weight2.ptr<float>(0);

body(ipixels, opixels, w0, Th);
body(ipixels + size1, opixels + size1, w1, Th);
body(ipixels + 2 * size1, opixels + 2 * size1, w2, Th);
}
else
{
body(ipixels, opixels, (float*)weight0.data, Th);
}
//body(ipixels, opixels,ipixels+size1, opixels+size1,ipixels+2*size1, opixels+2*size1,Th);

}

{
#ifdef _CALCTIME_
CalcTime t("div");
#endif

if (channel == 3)
{
float* d0 = &opixels[0];
float* d1 = &opixels[size1];
float* d2 = &opixels[2 * size1];

float* w0 = weight0.ptr<float>(0);
float* w1 = weight1.ptr<float>(0);
float* w2 = weight2.ptr<float>(0);

div(d0, d1, d2, w0, w1, w2, size1);

//guiAlphaBlend(weight0,weight1);
//guiAlphaBlend(weight0,weight2);
}
else
{
float* d0 = &opixels[0];
float* w0 = (float*)weight0.data;
div(d0, w0, size1);
}
}

{
#ifdef _CALCTIME_
CalcTime t("inv color");
#endif
// inverse 3-point DCT transform in the color dimension
if (channel == 3)
{
decorrelateColorInvert(opixels, opixels, width, height);
cvtColorOrder32F_BBBBGGGGRRRR2BGR(sum, im);
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

void RedundantDXTDenoise::test(Mat& src, Mat& dest, float sigma, Size psize)
{
if (src.size() != size || src.channels() != channel || psize != patch_size) init(src.size(), src.channels(), psize);
sum.setTo(0);
copyMakeBorder(src, im, psize.height, psize.height, psize.width, psize.width, cv::BORDER_REPLICATE);

const int width = im.cols;
const int height = im.rows;
const int size1 = width*height;

float* ipixels;
float* opixels;

// Threshold
float Th = 3 * sigma;

// DCT window size
int width_p, height_p;
width_p = psize.width;
height_p = psize.height;

{
#ifdef _CALCTIME_
CalcTime t("color");
#endif
cvtColorOrder32F_BGR2BBBBGGGGRRRR(im, buff);
sum = Mat::zeros(buff.size(), CV_32F);
ipixels = buff.ptr<float>(0);
opixels = sum.ptr<float>(0);

decorrelateColorForward(ipixels, ipixels, width, height);
}

{
#ifdef _CALCTIME_
CalcTime t("body");
#endif
bodyTest(ipixels, opixels, Th);
bodyTest(ipixels + size1, opixels + size1, Th);
bodyTest(ipixels + 2 * size1, opixels + 2 * size1, Th);
//body(ipixels, opixels,ipixels+size1, opixels+size1,ipixels+2*size1, opixels+2*size1,Th);

}
{
#ifdef _CALCTIME_
CalcTime t("div");
#endif
float* d0 = &opixels[0];
float* d1 = &opixels[size1];
float* d2 = &opixels[2 * size1];
div(d0, d1, d2, patch_size.area(), size1);
}

{
#ifdef _CALCTIME_
CalcTime t("inv color");
#endif
// inverse 3-point DCT transform in the color dimension

decorrelateColorInvert(opixels, opixels, width, height);
cvtColorOrder32F_BBBBGGGGRRRR2BGR(sum, im);
Mat(im(Rect(patch_size.width, patch_size.height, src.cols, src.rows))).copyTo(dest);
}
}
*/
void RedundantDXTDenoise::operator()(Mat& src_, Mat& dest, float sigma, Size psize, BASIS transform_basis)
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

	// DCT window size
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
		//body(ipixels, opixels,ipixels+size1, opixels+size1,ipixels+2*size1, opixels+2*size1,Th);

	}

	{
#ifdef _CALCTIME_
		CalcTime t("div");
#endif

		if (channel == 3)
		{
			float* d0 = &opixels[0];
			float* d1 = &opixels[size1];
			float* d2 = &opixels[2 * size1];
			div(d0, d1, d2, patch_size.area(), size1);
		}
		else
		{
			float* d0 = &opixels[0];
			div(d0, patch_size.area(), size1);
		}
	}

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
//}