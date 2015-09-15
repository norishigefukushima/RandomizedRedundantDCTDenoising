#include <nmmintrin.h> //SSE4.2
#define _USE_MATH_DEFINES
#include <math.h>


void transpose4x4(float* inplace);
void transpose4x4(const float* src, float* dest);
void transpose8x8(float* inplace);
void transpose8x8(const float* src, float* dest);
void transpose16x16(float* inplace);
void transpose16x16(const float* src, float* dest);

inline int getNonzero(float* s, int size)
{
	int ret = 0;
	for (int i = 0; i < size; i++)
	{
		if (s[i] != 0.f)ret++;
	}

	return ret;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//16x16//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Plonka, Gerlind, and Manfred Tasche. "Fast and numerically stable algorithms for discrete cosine transforms." Linear algebra and its applications 394 (2005) : 309 - 345.
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void fDCT16x16_threshold_keep00_iDCT16x16(const float* src, float* dest, float th);
void iDCT16x16(const float* src, float* dest);
void fDCT16x16(const float* src, float* dest);

static void fdct161d_base(const float *src, float *dst)
{
	for (int i = 0; i < 16; i++)
	{
		const float mx00 = src[0] + src[15];
		const float mx01 = src[1] + src[14];
		const float mx02 = src[2] + src[13];
		const float mx03 = src[3] + src[12];
		const float mx04 = src[4] + src[11];
		const float mx05 = src[5] + src[10];
		const float mx06 = src[6] + src[9];
		const float mx07 = src[7] + src[8];
		const float mx08 = src[0] - src[15];
		const float mx09 = src[1] - src[14];
		const float mx0a = src[2] - src[13];
		const float mx0b = src[3] - src[12];
		const float mx0c = src[4] - src[11];
		const float mx0d = src[5] - src[10];
		const float mx0e = src[6] - src[9];
		const float mx0f = src[7] - src[8];

		const float mx10 = mx00 + mx07;
		const float mx11 = mx01 + mx06;
		const float mx12 = mx02 + mx05;
		const float mx13 = mx03 + mx04;
		const float mx14 = mx00 - mx07;
		const float mx15 = mx01 - mx06;
		const float mx16 = mx02 - mx05;
		const float mx17 = mx03 - mx04;
		const float mx18 = mx10 + mx13;
		const float mx19 = mx11 + mx12;
		const float mx1a = mx10 - mx13;
		const float mx1b = mx11 - mx12;

		const float mx1c = 1.38703984532215f*mx14 + 0.275899379282943f*mx17;
		const float mx1d = 1.17587560241936f*mx15 + 0.785694958387102f*mx16;
		const float mx1e = -0.785694958387102f*mx15 + 1.17587560241936f *mx16;
		const float mx1f = 0.275899379282943f*mx14 - 1.38703984532215f *mx17;
		const float mx20 = 0.25f * (mx1c - mx1d);
		const float mx21 = 0.25f * (mx1e - mx1f);
		const float mx22 = 1.40740373752638f *mx08 + 0.138617169199091f*mx0f;
		const float mx23 = 1.35331800117435f *mx09 + 0.410524527522357f*mx0e;
		const float mx24 = 1.24722501298667f *mx0a + 0.666655658477747f*mx0d;
		const float mx25 = 1.09320186700176f *mx0b + 0.897167586342636f*mx0c;
		const float mx26 = -0.897167586342636f*mx0b + 1.09320186700176f *mx0c;
		const float mx27 = 0.666655658477747f*mx0a - 1.24722501298667f *mx0d;
		const float mx28 = -0.410524527522357f*mx09 + 1.35331800117435f *mx0e;
		const float mx29 = 0.138617169199091f*mx08 - 1.40740373752638f *mx0f;
		const float mx2a = mx22 + mx25;
		const float mx2b = mx23 + mx24;
		const float mx2c = mx22 - mx25;
		const float mx2d = mx23 - mx24;
		const float mx2e = 0.25f * (mx2a - mx2b);
		const float mx2f = 0.326640741219094f*mx2c + 0.135299025036549f*mx2d;
		const float mx30 = 0.135299025036549f*mx2c - 0.326640741219094f*mx2d;
		const float mx31 = mx26 + mx29;
		const float mx32 = mx27 + mx28;
		const float mx33 = mx26 - mx29;
		const float mx34 = mx27 - mx28;
		const float mx35 = 0.25f * (mx31 - mx32);
		const float mx36 = 0.326640741219094f*mx33 + 0.135299025036549f*mx34;
		const float mx37 = 0.135299025036549f*mx33 - 0.326640741219094f*mx34;

		dst[0] = 0.25f * (mx18 + mx19);
		dst[1] = 0.25f * (mx2a + mx2b);
		dst[2] = 0.25f * (mx1c + mx1d);
		dst[3] = 0.707106781186547f * (mx2f - mx37);
		dst[4] = 0.326640741219094f*mx1a + 0.135299025036549f*mx1b;
		dst[5] = 0.707106781186547f * (mx2f + mx37);
		dst[6] = 0.707106781186547f * (mx20 - mx21);
		dst[7] = 0.707106781186547f * (mx2e + mx35);
		dst[8] = 0.25f * (mx18 - mx19);
		dst[9] = 0.707106781186547f * (mx2e - mx35);
		dst[10] = 0.707106781186547f * (mx20 + mx21);
		dst[11] = 0.707106781186547f * (mx30 - mx36);
		dst[12] = 0.135299025036549f*mx1a - 0.326640741219094f*mx1b;
		dst[13] = 0.707106781186547f * (mx30 + mx36);
		dst[14] = 0.25f * (mx1e + mx1f);
		dst[15] = 0.25f * (mx31 + mx32);

		dst += 16;
		src += 16;
	}
}

static void fdct161d_sse(const float *s, float *d)
{
	float* src = (float*)s;
	float* dst = d;

	const __m128 c0135 = _mm_set1_ps(0.135299025036549f);
	const __m128 c0326 = _mm_set1_ps(0.326640741219094f);
	const __m128 c0250 = _mm_set1_ps(0.25f);
	const __m128 c0707 = _mm_set1_ps(0.707106781186547f);
	const __m128 c1414 = _mm_set1_ps(1.4142135623731f);

	for (int i = 0; i < 4; i++)
	{
		__m128 ms0 = _mm_load_ps(src);
		__m128 ms1 = _mm_load_ps(src + 16);
		__m128 ms2 = _mm_load_ps(src + 32);
		__m128 ms3 = _mm_load_ps(src + 48);
		__m128 ms4 = _mm_load_ps(src + 64);
		__m128 ms5 = _mm_load_ps(src + 80);
		__m128 ms6 = _mm_load_ps(src + 96);
		__m128 ms7 = _mm_load_ps(src + 112);
		__m128 ms8 = _mm_load_ps(src + 128);
		__m128 ms9 = _mm_load_ps(src + 144);
		__m128 ms10 = _mm_load_ps(src + 160);
		__m128 ms11 = _mm_load_ps(src + 176);
		__m128 ms12 = _mm_load_ps(src + 192);
		__m128 ms13 = _mm_load_ps(src + 208);
		__m128 ms14 = _mm_load_ps(src + 224);
		__m128 ms15 = _mm_load_ps(src + 240);

		__m128 mx00 = _mm_add_ps(ms0, ms15);
		__m128 mx01 = _mm_add_ps(ms1, ms14);
		__m128 mx02 = _mm_add_ps(ms2, ms13);
		__m128 mx03 = _mm_add_ps(ms3, ms12);
		__m128 mx04 = _mm_add_ps(ms4, ms11);
		__m128 mx05 = _mm_add_ps(ms5, ms10);
		__m128 mx06 = _mm_add_ps(ms6, ms9);
		__m128 mx07 = _mm_add_ps(ms7, ms8);
		__m128 mx08 = _mm_sub_ps(ms0, ms15);
		__m128 mx09 = _mm_sub_ps(ms1, ms14);
		__m128 mx0a = _mm_sub_ps(ms2, ms13);
		__m128 mx0b = _mm_sub_ps(ms3, ms12);
		__m128 mx0c = _mm_sub_ps(ms4, ms11);
		__m128 mx0d = _mm_sub_ps(ms5, ms10);
		__m128 mx0e = _mm_sub_ps(ms6, ms9);
		__m128 mx0f = _mm_sub_ps(ms7, ms8);

		__m128 mx10 = _mm_add_ps(mx00, mx07);
		__m128 mx11 = _mm_add_ps(mx01, mx06);
		__m128 mx12 = _mm_add_ps(mx02, mx05);
		__m128 mx13 = _mm_add_ps(mx03, mx04);
		__m128 mx14 = _mm_sub_ps(mx00, mx07);
		__m128 mx15 = _mm_sub_ps(mx01, mx06);
		__m128 mx16 = _mm_sub_ps(mx02, mx05);
		__m128 mx17 = _mm_sub_ps(mx03, mx04);
		__m128 mx18 = _mm_add_ps(mx10, mx13);
		__m128 mx19 = _mm_add_ps(mx11, mx12);
		__m128 mx1a = _mm_sub_ps(mx10, mx13);
		__m128 mx1b = _mm_sub_ps(mx11, mx12);

		__m128 mx1c = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.38703984532215f), mx14), _mm_mul_ps(_mm_set1_ps(0.275899379282943f), mx17));
		__m128 mx1d = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.17587560241936f), mx15), _mm_mul_ps(_mm_set1_ps(0.785694958387102f), mx16));
		__m128 mx1e = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-0.785694958387102f), mx15), _mm_mul_ps(_mm_set1_ps(1.17587560241936f), mx16));
		__m128 mx1f = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.275899379282943f), mx14), _mm_mul_ps(_mm_set1_ps(-1.38703984532215f), mx17));
		__m128 mx20 = _mm_mul_ps(c0250, _mm_sub_ps(mx1c, mx1d));
		__m128 mx21 = _mm_mul_ps(c0250, _mm_sub_ps(mx1e, mx1f));
		__m128 mx22 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.40740373752638f), mx08), _mm_mul_ps(_mm_set1_ps(0.138617169199091f), mx0f));
		__m128 mx23 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.35331800117435f), mx09), _mm_mul_ps(_mm_set1_ps(0.410524527522357f), mx0e));
		__m128 mx24 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.24722501298667f), mx0a), _mm_mul_ps(_mm_set1_ps(0.666655658477747f), mx0d));
		__m128 mx25 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.09320186700176f), mx0b), _mm_mul_ps(_mm_set1_ps(0.897167586342636f), mx0c));
		__m128 mx26 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-0.897167586342636f), mx0b), _mm_mul_ps(_mm_set1_ps(1.09320186700176f), mx0c));
		__m128 mx27 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.666655658477747f), mx0a), _mm_mul_ps(_mm_set1_ps(-1.24722501298667f), mx0d));
		__m128 mx28 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-0.410524527522357f), mx09), _mm_mul_ps(_mm_set1_ps(1.35331800117435f), mx0e));
		__m128 mx29 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.138617169199091f), mx08), _mm_mul_ps(_mm_set1_ps(-1.40740373752638f), mx0f));
		__m128 mx2a = _mm_add_ps(mx22, mx25);
		__m128 mx2b = _mm_add_ps(mx23, mx24);
		__m128 mx2c = _mm_sub_ps(mx22, mx25);
		__m128 mx2d = _mm_sub_ps(mx23, mx24);
		__m128 mx2e = _mm_mul_ps(c0250, _mm_sub_ps(mx2a, mx2b));
		__m128 mx2f = _mm_add_ps(_mm_mul_ps(c0326, mx2c), _mm_mul_ps(c0135, mx2d));
		__m128 mx30 = _mm_sub_ps(_mm_mul_ps(c0135, mx2c), _mm_mul_ps(c0326, mx2d));
		__m128 mx31 = _mm_add_ps(mx26, mx29);
		__m128 mx32 = _mm_add_ps(mx27, mx28);
		__m128 mx33 = _mm_sub_ps(mx26, mx29);
		__m128 mx34 = _mm_sub_ps(mx27, mx28);
		__m128 mx35 = _mm_mul_ps(c0250, _mm_sub_ps(mx31, mx32));
		__m128 mx36 = _mm_add_ps(_mm_mul_ps(c0326, mx33), _mm_mul_ps(c0135, mx34));
		__m128 mx37 = _mm_sub_ps(_mm_mul_ps(c0135, mx33), _mm_mul_ps(c0326, mx34));

		_mm_store_ps(dst + 0, _mm_mul_ps(c0250, _mm_add_ps(mx18, mx19)));
		_mm_store_ps(dst + 16, _mm_mul_ps(c0250, _mm_add_ps(mx2a, mx2b)));
		_mm_store_ps(dst + 32, _mm_mul_ps(c0250, _mm_add_ps(mx1c, mx1d)));
		_mm_store_ps(dst + 48, _mm_mul_ps(c0707, _mm_sub_ps(mx2f, mx37)));
		_mm_store_ps(dst + 64, _mm_add_ps(_mm_mul_ps(c0326, mx1a), _mm_mul_ps(c0135, mx1b)));
		_mm_store_ps(dst + 80, _mm_mul_ps(c0707, _mm_add_ps(mx2f, mx37)));
		_mm_store_ps(dst + 96, _mm_mul_ps(c0707, _mm_sub_ps(mx20, mx21)));
		_mm_store_ps(dst + 112, _mm_mul_ps(c0707, _mm_add_ps(mx2e, mx35)));
		_mm_store_ps(dst + 128, _mm_mul_ps(c0250, _mm_sub_ps(mx18, mx19)));
		_mm_store_ps(dst + 144, _mm_mul_ps(c0707, _mm_sub_ps(mx2e, mx35)));
		_mm_store_ps(dst + 160, _mm_mul_ps(c0707, _mm_add_ps(mx20, mx21)));
		_mm_store_ps(dst + 176, _mm_mul_ps(c0707, _mm_sub_ps(mx30, mx36)));
		_mm_store_ps(dst + 192, _mm_sub_ps(_mm_mul_ps(c0135, mx1a), _mm_mul_ps(c0326, mx1b)));
		_mm_store_ps(dst + 208, _mm_mul_ps(c0707, _mm_add_ps(mx30, mx36)));
		_mm_store_ps(dst + 224, _mm_mul_ps(c0250, _mm_add_ps(mx1e, mx1f)));
		_mm_store_ps(dst + 240, _mm_mul_ps(c0250, _mm_add_ps(mx31, mx32)));

		dst += 4;
		src += 4;
	}
}

static void fdct161d_threshold_keep00_sse(const float *s, float *d, float th)
{
	const int __declspec(align(16)) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
	const __m128 mth = _mm_set1_ps(th);
	const __m128 zeros = _mm_setzero_ps();

	float* src = (float*)s;
	float* dst = d;

	const __m128 c0135 = _mm_set1_ps(0.135299025036549f);
	const __m128 c0326 = _mm_set1_ps(0.326640741219094f);
	const __m128 c0250 = _mm_set1_ps(0.25f);
	const __m128 c0707 = _mm_set1_ps(0.707106781186547f);
	const __m128 c1414 = _mm_set1_ps(1.4142135623731f);

	for (int i = 0; i < 4; i++)
	{
		__m128 ms0 = _mm_load_ps(src);
		__m128 ms1 = _mm_load_ps(src + 16);
		__m128 ms2 = _mm_load_ps(src + 32);
		__m128 ms3 = _mm_load_ps(src + 48);
		__m128 ms4 = _mm_load_ps(src + 64);
		__m128 ms5 = _mm_load_ps(src + 80);
		__m128 ms6 = _mm_load_ps(src + 96);
		__m128 ms7 = _mm_load_ps(src + 112);
		__m128 ms8 = _mm_load_ps(src + 128);
		__m128 ms9 = _mm_load_ps(src + 144);
		__m128 ms10 = _mm_load_ps(src + 160);
		__m128 ms11 = _mm_load_ps(src + 176);
		__m128 ms12 = _mm_load_ps(src + 192);
		__m128 ms13 = _mm_load_ps(src + 208);
		__m128 ms14 = _mm_load_ps(src + 224);
		__m128 ms15 = _mm_load_ps(src + 240);

		__m128 mx00 = _mm_add_ps(ms0, ms15);
		__m128 mx01 = _mm_add_ps(ms1, ms14);
		__m128 mx02 = _mm_add_ps(ms2, ms13);
		__m128 mx03 = _mm_add_ps(ms3, ms12);
		__m128 mx04 = _mm_add_ps(ms4, ms11);
		__m128 mx05 = _mm_add_ps(ms5, ms10);
		__m128 mx06 = _mm_add_ps(ms6, ms9);
		__m128 mx07 = _mm_add_ps(ms7, ms8);
		__m128 mx08 = _mm_sub_ps(ms0, ms15);
		__m128 mx09 = _mm_sub_ps(ms1, ms14);
		__m128 mx0a = _mm_sub_ps(ms2, ms13);
		__m128 mx0b = _mm_sub_ps(ms3, ms12);
		__m128 mx0c = _mm_sub_ps(ms4, ms11);
		__m128 mx0d = _mm_sub_ps(ms5, ms10);
		__m128 mx0e = _mm_sub_ps(ms6, ms9);
		__m128 mx0f = _mm_sub_ps(ms7, ms8);

		__m128 mx10 = _mm_add_ps(mx00, mx07);
		__m128 mx11 = _mm_add_ps(mx01, mx06);
		__m128 mx12 = _mm_add_ps(mx02, mx05);
		__m128 mx13 = _mm_add_ps(mx03, mx04);
		__m128 mx14 = _mm_sub_ps(mx00, mx07);
		__m128 mx15 = _mm_sub_ps(mx01, mx06);
		__m128 mx16 = _mm_sub_ps(mx02, mx05);
		__m128 mx17 = _mm_sub_ps(mx03, mx04);
		__m128 mx18 = _mm_add_ps(mx10, mx13);
		__m128 mx19 = _mm_add_ps(mx11, mx12);
		__m128 mx1a = _mm_sub_ps(mx10, mx13);
		__m128 mx1b = _mm_sub_ps(mx11, mx12);

		__m128 mx1c = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.38703984532215f), mx14), _mm_mul_ps(_mm_set1_ps(0.275899379282943f), mx17));
		__m128 mx1d = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.17587560241936f), mx15), _mm_mul_ps(_mm_set1_ps(0.785694958387102f), mx16));
		__m128 mx1e = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-0.785694958387102f), mx15), _mm_mul_ps(_mm_set1_ps(1.17587560241936f), mx16));
		__m128 mx1f = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.275899379282943f), mx14), _mm_mul_ps(_mm_set1_ps(-1.38703984532215f), mx17));
		__m128 mx20 = _mm_mul_ps(c0250, _mm_sub_ps(mx1c, mx1d));
		__m128 mx21 = _mm_mul_ps(c0250, _mm_sub_ps(mx1e, mx1f));
		__m128 mx22 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.40740373752638f), mx08), _mm_mul_ps(_mm_set1_ps(0.138617169199091f), mx0f));
		__m128 mx23 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.35331800117435f), mx09), _mm_mul_ps(_mm_set1_ps(0.410524527522357f), mx0e));
		__m128 mx24 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.24722501298667f), mx0a), _mm_mul_ps(_mm_set1_ps(0.666655658477747f), mx0d));
		__m128 mx25 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.09320186700176f), mx0b), _mm_mul_ps(_mm_set1_ps(0.897167586342636f), mx0c));
		__m128 mx26 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-0.897167586342636f), mx0b), _mm_mul_ps(_mm_set1_ps(1.09320186700176f), mx0c));
		__m128 mx27 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.666655658477747f), mx0a), _mm_mul_ps(_mm_set1_ps(-1.24722501298667f), mx0d));
		__m128 mx28 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-0.410524527522357f), mx09), _mm_mul_ps(_mm_set1_ps(1.35331800117435f), mx0e));
		__m128 mx29 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.138617169199091f), mx08), _mm_mul_ps(_mm_set1_ps(-1.40740373752638f), mx0f));
		__m128 mx2a = _mm_add_ps(mx22, mx25);
		__m128 mx2b = _mm_add_ps(mx23, mx24);
		__m128 mx2c = _mm_sub_ps(mx22, mx25);
		__m128 mx2d = _mm_sub_ps(mx23, mx24);
		__m128 mx2e = _mm_mul_ps(c0250, _mm_sub_ps(mx2a, mx2b));
		__m128 mx2f = _mm_add_ps(_mm_mul_ps(c0326, mx2c), _mm_mul_ps(c0135, mx2d));
		__m128 mx30 = _mm_sub_ps(_mm_mul_ps(c0135, mx2c), _mm_mul_ps(c0326, mx2d));
		__m128 mx31 = _mm_add_ps(mx26, mx29);
		__m128 mx32 = _mm_add_ps(mx27, mx28);
		__m128 mx33 = _mm_sub_ps(mx26, mx29);
		__m128 mx34 = _mm_sub_ps(mx27, mx28);
		__m128 mx35 = _mm_mul_ps(c0250, _mm_sub_ps(mx31, mx32));
		__m128 mx36 = _mm_add_ps(_mm_mul_ps(c0326, mx33), _mm_mul_ps(c0135, mx34));
		__m128 mx37 = _mm_sub_ps(_mm_mul_ps(c0135, mx33), _mm_mul_ps(c0326, mx34));

		// keep 00 coef.
		__m128 v = _mm_mul_ps(c0250, _mm_add_ps(mx18, mx19));
		__m128 msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
		__m128 v2 = _mm_blendv_ps(zeros, v, msk);
		v2 = _mm_blend_ps(v2, v, 1);
		_mm_store_ps(dst + 0, v2);

		v = _mm_mul_ps(c0250, _mm_add_ps(mx2a, mx2b));
		msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
		v = _mm_blendv_ps(zeros, v, msk);
		_mm_store_ps(dst + 16, v);

		v = _mm_mul_ps(c0250, _mm_add_ps(mx1c, mx1d));
		msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
		v = _mm_blendv_ps(zeros, v, msk);
		_mm_store_ps(dst + 32, v);

		v = _mm_mul_ps(c0707, _mm_sub_ps(mx2f, mx37));
		msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
		v = _mm_blendv_ps(zeros, v, msk);
		_mm_store_ps(dst + 48, v);

		v = _mm_add_ps(_mm_mul_ps(c0326, mx1a), _mm_mul_ps(c0135, mx1b));
		msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
		v = _mm_blendv_ps(zeros, v, msk);
		_mm_store_ps(dst + 64, v);

		v = _mm_mul_ps(c0707, _mm_add_ps(mx2f, mx37));
		msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
		v = _mm_blendv_ps(zeros, v, msk);
		_mm_store_ps(dst + 80, v);

		v = _mm_mul_ps(c0707, _mm_sub_ps(mx20, mx21));
		msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
		v = _mm_blendv_ps(zeros, v, msk);
		_mm_store_ps(dst + 96, v);

		v = _mm_mul_ps(c0707, _mm_add_ps(mx2e, mx35));
		msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
		v = _mm_blendv_ps(zeros, v, msk);
		_mm_store_ps(dst + 112, v);

		v = _mm_mul_ps(c0250, _mm_sub_ps(mx18, mx19));
		msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
		v = _mm_blendv_ps(zeros, v, msk);
		_mm_store_ps(dst + 128, v);

		v = _mm_mul_ps(c0707, _mm_sub_ps(mx2e, mx35));
		msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
		v = _mm_blendv_ps(zeros, v, msk);
		_mm_store_ps(dst + 144, v);

		v = _mm_mul_ps(c0707, _mm_add_ps(mx20, mx21));
		msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
		v = _mm_blendv_ps(zeros, v, msk);
		_mm_store_ps(dst + 160, v);

		v = _mm_mul_ps(c0707, _mm_sub_ps(mx30, mx36));
		msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
		v = _mm_blendv_ps(zeros, v, msk);
		_mm_store_ps(dst + 176, v);

		v = _mm_sub_ps(_mm_mul_ps(c0135, mx1a), _mm_mul_ps(c0326, mx1b));
		msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
		v = _mm_blendv_ps(zeros, v, msk);
		_mm_store_ps(dst + 192, v);

		v = _mm_mul_ps(c0707, _mm_add_ps(mx30, mx36));
		msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
		v = _mm_blendv_ps(zeros, v, msk);
		_mm_store_ps(dst + 208, v);

		v = _mm_mul_ps(c0250, _mm_add_ps(mx1e, mx1f));
		msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
		v = _mm_blendv_ps(zeros, v, msk);
		_mm_store_ps(dst + 224, v);

		v = _mm_mul_ps(c0250, _mm_add_ps(mx31, mx32));
		msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
		v = _mm_blendv_ps(zeros, v, msk);
		_mm_store_ps(dst + 240, v);

		dst += 4;
		src += 4;
	}
}

static void idct_161d_base(const float *src, float *dst)
{
	for (int i = 0; i < 16; i++)
	{
		const float mx00 = 1.4142135623731f  *src[0];
		const float mx01 = 1.40740373752638f *src[1] + 0.138617169199091f*src[15];
		const float mx02 = 1.38703984532215f *src[2] + 0.275899379282943f*src[14];
		const float mx03 = 1.35331800117435f *src[3] + 0.410524527522357f*src[13];
		const float mx04 = 1.30656296487638f *src[4] + 0.541196100146197f*src[12];
		const float mx05 = 1.24722501298667f *src[5] + 0.666655658477747f*src[11];
		const float mx06 = 1.17587560241936f *src[6] + 0.785694958387102f*src[10];
		const float mx07 = 1.09320186700176f *src[7] + 0.897167586342636f*src[9];
		const float mx08 = 1.4142135623731f  *src[8];
		const float mx09 = -0.897167586342636f*src[7] + 1.09320186700176f*src[9];
		const float mx0a = 0.785694958387102f*src[6] - 1.17587560241936f*src[10];
		const float mx0b = -0.666655658477747f*src[5] + 1.24722501298667f*src[11];
		const float mx0c = 0.541196100146197f*src[4] - 1.30656296487638f*src[12];
		const float mx0d = -0.410524527522357f*src[3] + 1.35331800117435f*src[13];
		const float mx0e = 0.275899379282943f*src[2] - 1.38703984532215f*src[14];
		const float mx0f = -0.138617169199091f*src[1] + 1.40740373752638f*src[15];
		const float mx12 = mx00 + mx08;
		const float mx13 = mx01 + mx07;
		const float mx14 = mx02 + mx06;
		const float mx15 = mx03 + mx05;
		const float mx16 = 1.4142135623731f*mx04;
		const float mx17 = mx00 - mx08;
		const float mx18 = mx01 - mx07;
		const float mx19 = mx02 - mx06;
		const float mx1a = mx03 - mx05;
		const float mx1d = mx12 + mx16;
		const float mx1e = mx13 + mx15;
		const float mx1f = 1.4142135623731f*mx14;
		const float mx20 = mx12 - mx16;
		const float mx21 = mx13 - mx15;
		const float mx22 = 0.25f * (mx1d - mx1f);
		const float mx23 = 0.25f * (mx20 + mx21);
		const float mx24 = 0.25f * (mx20 - mx21);
		const float mx25 = 1.4142135623731f*mx17;
		const float mx26 = 1.30656296487638f*mx18 + 0.541196100146197f*mx1a;
		const float mx27 = 1.4142135623731f*mx19;
		const float mx28 = -0.541196100146197f*mx18 + 1.30656296487638f*mx1a;
		const float mx29 = 0.176776695296637f * (mx25 + mx27) + 0.25f*mx26;
		const float mx2a = 0.25f * (mx25 - mx27);
		const float mx2b = 0.176776695296637f * (mx25 + mx27) - 0.25f*mx26;
		const float mx2c = 0.353553390593274f*mx28;
		const float mx1b = 0.707106781186547f * (mx2a - mx2c);
		const float mx1c = 0.707106781186547f * (mx2a + mx2c);
		const float mx2d = 1.4142135623731f*mx0c;
		const float mx2e = mx0b + mx0d;
		const float mx2f = mx0a + mx0e;
		const float mx30 = mx09 + mx0f;
		const float mx31 = mx09 - mx0f;
		const float mx32 = mx0a - mx0e;
		const float mx33 = mx0b - mx0d;
		const float mx37 = 1.4142135623731f*mx2d;
		const float mx38 = 1.30656296487638f*mx2e + 0.541196100146197f*mx30;
		const float mx39 = 1.4142135623731f*mx2f;
		const float mx3a = -0.541196100146197f*mx2e + 1.30656296487638f*mx30;
		const float mx3b = 0.176776695296637f * (mx37 + mx39) + 0.25f*mx38;
		const float mx3c = 0.25f * (mx37 - mx39);
		const float mx3d = 0.176776695296637f * (mx37 + mx39) - 0.25f*mx38;
		const float mx3e = 0.353553390593274f*mx3a;
		const float mx34 = 0.707106781186547f * (mx3c - mx3e);
		const float mx35 = 0.707106781186547f * (mx3c + mx3e);
		const float mx3f = 1.4142135623731f*mx32;
		const float mx40 = mx31 + mx33;
		const float mx41 = mx31 - mx33;
		const float mx42 = 0.25f * (mx3f + mx40);
		const float mx43 = 0.25f * (mx3f - mx40);
		const float mx44 = 0.353553390593274f*mx41;

		dst[0] = 0.176776695296637f * (mx1d + mx1f) + 0.25f*mx1e;
		dst[1] = 0.707106781186547f * (mx29 + mx3d);
		dst[2] = 0.707106781186547f * (mx29 - mx3d);
		dst[3] = 0.707106781186547f * (mx23 - mx43);
		dst[4] = 0.707106781186547f * (mx23 + mx43);
		dst[5] = 0.707106781186547f * (mx1b - mx35);
		dst[6] = 0.707106781186547f * (mx1b + mx35);
		dst[7] = 0.707106781186547f * (mx22 + mx44);
		dst[8] = 0.707106781186547f * (mx22 - mx44);
		dst[9] = 0.707106781186547f * (mx1c + mx34);
		dst[10] = 0.707106781186547f * (mx1c - mx34);
		dst[11] = 0.707106781186547f * (mx24 + mx42);
		dst[12] = 0.707106781186547f * (mx24 - mx42);
		dst[13] = 0.707106781186547f * (mx2b - mx3b);
		dst[14] = 0.707106781186547f * (mx2b + mx3b);
		dst[15] = 0.176776695296637f * (mx1d + mx1f) - 0.25f*mx1e;
		dst += 16;
		src += 16;
	}
}

static void idct161d_sse(const float *s, float *d)
{
	float* src = (float*)s;
	float* dst = d;

	const __m128 c0176 = _mm_set1_ps(0.176776695296637f);
	const __m128 c0250 = _mm_set1_ps(0.25f);
	const __m128 c0353 = _mm_set1_ps(0.353553390593274f);
	const __m128 c0707 = _mm_set1_ps(0.707106781186547f);
	const __m128 c1306 = _mm_set1_ps(1.30656296487638f);
	const __m128 c1407 = _mm_set1_ps(1.40740373752638f);
	const __m128 c1414 = _mm_set1_ps(1.4142135623731f);
	const __m128 c0541 = _mm_set1_ps(0.541196100146197f);

	for (int i = 0; i < 4; i++)
	{
		__m128 ms0 = _mm_load_ps(src);
		__m128 ms1 = _mm_load_ps(src + 16);
		__m128 ms2 = _mm_load_ps(src + 32);
		__m128 ms3 = _mm_load_ps(src + 48);
		__m128 ms4 = _mm_load_ps(src + 64);
		__m128 ms5 = _mm_load_ps(src + 80);
		__m128 ms6 = _mm_load_ps(src + 96);
		__m128 ms7 = _mm_load_ps(src + 112);
		__m128 ms8 = _mm_load_ps(src + 128);
		__m128 ms9 = _mm_load_ps(src + 144);
		__m128 ms10 = _mm_load_ps(src + 160);
		__m128 ms11 = _mm_load_ps(src + 176);
		__m128 ms12 = _mm_load_ps(src + 192);
		__m128 ms13 = _mm_load_ps(src + 208);
		__m128 ms14 = _mm_load_ps(src + 224);
		__m128 ms15 = _mm_load_ps(src + 240);

		__m128 mx00 = _mm_mul_ps(c1414, ms0);
		__m128 mx01 = _mm_add_ps(_mm_mul_ps(c1407, ms1), _mm_mul_ps(_mm_set1_ps(0.138617169199091f), ms15));
		__m128 mx02 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.38703984532215f), ms2), _mm_mul_ps(_mm_set1_ps(0.275899379282943f), ms14));
		__m128 mx03 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.35331800117435f), ms3), _mm_mul_ps(_mm_set1_ps(0.410524527522357f), ms13));
		__m128 mx04 = _mm_add_ps(_mm_mul_ps(c1306, ms4), _mm_mul_ps(c0541, ms12));
		__m128 mx05 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.24722501298667f), ms5), _mm_mul_ps(_mm_set1_ps(0.666655658477747f), ms11));
		__m128 mx06 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.17587560241936f), ms6), _mm_mul_ps(_mm_set1_ps(0.785694958387102f), ms10));
		__m128 mx07 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.09320186700176f), ms7), _mm_mul_ps(_mm_set1_ps(0.897167586342636f), ms9));
		__m128 mx08 = _mm_mul_ps(c1414, ms8);
		__m128 mx09 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-0.897167586342636f), ms7), _mm_mul_ps(_mm_set1_ps(1.09320186700176f), ms9));
		__m128 mx0a = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.785694958387102f), ms6), _mm_mul_ps(_mm_set1_ps(-1.17587560241936f), ms10));
		__m128 mx0b = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-0.666655658477747f), ms5), _mm_mul_ps(_mm_set1_ps(1.24722501298667f), ms11));
		__m128 mx0c = _mm_add_ps(_mm_mul_ps(c0541, ms4), _mm_mul_ps(_mm_set1_ps(-1.30656296487638f), ms12));
		__m128 mx0d = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-0.410524527522357f), ms3), _mm_mul_ps(_mm_set1_ps(1.35331800117435f), ms13));
		__m128 mx0e = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.275899379282943f), ms2), _mm_mul_ps(_mm_set1_ps(-1.38703984532215f), ms14));
		__m128 mx0f = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-0.138617169199091f), ms1), _mm_mul_ps(c1407, ms15));

		__m128 mx12 = _mm_add_ps(mx00, mx08);
		__m128 mx13 = _mm_add_ps(mx01, mx07);
		__m128 mx14 = _mm_add_ps(mx02, mx06);
		__m128 mx15 = _mm_add_ps(mx03, mx05);
		__m128 mx16 = _mm_mul_ps(c1414, mx04);
		__m128 mx17 = _mm_sub_ps(mx00, mx08);
		__m128 mx18 = _mm_sub_ps(mx01, mx07);
		__m128 mx19 = _mm_sub_ps(mx02, mx06);
		__m128 mx1a = _mm_sub_ps(mx03, mx05);
		__m128 mx1d = _mm_add_ps(mx12, mx16);
		__m128 mx1e = _mm_add_ps(mx13, mx15);
		__m128 mx1f = _mm_mul_ps(c1414, mx14);
		__m128 mx20 = _mm_sub_ps(mx12, mx16);
		__m128 mx21 = _mm_sub_ps(mx13, mx15);
		__m128 mx22 = _mm_mul_ps(c0250, _mm_sub_ps(mx1d, mx1f));
		__m128 mx23 = _mm_mul_ps(c0250, _mm_add_ps(mx20, mx21));
		__m128 mx24 = _mm_mul_ps(c0250, _mm_sub_ps(mx20, mx21));
		__m128 mx25 = _mm_mul_ps(c1414, mx17);
		__m128 mx26 = _mm_add_ps(_mm_mul_ps(c1306, mx18), _mm_mul_ps(c0541, mx1a));
		__m128 mx27 = _mm_mul_ps(c1414, mx19);
		__m128 mx28 = _mm_sub_ps(_mm_mul_ps(c1306, mx1a), _mm_mul_ps(c0541, mx18));//inv
		__m128 mx29 = _mm_add_ps(_mm_mul_ps(c0176, _mm_add_ps(mx25, mx27)), _mm_mul_ps(c0250, mx26));
		__m128 mx2a = _mm_mul_ps(c0250, _mm_sub_ps(mx25, mx27));
		__m128 mx2b = _mm_sub_ps(_mm_mul_ps(c0176, _mm_add_ps(mx25, mx27)), _mm_mul_ps(c0250, mx26));
		__m128 mx2c = _mm_mul_ps(c0353, mx28);
		__m128 mx1b = _mm_mul_ps(c0707, _mm_sub_ps(mx2a, mx2c));
		__m128 mx1c = _mm_mul_ps(c0707, _mm_add_ps(mx2a, mx2c));
		__m128 mx2d = _mm_mul_ps(c1414, mx0c);
		__m128 mx2e = _mm_add_ps(mx0b, mx0d);
		__m128 mx2f = _mm_add_ps(mx0a, mx0e);
		__m128 mx30 = _mm_add_ps(mx09, mx0f);
		__m128 mx31 = _mm_sub_ps(mx09, mx0f);
		__m128 mx32 = _mm_sub_ps(mx0a, mx0e);
		__m128 mx33 = _mm_sub_ps(mx0b, mx0d);
		__m128 mx37 = _mm_mul_ps(c1414, mx2d);
		__m128 mx38 = _mm_add_ps(_mm_mul_ps(c1306, mx2e), _mm_mul_ps(c0541, mx30));
		__m128 mx39 = _mm_mul_ps(c1414, mx2f);
		__m128 mx3a = _mm_sub_ps(_mm_mul_ps(c1306, mx30), _mm_mul_ps(c0541, mx2e));//inv
		__m128 mx3b = _mm_add_ps(_mm_mul_ps(c0176, _mm_add_ps(mx37, mx39)), _mm_mul_ps(c0250, mx38));
		__m128 mx3c = _mm_mul_ps(c0250, _mm_sub_ps(mx37, mx39));
		__m128 mx3d = _mm_sub_ps(_mm_mul_ps(c0176, _mm_add_ps(mx37, mx39)), _mm_mul_ps(c0250, mx38));
		__m128 mx3e = _mm_mul_ps(c0353, mx3a);
		__m128 mx34 = _mm_mul_ps(c0707, _mm_sub_ps(mx3c, mx3e));
		__m128 mx35 = _mm_mul_ps(c0707, _mm_add_ps(mx3c, mx3e));
		__m128 mx3f = _mm_mul_ps(c1414, mx32);
		__m128 mx40 = _mm_add_ps(mx31, mx33);
		__m128 mx41 = _mm_sub_ps(mx31, mx33);
		__m128 mx42 = _mm_mul_ps(c0250, _mm_add_ps(mx3f, mx40));
		__m128 mx43 = _mm_mul_ps(c0250, _mm_sub_ps(mx3f, mx40));
		__m128 mx44 = _mm_mul_ps(c0353, mx41);

		_mm_store_ps(dst + 0, _mm_add_ps(_mm_mul_ps(c0176, _mm_add_ps(mx1d, mx1f)), _mm_mul_ps(c0250, mx1e)));
		_mm_store_ps(dst + 16, _mm_mul_ps(c0707, _mm_add_ps(mx29, mx3d)));
		_mm_store_ps(dst + 32, _mm_mul_ps(c0707, _mm_sub_ps(mx29, mx3d)));
		_mm_store_ps(dst + 48, _mm_mul_ps(c0707, _mm_sub_ps(mx23, mx43)));
		_mm_store_ps(dst + 64, _mm_mul_ps(c0707, _mm_add_ps(mx23, mx43)));
		_mm_store_ps(dst + 80, _mm_mul_ps(c0707, _mm_sub_ps(mx1b, mx35)));
		_mm_store_ps(dst + 96, _mm_mul_ps(c0707, _mm_add_ps(mx1b, mx35)));
		_mm_store_ps(dst + 112, _mm_mul_ps(c0707, _mm_add_ps(mx22, mx44)));
		_mm_store_ps(dst + 128, _mm_mul_ps(c0707, _mm_sub_ps(mx22, mx44)));
		_mm_store_ps(dst + 144, _mm_mul_ps(c0707, _mm_add_ps(mx1c, mx34)));
		_mm_store_ps(dst + 160, _mm_mul_ps(c0707, _mm_sub_ps(mx1c, mx34)));
		_mm_store_ps(dst + 176, _mm_mul_ps(c0707, _mm_add_ps(mx24, mx42)));
		_mm_store_ps(dst + 192, _mm_mul_ps(c0707, _mm_sub_ps(mx24, mx42)));
		_mm_store_ps(dst + 208, _mm_mul_ps(c0707, _mm_sub_ps(mx2b, mx3b)));
		_mm_store_ps(dst + 224, _mm_mul_ps(c0707, _mm_add_ps(mx2b, mx3b)));
		_mm_store_ps(dst + 240, _mm_sub_ps(_mm_mul_ps(c0176, _mm_add_ps(mx1d, mx1f)), _mm_mul_ps(c0250, mx1e)));

		dst += 4;
		src += 4;
	}
}

void fDCT16x16_threshold_keep00_iDCT16x16(const float* src, float* dest, float th)
{
	fdct161d_sse(src, dest);
	//fdct16_1d(src, dest);
	transpose16x16(dest);

	fdct161d_threshold_keep00_sse(src, dest, th);
	//fdct16_1d_sse(dest, dest);
	//fdct16_1d(dest, dest);

	idct161d_sse(src, dest);
	//idct16_1d(src, dest);
	transpose16x16(dest);
	idct161d_sse(dest, dest);
}

void fDCT16x16(const float* src, float* dest)
{
	fdct161d_sse(src, dest);
	//fdct161d_base(src, dest);
	transpose16x16(dest);

	fdct161d_sse(dest, dest);
	//fdct161d_base(dest, dest);
	transpose16x16(dest);
}

void iDCT16x16(const float* src, float* dest)
{
	idct161d_sse(src, dest);
	//idct16_1d(src, dest);
	transpose16x16(dest);

	idct161d_sse(dest, dest);
	//idct16_1d(dest, dest);
	transpose16x16(dest);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//8x8//////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

//Plonka, Gerlind, and Manfred Tasche. "Fast and numerically stable algorithms for discrete cosine transforms." Linear algebra and its applications 394 (2005) : 309 - 345.

static void fdct81d_GT(const float *src, float *dst)
{
	for (int i = 0; i < 2; i++)
	{

		const float mx00 = src[0] + src[7];
		const float mx01 = src[1] + src[6];
		const float mx02 = src[2] + src[5];
		const float mx03 = src[3] + src[4];
		const float mx04 = src[0] - src[7];
		const float mx05 = src[1] - src[6];
		const float mx06 = src[2] - src[5];
		const float mx07 = src[3] - src[4];
		const float mx08 = mx00 + mx03;
		const float mx09 = mx01 + mx02;
		const float mx0a = mx00 - mx03;
		const float mx0b = mx01 - mx02;
		const float mx0c = 1.38703984532215f*mx04 + 0.275899379282943f*mx07;
		const float mx0d = 1.17587560241936f*mx05 + 0.785694958387102f*mx06;
		const float mx0e = -0.785694958387102f*mx05 + 1.17587560241936f*mx06;
		const float mx0f = 0.275899379282943f*mx04 - 1.38703984532215f*mx07;
		const float mx10 = 0.353553390593274f * (mx0c - mx0d);
		const float mx11 = 0.353553390593274f * (mx0e - mx0f);
		dst[0] = 0.353553390593274f * (mx08 + mx09);
		dst[1] = 0.353553390593274f * (mx0c + mx0d);
		dst[2] = 0.461939766255643f*mx0a + 0.191341716182545f*mx0b;
		dst[3] = 0.707106781186547f * (mx10 - mx11);
		dst[4] = 0.353553390593274f * (mx08 - mx09);
		dst[5] = 0.707106781186547f * (mx10 + mx11);
		dst[6] = 0.191341716182545f*mx0a - 0.461939766255643f*mx0b;
		dst[7] = 0.353553390593274f * (mx0e + mx0f);
		dst += 4;
		src += 4;
	}
}

static void idct81d_GT(const float *src, float *dst)
{
	for (int i = 0; i < 8; i++)
	{
		const float mx00 = 1.4142135623731f  *src[0];
		const float mx01 = 1.38703984532215f *src[1] + 0.275899379282943f*src[7];
		const float mx02 = 1.30656296487638f *src[2] + 0.541196100146197f*src[6];
		const float mx03 = 1.17587560241936f *src[3] + 0.785694958387102f*src[5];
		const float mx04 = 1.4142135623731f  *src[4];
		const float mx05 = -0.785694958387102f*src[3] + 1.17587560241936f*src[5];
		const float mx06 = 0.541196100146197f*src[2] - 1.30656296487638f*src[6];
		const float mx07 = -0.275899379282943f*src[1] + 1.38703984532215f*src[7];
		const float mx09 = mx00 + mx04;
		const float mx0a = mx01 + mx03;
		const float mx0b = 1.4142135623731f*mx02;
		const float mx0c = mx00 - mx04;
		const float mx0d = mx01 - mx03;
		const float mx0e = 0.353553390593274f * (mx09 - mx0b);
		const float mx0f = 0.353553390593274f * (mx0c + mx0d);
		const float mx10 = 0.353553390593274f * (mx0c - mx0d);
		const float mx11 = 1.4142135623731f*mx06;
		const float mx12 = mx05 + mx07;
		const float mx13 = mx05 - mx07;
		const float mx14 = 0.353553390593274f * (mx11 + mx12);
		const float mx15 = 0.353553390593274f * (mx11 - mx12);
		const float mx16 = 0.5f*mx13;
		dst[0] = 0.25f * (mx09 + mx0b) + 0.353553390593274f*mx0a;
		dst[1] = 0.707106781186547f * (mx0f + mx15);
		dst[2] = 0.707106781186547f * (mx0f - mx15);
		dst[3] = 0.707106781186547f * (mx0e + mx16);
		dst[4] = 0.707106781186547f * (mx0e - mx16);
		dst[5] = 0.707106781186547f * (mx10 - mx14);
		dst[6] = 0.707106781186547f * (mx10 + mx14);
		dst[7] = 0.25f * (mx09 + mx0b) - 0.353553390593274f*mx0a;
		dst += 8;
		src += 8;
	}
}

static void fdct81d_sse_GT(const float *src, float *dst)
{
	const __m128 c0353 = _mm_set1_ps(0.353553390593274f);
	const __m128 c0707 = _mm_set1_ps(0.707106781186547f);
	for (int i = 0; i < 2; i++)
	{
		__m128 ms0 = _mm_load_ps(src);
		__m128 ms1 = _mm_load_ps(src + 8);
		__m128 ms2 = _mm_load_ps(src + 16);
		__m128 ms3 = _mm_load_ps(src + 24);
		__m128 ms4 = _mm_load_ps(src + 32);
		__m128 ms5 = _mm_load_ps(src + 40);
		__m128 ms6 = _mm_load_ps(src + 48);
		__m128 ms7 = _mm_load_ps(src + 56);

		__m128 mx00 = _mm_add_ps(ms0, ms7);
		__m128 mx01 = _mm_add_ps(ms1, ms6);
		__m128 mx02 = _mm_add_ps(ms2, ms5);
		__m128 mx03 = _mm_add_ps(ms3, ms4);
		__m128 mx04 = _mm_sub_ps(ms0, ms7);
		__m128 mx05 = _mm_sub_ps(ms1, ms6);
		__m128 mx06 = _mm_sub_ps(ms2, ms5);
		__m128 mx07 = _mm_sub_ps(ms3, ms4);
		__m128 mx08 = _mm_add_ps(mx00, mx03);
		__m128 mx09 = _mm_add_ps(mx01, mx02);
		__m128 mx0a = _mm_sub_ps(mx00, mx03);
		__m128 mx0b = _mm_sub_ps(mx01, mx02);

		__m128 mx0c = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.38703984532215f), mx04), _mm_mul_ps(_mm_set1_ps(0.275899379282943f), mx07));
		__m128 mx0d = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.17587560241936f), mx05), _mm_mul_ps(_mm_set1_ps(+0.785694958387102f), mx06));
		__m128 mx0e = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-0.785694958387102f), mx05), _mm_mul_ps(_mm_set1_ps(+1.17587560241936f), mx06));
		__m128 mx0f = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.275899379282943f), mx04), _mm_mul_ps(_mm_set1_ps(-1.38703984532215f), mx07));
		__m128 mx10 = _mm_mul_ps(c0353, _mm_sub_ps(mx0c, mx0d));
		__m128 mx11 = _mm_mul_ps(c0353, _mm_sub_ps(mx0e, mx0f));

		_mm_store_ps(dst + 0, _mm_mul_ps(c0353, _mm_add_ps(mx08, mx09)));
		_mm_store_ps(dst + 8, _mm_mul_ps(c0353, _mm_add_ps(mx0c, mx0d)));
		_mm_store_ps(dst + 16, _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.461939766255643f), mx0a), _mm_mul_ps(_mm_set1_ps(0.191341716182545f), mx0b)));
		_mm_store_ps(dst + 24, _mm_mul_ps(c0707, _mm_sub_ps(mx10, mx11)));
		_mm_store_ps(dst + 32, _mm_mul_ps(c0353, _mm_sub_ps(mx08, mx09)));
		_mm_store_ps(dst + 40, _mm_mul_ps(c0707, _mm_add_ps(mx10, mx11)));
		_mm_store_ps(dst + 48, _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.191341716182545f), mx0a), _mm_mul_ps(_mm_set1_ps(-0.461939766255643f), mx0b)));
		_mm_store_ps(dst + 56, _mm_mul_ps(c0353, _mm_add_ps(mx0e, mx0f)));
		dst += 4;
		src += 4;
	}
}

static void fdct88_sse_GT(const float *src, float *dst)
{
	const __m128 c0353 = _mm_set1_ps(0.353553390593274f);
	const __m128 c0707 = _mm_set1_ps(0.707106781186547f);
	
		__m128 ms0 = _mm_load_ps(src);
		__m128 ms1 = _mm_load_ps(src + 8);
		__m128 ms2 = _mm_load_ps(src + 16);
		__m128 ms3 = _mm_load_ps(src + 24);
		__m128 ms4 = _mm_load_ps(src + 32);
		__m128 ms5 = _mm_load_ps(src + 40);
		__m128 ms6 = _mm_load_ps(src + 48);
		__m128 ms7 = _mm_load_ps(src + 56);

		__m128 mx00 = _mm_add_ps(ms0, ms7);
		__m128 mx01 = _mm_add_ps(ms1, ms6);
		__m128 mx02 = _mm_add_ps(ms2, ms5);
		__m128 mx03 = _mm_add_ps(ms3, ms4);
		__m128 mx04 = _mm_sub_ps(ms0, ms7);
		__m128 mx05 = _mm_sub_ps(ms1, ms6);
		__m128 mx06 = _mm_sub_ps(ms2, ms5);
		__m128 mx07 = _mm_sub_ps(ms3, ms4);
		__m128 mx08 = _mm_add_ps(mx00, mx03);
		__m128 mx09 = _mm_add_ps(mx01, mx02);
		__m128 mx0a = _mm_sub_ps(mx00, mx03);
		__m128 mx0b = _mm_sub_ps(mx01, mx02);

		__m128 mx0c = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.38703984532215f), mx04), _mm_mul_ps(_mm_set1_ps(0.275899379282943f), mx07));
		__m128 mx0d = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.17587560241936f), mx05), _mm_mul_ps(_mm_set1_ps(+0.785694958387102f), mx06));
		__m128 mx0e = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-0.785694958387102f), mx05), _mm_mul_ps(_mm_set1_ps(+1.17587560241936f), mx06));
		__m128 mx0f = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.275899379282943f), mx04), _mm_mul_ps(_mm_set1_ps(-1.38703984532215f), mx07));
		__m128 mx10 = _mm_mul_ps(c0353, _mm_sub_ps(mx0c, mx0d));
		__m128 mx11 = _mm_mul_ps(c0353, _mm_sub_ps(mx0e, mx0f));

		__m128 md00 = _mm_mul_ps(c0353, _mm_add_ps(mx08, mx09));
		__m128 md01 = _mm_mul_ps(c0353, _mm_add_ps(mx0c, mx0d));
		__m128 md02 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.461939766255643f), mx0a), _mm_mul_ps(_mm_set1_ps(0.191341716182545f), mx0b));
		__m128 md03 = _mm_mul_ps(c0707, _mm_sub_ps(mx10, mx11));
		
		_MM_TRANSPOSE4_PS(md00, md01, md02, md03);

		__m128 md10 = _mm_mul_ps(c0353, _mm_sub_ps(mx08, mx09));
		__m128 md11 = _mm_mul_ps(c0707, _mm_add_ps(mx10, mx11));
		__m128 md12 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.191341716182545f), mx0a), _mm_mul_ps(_mm_set1_ps(-0.461939766255643f), mx0b));
		__m128 md13 = _mm_mul_ps(c0353, _mm_add_ps(mx0e, mx0f));
		_MM_TRANSPOSE4_PS(md10, md11, md12, md13);
		
		src += 4;
		ms0 = _mm_load_ps(src);
		ms1 = _mm_load_ps(src + 8);
		ms2 = _mm_load_ps(src + 16);
		ms3 = _mm_load_ps(src + 24);
		ms4 = _mm_load_ps(src + 32);
		ms5 = _mm_load_ps(src + 40);
		ms6 = _mm_load_ps(src + 48);
		ms7 = _mm_load_ps(src + 56);

		mx00 = _mm_add_ps(ms0, ms7);
		mx01 = _mm_add_ps(ms1, ms6);
		mx02 = _mm_add_ps(ms2, ms5);
		mx03 = _mm_add_ps(ms3, ms4);
		mx04 = _mm_sub_ps(ms0, ms7);
		mx05 = _mm_sub_ps(ms1, ms6);
		mx06 = _mm_sub_ps(ms2, ms5);
		mx07 = _mm_sub_ps(ms3, ms4);
		mx08 = _mm_add_ps(mx00, mx03);
		mx09 = _mm_add_ps(mx01, mx02);
		mx0a = _mm_sub_ps(mx00, mx03);
		mx0b = _mm_sub_ps(mx01, mx02);

		mx0c = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.38703984532215f), mx04), _mm_mul_ps(_mm_set1_ps(0.275899379282943f), mx07));
		mx0d = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.17587560241936f), mx05), _mm_mul_ps(_mm_set1_ps(+0.785694958387102f), mx06));
		mx0e = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-0.785694958387102f), mx05), _mm_mul_ps(_mm_set1_ps(+1.17587560241936f), mx06));
		mx0f = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.275899379282943f), mx04), _mm_mul_ps(_mm_set1_ps(-1.38703984532215f), mx07));
		mx10 = _mm_mul_ps(c0353, _mm_sub_ps(mx0c, mx0d));
		mx11 = _mm_mul_ps(c0353, _mm_sub_ps(mx0e, mx0f));

		__m128 md04 = _mm_mul_ps(c0353, _mm_add_ps(mx08, mx09));
		__m128 md05 = _mm_mul_ps(c0353, _mm_add_ps(mx0c, mx0d));
		__m128 md06 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.461939766255643f), mx0a), _mm_mul_ps(_mm_set1_ps(0.191341716182545f), mx0b));
		__m128 md07 = _mm_mul_ps(c0707, _mm_sub_ps(mx10, mx11));
		_MM_TRANSPOSE4_PS(md04, md05, md06, md07);
		
		__m128 md14 = _mm_mul_ps(c0353, _mm_sub_ps(mx08, mx09));
		__m128 md15 = _mm_mul_ps(c0707, _mm_add_ps(mx10, mx11));
		__m128 md16 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.191341716182545f), mx0a), _mm_mul_ps(_mm_set1_ps(-0.461939766255643f), mx0b));
		__m128 md17 = _mm_mul_ps(c0353, _mm_add_ps(mx0e, mx0f));
		_MM_TRANSPOSE4_PS(md14, md15, md16, md17);

		src -= 4;
		ms0 = md00;
		ms1 = md01;
		ms2 = md02;
		ms3 = md03;
		ms4 = md04;
		ms5 = md05;
		ms6 = md06;
		ms7 = md07;

		mx00 = _mm_add_ps(ms0, ms7);
		mx01 = _mm_add_ps(ms1, ms6);
		mx02 = _mm_add_ps(ms2, ms5);
		mx03 = _mm_add_ps(ms3, ms4);
		mx04 = _mm_sub_ps(ms0, ms7);
		mx05 = _mm_sub_ps(ms1, ms6);
		mx06 = _mm_sub_ps(ms2, ms5);
		mx07 = _mm_sub_ps(ms3, ms4);
		mx08 = _mm_add_ps(mx00, mx03);
		mx09 = _mm_add_ps(mx01, mx02);
		mx0a = _mm_sub_ps(mx00, mx03);
		mx0b = _mm_sub_ps(mx01, mx02);

		mx0c = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.38703984532215f), mx04), _mm_mul_ps(_mm_set1_ps(0.275899379282943f), mx07));
		mx0d = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.17587560241936f), mx05), _mm_mul_ps(_mm_set1_ps(+0.785694958387102f), mx06));
		mx0e = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-0.785694958387102f), mx05), _mm_mul_ps(_mm_set1_ps(+1.17587560241936f), mx06));
		mx0f = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.275899379282943f), mx04), _mm_mul_ps(_mm_set1_ps(-1.38703984532215f), mx07));
		mx10 = _mm_mul_ps(c0353, _mm_sub_ps(mx0c, mx0d));
		mx11 = _mm_mul_ps(c0353, _mm_sub_ps(mx0e, mx0f));

		__m128 a = _mm_mul_ps(c0353, _mm_add_ps(mx08, mx09));
		__m128 b = _mm_mul_ps(c0353, _mm_add_ps(mx0c, mx0d));
		__m128 c = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.461939766255643f), mx0a), _mm_mul_ps(_mm_set1_ps(0.191341716182545f), mx0b));
		__m128 d = _mm_mul_ps(c0707, _mm_sub_ps(mx10, mx11));

		_mm_store_ps(dst + 0, a);
		_mm_store_ps(dst + 8, b);
		_mm_store_ps(dst + 16, c);
		_mm_store_ps(dst + 24, d);

		a = _mm_mul_ps(c0353, _mm_sub_ps(mx08, mx09));
		b = _mm_mul_ps(c0707, _mm_add_ps(mx10, mx11));
		c = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.191341716182545f), mx0a), _mm_mul_ps(_mm_set1_ps(-0.461939766255643f), mx0b));
		d = _mm_mul_ps(c0353, _mm_add_ps(mx0e, mx0f));
		_MM_TRANSPOSE4_PS(a, b, c, d);
		dst += 4;
		_mm_store_ps(dst + 0, a);
		_mm_store_ps(dst + 8, b);
		_mm_store_ps(dst + 16, c);
		_mm_store_ps(dst + 24, d);
		
		ms0 = md10;
		ms1 = md11;
		ms2 = md12;
		ms3 = md13;
		ms4 = md14;
		ms5 = md15;
		ms6 = md16;
		ms7 = md17;

		mx00 = _mm_add_ps(ms0, ms7);
		mx01 = _mm_add_ps(ms1, ms6);
		mx02 = _mm_add_ps(ms2, ms5);
		mx03 = _mm_add_ps(ms3, ms4);
		mx04 = _mm_sub_ps(ms0, ms7);
		mx05 = _mm_sub_ps(ms1, ms6);
		mx06 = _mm_sub_ps(ms2, ms5);
		mx07 = _mm_sub_ps(ms3, ms4);
		mx08 = _mm_add_ps(mx00, mx03);
		mx09 = _mm_add_ps(mx01, mx02);
		mx0a = _mm_sub_ps(mx00, mx03);
		mx0b = _mm_sub_ps(mx01, mx02);

		mx0c = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.38703984532215f), mx04), _mm_mul_ps(_mm_set1_ps(0.275899379282943f), mx07));
		mx0d = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.17587560241936f), mx05), _mm_mul_ps(_mm_set1_ps(+0.785694958387102f), mx06));
		mx0e = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-0.785694958387102f), mx05), _mm_mul_ps(_mm_set1_ps(+1.17587560241936f), mx06));
		mx0f = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.275899379282943f), mx04), _mm_mul_ps(_mm_set1_ps(-1.38703984532215f), mx07));
		mx10 = _mm_mul_ps(c0353, _mm_sub_ps(mx0c, mx0d));
		mx11 = _mm_mul_ps(c0353, _mm_sub_ps(mx0e, mx0f));

		a = _mm_mul_ps(c0353, _mm_add_ps(mx08, mx09));
		b = _mm_mul_ps(c0353, _mm_add_ps(mx0c, mx0d));
		c = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.461939766255643f), mx0a), _mm_mul_ps(_mm_set1_ps(0.191341716182545f), mx0b));
		d = _mm_mul_ps(c0707, _mm_sub_ps(mx10, mx11));
		_MM_TRANSPOSE4_PS(a, b, c, d);
		_mm_store_ps(dst + 28, a);
		_mm_store_ps(dst + 36, b);
		_mm_store_ps(dst + 44, c);
		_mm_store_ps(dst + 52, d);

		a = _mm_mul_ps(c0353, _mm_sub_ps(mx08, mx09));
		b = _mm_mul_ps(c0707, _mm_add_ps(mx10, mx11));
		c = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.191341716182545f), mx0a), _mm_mul_ps(_mm_set1_ps(-0.461939766255643f), mx0b));
		d = _mm_mul_ps(c0353, _mm_add_ps(mx0e, mx0f));
		_MM_TRANSPOSE4_PS(a, b, c, d);

		_mm_store_ps(dst + 32, a);
		_mm_store_ps(dst + 40, b);
		_mm_store_ps(dst + 48, c);
		_mm_store_ps(dst + 56, d);
}

static void fDCT8x8GT_threshold_keep00(const float *src, float *dst, float threshold)
{
	const int __declspec(align(16)) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
	const __m128 mth = _mm_set1_ps(threshold);
	const __m128 zeros = _mm_setzero_ps();
	const __m128 c0353 = _mm_set1_ps(0.353553390593274f);
	const __m128 c0707 = _mm_set1_ps(0.707106781186547f);

	__m128 ms0 = _mm_load_ps(src);
	__m128 ms1 = _mm_load_ps(src + 8);
	__m128 ms2 = _mm_load_ps(src + 16);
	__m128 ms3 = _mm_load_ps(src + 24);
	__m128 ms4 = _mm_load_ps(src + 32);
	__m128 ms5 = _mm_load_ps(src + 40);
	__m128 ms6 = _mm_load_ps(src + 48);
	__m128 ms7 = _mm_load_ps(src + 56);

	__m128 mx00 = _mm_add_ps(ms0, ms7);
	__m128 mx01 = _mm_add_ps(ms1, ms6);
	__m128 mx02 = _mm_add_ps(ms2, ms5);
	__m128 mx03 = _mm_add_ps(ms3, ms4);
	__m128 mx04 = _mm_sub_ps(ms0, ms7);
	__m128 mx05 = _mm_sub_ps(ms1, ms6);
	__m128 mx06 = _mm_sub_ps(ms2, ms5);
	__m128 mx07 = _mm_sub_ps(ms3, ms4);
	__m128 mx08 = _mm_add_ps(mx00, mx03);
	__m128 mx09 = _mm_add_ps(mx01, mx02);
	__m128 mx0a = _mm_sub_ps(mx00, mx03);
	__m128 mx0b = _mm_sub_ps(mx01, mx02);

	__m128 mx0c = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.38703984532215f), mx04), _mm_mul_ps(_mm_set1_ps(0.275899379282943f), mx07));
	__m128 mx0d = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.17587560241936f), mx05), _mm_mul_ps(_mm_set1_ps(+0.785694958387102f), mx06));
	__m128 mx0e = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-0.785694958387102f), mx05), _mm_mul_ps(_mm_set1_ps(+1.17587560241936f), mx06));
	__m128 mx0f = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.275899379282943f), mx04), _mm_mul_ps(_mm_set1_ps(-1.38703984532215f), mx07));
	__m128 mx10 = _mm_mul_ps(c0353, _mm_sub_ps(mx0c, mx0d));
	__m128 mx11 = _mm_mul_ps(c0353, _mm_sub_ps(mx0e, mx0f));

	__m128 md00 = _mm_mul_ps(c0353, _mm_add_ps(mx08, mx09));
	__m128 md01 = _mm_mul_ps(c0353, _mm_add_ps(mx0c, mx0d));
	__m128 md02 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.461939766255643f), mx0a), _mm_mul_ps(_mm_set1_ps(0.191341716182545f), mx0b));
	__m128 md03 = _mm_mul_ps(c0707, _mm_sub_ps(mx10, mx11));

	_MM_TRANSPOSE4_PS(md00, md01, md02, md03);

	__m128 md10 = _mm_mul_ps(c0353, _mm_sub_ps(mx08, mx09));
	__m128 md11 = _mm_mul_ps(c0707, _mm_add_ps(mx10, mx11));
	__m128 md12 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.191341716182545f), mx0a), _mm_mul_ps(_mm_set1_ps(-0.461939766255643f), mx0b));
	__m128 md13 = _mm_mul_ps(c0353, _mm_add_ps(mx0e, mx0f));
	_MM_TRANSPOSE4_PS(md10, md11, md12, md13);

	src += 4;
	ms0 = _mm_load_ps(src);
	ms1 = _mm_load_ps(src + 8);
	ms2 = _mm_load_ps(src + 16);
	ms3 = _mm_load_ps(src + 24);
	ms4 = _mm_load_ps(src + 32);
	ms5 = _mm_load_ps(src + 40);
	ms6 = _mm_load_ps(src + 48);
	ms7 = _mm_load_ps(src + 56);

	mx00 = _mm_add_ps(ms0, ms7);
	mx01 = _mm_add_ps(ms1, ms6);
	mx02 = _mm_add_ps(ms2, ms5);
	mx03 = _mm_add_ps(ms3, ms4);
	mx04 = _mm_sub_ps(ms0, ms7);
	mx05 = _mm_sub_ps(ms1, ms6);
	mx06 = _mm_sub_ps(ms2, ms5);
	mx07 = _mm_sub_ps(ms3, ms4);
	mx08 = _mm_add_ps(mx00, mx03);
	mx09 = _mm_add_ps(mx01, mx02);
	mx0a = _mm_sub_ps(mx00, mx03);
	mx0b = _mm_sub_ps(mx01, mx02);

	mx0c = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.38703984532215f), mx04), _mm_mul_ps(_mm_set1_ps(0.275899379282943f), mx07));
	mx0d = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.17587560241936f), mx05), _mm_mul_ps(_mm_set1_ps(+0.785694958387102f), mx06));
	mx0e = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-0.785694958387102f), mx05), _mm_mul_ps(_mm_set1_ps(+1.17587560241936f), mx06));
	mx0f = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.275899379282943f), mx04), _mm_mul_ps(_mm_set1_ps(-1.38703984532215f), mx07));
	mx10 = _mm_mul_ps(c0353, _mm_sub_ps(mx0c, mx0d));
	mx11 = _mm_mul_ps(c0353, _mm_sub_ps(mx0e, mx0f));

	__m128 md04 = _mm_mul_ps(c0353, _mm_add_ps(mx08, mx09));
	__m128 md05 = _mm_mul_ps(c0353, _mm_add_ps(mx0c, mx0d));
	__m128 md06 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.461939766255643f), mx0a), _mm_mul_ps(_mm_set1_ps(0.191341716182545f), mx0b));
	__m128 md07 = _mm_mul_ps(c0707, _mm_sub_ps(mx10, mx11));
	_MM_TRANSPOSE4_PS(md04, md05, md06, md07);

	__m128 md14 = _mm_mul_ps(c0353, _mm_sub_ps(mx08, mx09));
	__m128 md15 = _mm_mul_ps(c0707, _mm_add_ps(mx10, mx11));
	__m128 md16 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.191341716182545f), mx0a), _mm_mul_ps(_mm_set1_ps(-0.461939766255643f), mx0b));
	__m128 md17 = _mm_mul_ps(c0353, _mm_add_ps(mx0e, mx0f));
	_MM_TRANSPOSE4_PS(md14, md15, md16, md17);

	src -= 4;
	ms0 = md00;
	ms1 = md01;
	ms2 = md02;
	ms3 = md03;
	ms4 = md04;
	ms5 = md05;
	ms6 = md06;
	ms7 = md07;

	mx00 = _mm_add_ps(ms0, ms7);
	mx01 = _mm_add_ps(ms1, ms6);
	mx02 = _mm_add_ps(ms2, ms5);
	mx03 = _mm_add_ps(ms3, ms4);
	mx04 = _mm_sub_ps(ms0, ms7);
	mx05 = _mm_sub_ps(ms1, ms6);
	mx06 = _mm_sub_ps(ms2, ms5);
	mx07 = _mm_sub_ps(ms3, ms4);
	mx08 = _mm_add_ps(mx00, mx03);
	mx09 = _mm_add_ps(mx01, mx02);
	mx0a = _mm_sub_ps(mx00, mx03);
	mx0b = _mm_sub_ps(mx01, mx02);

	mx0c = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.38703984532215f), mx04), _mm_mul_ps(_mm_set1_ps(0.275899379282943f), mx07));
	mx0d = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.17587560241936f), mx05), _mm_mul_ps(_mm_set1_ps(+0.785694958387102f), mx06));
	mx0e = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-0.785694958387102f), mx05), _mm_mul_ps(_mm_set1_ps(+1.17587560241936f), mx06));
	mx0f = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.275899379282943f), mx04), _mm_mul_ps(_mm_set1_ps(-1.38703984532215f), mx07));
	mx10 = _mm_mul_ps(c0353, _mm_sub_ps(mx0c, mx0d));
	mx11 = _mm_mul_ps(c0353, _mm_sub_ps(mx0e, mx0f));

	__m128 v = _mm_mul_ps(c0353, _mm_add_ps(mx08, mx09));
	__m128 msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	// keep 00 coef.
	__m128 v2 = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(dst, _mm_blend_ps(v2, v, 1));

	v = _mm_mul_ps(c0353, _mm_add_ps(mx0c, mx0d));
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(dst + 8, v);
	
	v = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.461939766255643f), mx0a), _mm_mul_ps(_mm_set1_ps(0.191341716182545f), mx0b));
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(dst + 16, v);

	v = _mm_mul_ps(c0707, _mm_sub_ps(mx10, mx11));
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(dst + 24, v);

	dst += 4;

	v = _mm_mul_ps(c0353, _mm_sub_ps(mx08, mx09));
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(dst, v);

	v = _mm_mul_ps(c0707, _mm_add_ps(mx10, mx11));
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(dst + 8, v);

	v = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.191341716182545f), mx0a), _mm_mul_ps(_mm_set1_ps(-0.461939766255643f), mx0b));
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(dst + 16, v);

	v = _mm_mul_ps(c0353, _mm_add_ps(mx0e, mx0f));
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(dst + 24, v);

	ms0 = md10;
	ms1 = md11;
	ms2 = md12;
	ms3 = md13;
	ms4 = md14;
	ms5 = md15;
	ms6 = md16;
	ms7 = md17;

	mx00 = _mm_add_ps(ms0, ms7);
	mx01 = _mm_add_ps(ms1, ms6);
	mx02 = _mm_add_ps(ms2, ms5);
	mx03 = _mm_add_ps(ms3, ms4);
	mx04 = _mm_sub_ps(ms0, ms7);
	mx05 = _mm_sub_ps(ms1, ms6);
	mx06 = _mm_sub_ps(ms2, ms5);
	mx07 = _mm_sub_ps(ms3, ms4);
	mx08 = _mm_add_ps(mx00, mx03);
	mx09 = _mm_add_ps(mx01, mx02);
	mx0a = _mm_sub_ps(mx00, mx03);
	mx0b = _mm_sub_ps(mx01, mx02);

	mx0c = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.38703984532215f), mx04), _mm_mul_ps(_mm_set1_ps(0.275899379282943f), mx07));
	mx0d = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.17587560241936f), mx05), _mm_mul_ps(_mm_set1_ps(+0.785694958387102f), mx06));
	mx0e = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-0.785694958387102f), mx05), _mm_mul_ps(_mm_set1_ps(+1.17587560241936f), mx06));
	mx0f = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.275899379282943f), mx04), _mm_mul_ps(_mm_set1_ps(-1.38703984532215f), mx07));
	mx10 = _mm_mul_ps(c0353, _mm_sub_ps(mx0c, mx0d));
	mx11 = _mm_mul_ps(c0353, _mm_sub_ps(mx0e, mx0f));

	v = _mm_mul_ps(c0353, _mm_add_ps(mx08, mx09));
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(dst+28, v);

	v = _mm_mul_ps(c0353, _mm_add_ps(mx0c, mx0d));
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(dst + 36, v);

	v = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.461939766255643f), mx0a), _mm_mul_ps(_mm_set1_ps(0.191341716182545f), mx0b));
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(dst + 44, v);

	v = _mm_mul_ps(c0707, _mm_sub_ps(mx10, mx11));
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(dst + 52, v);

	v = _mm_mul_ps(c0353, _mm_sub_ps(mx08, mx09));
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(dst+32, v);

	v = _mm_mul_ps(c0707, _mm_add_ps(mx10, mx11));
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(dst + 40, v);

	v = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.191341716182545f), mx0a), _mm_mul_ps(_mm_set1_ps(-0.461939766255643f), mx0b));
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(dst + 48, v);

	v = _mm_mul_ps(c0353, _mm_add_ps(mx0e, mx0f));
	msk = _mm_cmpgt_ps(_mm_and_ps(v, *(const __m128*)v32f_absmask), mth);
	v = _mm_blendv_ps(zeros, v, msk);
	_mm_store_ps(dst + 56, v);
}

static void idct81d_sse_GT(const float *s, float *d)
{
	float* dst = d;
	float* src = (float*)s;
	const __m128 c1414 = _mm_set1_ps(1.4142135623731f);
	const __m128 c0250 = _mm_set1_ps(0.25f);
	const __m128 c0353 = _mm_set1_ps(0.353553390593274f);
	const __m128 c0707 = _mm_set1_ps(0.707106781186547f);

	for (int i = 0; i < 2; i++)
	{
		__m128 ms0 = _mm_load_ps(src);
		__m128 ms1 = _mm_load_ps(src + 8);
		__m128 ms2 = _mm_load_ps(src + 16);
		__m128 ms3 = _mm_load_ps(src + 24);
		__m128 ms4 = _mm_load_ps(src + 32);
		__m128 ms5 = _mm_load_ps(src + 40);
		__m128 ms6 = _mm_load_ps(src + 48);
		__m128 ms7 = _mm_load_ps(src + 56);
		
		__m128 mx00 = _mm_mul_ps(c1414, ms0);
		__m128 mx01 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.38703984532215f), ms1), _mm_mul_ps(_mm_set1_ps(0.275899379282943f), ms7)); 
		__m128 mx02 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.30656296487638f), ms2), _mm_mul_ps(_mm_set1_ps(0.541196100146197f), ms6)); 
		__m128 mx03 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.17587560241936f), ms3), _mm_mul_ps(_mm_set1_ps(0.785694958387102f), ms5)); 
		__m128 mx04 = _mm_mul_ps(c1414, ms4);
		__m128 mx05 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-0.785694958387102f), ms3), _mm_mul_ps(_mm_set1_ps(+1.17587560241936f), ms5));
		__m128 mx06 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.541196100146197f), ms2), _mm_mul_ps(_mm_set1_ps(-1.30656296487638f), ms6));
		__m128 mx07 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-0.275899379282943f), ms1), _mm_mul_ps(_mm_set1_ps(1.38703984532215f), ms7));
		__m128 mx09 = _mm_add_ps(mx00 , mx04);
		__m128 mx0a = _mm_add_ps(mx01 , mx03);
		__m128 mx0b = _mm_mul_ps(c1414, mx02);
		__m128 mx0c = _mm_sub_ps(mx00 , mx04);
		__m128 mx0d = _mm_sub_ps(mx01 , mx03);
		__m128 mx0e = _mm_mul_ps(c0353, _mm_sub_ps(mx09, mx0b));
		__m128 mx0f = _mm_mul_ps(c0353, _mm_add_ps(mx0c, mx0d));
		__m128 mx10 = _mm_mul_ps(c0353, _mm_sub_ps(mx0c, mx0d));
		__m128 mx11 = _mm_mul_ps(c1414, mx06);
		__m128 mx12 = _mm_add_ps(mx05 , mx07);
		__m128 mx13 = _mm_sub_ps(mx05 , mx07);
		__m128 mx14 = _mm_mul_ps(c0353, _mm_add_ps(mx11, mx12));
		__m128 mx15 = _mm_mul_ps(c0353, _mm_sub_ps(mx11, mx12));
		__m128 mx16 = _mm_mul_ps(_mm_set1_ps(0.5f), mx13); 

		_mm_store_ps(dst +  0, _mm_add_ps(_mm_mul_ps(c0250, _mm_add_ps(mx09, mx0b)) , _mm_mul_ps(c0353, mx0a)));
		_mm_store_ps(dst +  8, _mm_mul_ps(c0707, _mm_add_ps(mx0f , mx15)));
		_mm_store_ps(dst + 16, _mm_mul_ps(c0707, _mm_sub_ps(mx0f , mx15)));
		_mm_store_ps(dst + 24, _mm_mul_ps(c0707, _mm_add_ps(mx0e , mx16)));
		_mm_store_ps(dst + 32, _mm_mul_ps(c0707, _mm_sub_ps(mx0e , mx16)));
		_mm_store_ps(dst + 40, _mm_mul_ps(c0707, _mm_sub_ps(mx10 , mx14)));
		_mm_store_ps(dst + 48, _mm_mul_ps(c0707, _mm_add_ps(mx10 , mx14)));
		_mm_store_ps(dst + 56, _mm_sub_ps(_mm_mul_ps(c0250, _mm_add_ps(mx09, mx0b)), _mm_mul_ps(c0353, mx0a)));
		dst += 4;
		src += 4;
	}
}


//paper LLM89
//C. Loeffler, A. Ligtenberg, and G. S. Moschytz, 
//"Practical fast 1-D DCT algorithms with 11 multiplications,"
//Proc. Int'l. Conf. on Acoustics, Speech, and Signal Processing (ICASSP89), pp. 988-991, 1989.

static void fDCT2D8x4_and_threshold_keep00_32f(const float* x, float* y, float thresh)
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

static void fDCT2D8x4_and_threshold_32f(const float* x, float* y, float thresh)
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


static void fDCT2D8x4noscale_32f(const float* x, float* y)
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

static void fDCT2D8x4_32f(const float* x, float* y)
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

void fDCT8x8GT(const float* s, float* d)
{
	fdct88_sse_GT(s, d);
	/*fdct81d_sse_GT(s, d);
	transpose8x8(d);
	fdct81d_sse_GT(d, d);
	transpose8x8(d);*/
}

void iDCT8x8GT(const float* s, float* d)
{
	idct81d_sse_GT(s, d);
	transpose8x8(d);
	idct81d_sse_GT(d, d);
	transpose8x8(d);
}

void fDCT8x8(const float* s, float* d)
{
	__declspec(align(16)) float temp[64];
	transpose8x8(s, temp);

	fDCT2D8x4_32f(temp, d);
	fDCT2D8x4_32f(temp + 4, d + 4);

	transpose8x8(d, temp);
	fDCT2D8x4_32f(temp, d);
	fDCT2D8x4_32f(temp + 4, d + 4);
}

void fDCT8x8(const float* s, float* d, float* temp)
{
	transpose8x8(s, temp);

	fDCT2D8x4_32f(temp, d);
	fDCT2D8x4_32f(temp + 4, d + 4);

	transpose8x8(d, temp);
	fDCT2D8x4_32f(temp, d);
	fDCT2D8x4_32f(temp + 4, d + 4);
}

static void fDCT1Dllm_32f(const float* x, float* y)
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

static void fDCT2Dllm_32f(const float* s, float* d, float* temp)
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

static void iDCT1Dllm_32f(const float* y, float* x)
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

static void iDCT2Dllm_32f(const float* s, float* d, float* temp)
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

static void iDCT2D8x4_32f(const float* y, float* x)
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

void iDCT8x8(const float* s, float* d)
{
	__declspec(align(16)) float temp[64];
	transpose8x8((float*)s, temp);
	iDCT2D8x4_32f(temp, d);
	iDCT2D8x4_32f(temp + 4, d + 4);

	transpose8x8(d, temp);
	iDCT2D8x4_32f(temp, d);
	iDCT2D8x4_32f(temp + 4, d + 4);
}

void iDCT8x8(const float* s, float* d, float* temp)
{
	transpose8x8((float*)s, temp);
	iDCT2D8x4_32f(temp, d);
	iDCT2D8x4_32f(temp + 4, d + 4);

	transpose8x8(d, temp);
	iDCT2D8x4_32f(temp, d);
	iDCT2D8x4_32f(temp + 4, d + 4);
}

static void fDCT8x8_32f_and_threshold(const float* s, float* d, float threshold, float* temp)
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

void fDCT8x8_threshold_keep00_iDCT8x8(float* s, float threshold)
{
	fDCT8x8GT_threshold_keep00(s, s, threshold);

	//fDCT2D8x4_32f(s, s);
	//fDCT2D8x4_32f(s + 4, s + 4);
	//transpose8x8(s);
	//fDCT2D8x4_and_threshold_keep00_32f(s, s, threshold);
	////fDCT2D8x4_and_threshold_32f(s, s, threshold);
	//fDCT2D8x4_and_threshold_32f(s + 4, s + 4, threshold);
	//ommiting transform
	////transpose8x8(s);
	////transpose8x8(s);

//	idct81d_sse_GT(s, s);
	iDCT2D8x4_32f(s, s);
	iDCT2D8x4_32f(s + 4, s + 4);
	transpose8x8(s);
//	idct81d_sse_GT(s, s);
	iDCT2D8x4_32f(s, s);
	iDCT2D8x4_32f(s + 4, s + 4);

	return;
}

int fDCT8x8__threshold_keep00_iDCT8x8_nonzero(float* s, float threshold)
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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//4x4
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static void dct4x4_1d_llm_fwd_sse(float* s, float* d)//8add, 4 mul
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

static void dct4x4_1d_llm_fwd_sse_and_transpose(float* s, float* d)//8add, 4 mul
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

static void dct4x4_1d_llm_inv_sse(float* s, float* d)
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

static void dct4x4_1d_llm_inv_sse_and_transpose(float* s, float* d)
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

void iDCT4x4(float* a, float* b)
{
	__declspec(align(16)) float temp[16];
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

void iDCT4x4(float* a, float* b, float* temp)
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

void fDCT4x4(float* a, float* b)
{
	__declspec(align(16)) float temp[16];
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

void fDCT4x4(float* a, float* b, float* temp)
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

static void fDCT2D4x4_and_threshold_keep00_32f(float* s, float* d, float thresh)
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

static void fDCT2D4x4_and_threshold_32f(float* s, float* d, float thresh)
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

void fDCT4x4_threshold_keep00_iDCT4x4(float* s, float threshold)
{
	dct4x4_1d_llm_fwd_sse_and_transpose(s, s);
	fDCT2D4x4_and_threshold_keep00_32f(s, s, 4 * threshold);
	//fDCT2D8x4_and_threshold_32f(s, s, 4 * threshold);
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

int fDCT4x4_threshold_keep00_iDCT4x4_nonzero(float* s, float threshold)
{
	dct4x4_1d_llm_fwd_sse_and_transpose(s, s);

	fDCT2D4x4_and_threshold_keep00_32f(s, s, 4 * threshold);
	//fDCT2D8x4_and_threshold_32f(s, s, 4 * threshold);

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

//////////////////////////////////////////////////////////////
//2x2
//////////////////////////////////////////////////////////////
//2x2

static void dct1d2_32f(float* src, float* dest)
{
	dest[0] = 0.7071067812f*(src[0] + src[1]);
	dest[1] = 0.7071067812f*(src[0] - src[1]);
}

void fDCT2x2(float* src, float* dest, float* temp)
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

void iDCT2x2(float* src, float* dest, float* temp)
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

static void dct1d2_32f_and_thresh(float* src, float* dest, float thresh)
{
	float v = 0.7071068f*(src[0] + src[1]);
	dest[0] = (abs(v) < thresh) ? 0.f : v;
	v = 0.7071068f*(src[0] - src[1]);
	dest[1] = (abs(v) < thresh) ? 0 : v;
}

static void fDCT2x2_32f_and_threshold(float* src, float* dest, float* temp, float thresh)
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

void fDCT2x2_2pack_thresh_keep00_iDCT2x2_2pack(float* src, float* dest, float thresh)
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