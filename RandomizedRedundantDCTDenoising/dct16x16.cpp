#include <nmmintrin.h> //SSE4.2

//Plonka, Gerlind, and Manfred Tasche. "Fast and numerically stable algorithms for discrete cosine transforms." Linear algebra and its applications 394 (2005) : 309 - 345.

void transpose16x16(float* src);

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
	;
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
	//fdct16_1d(src, dest);
	transpose16x16(dest);

	fdct161d_sse(dest, dest);
	//fdct16_1d(dest, dest);
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