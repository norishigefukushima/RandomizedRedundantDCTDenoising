static void fdct81d_base(const float *src, float *dst)
{
	for (int i = 0; i < 8; i++)
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
		dst += 8;
		src += 8;
	}
}

static void   idct81d_base(const float *src, float *dst)
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