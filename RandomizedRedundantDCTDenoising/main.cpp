#include "RedundantDXTDenoise.h"

using namespace std;
using namespace cv;
using namespace cp;
using namespace lab;

int main()
{
	Mat src = imread("img/Kodak/kodim02.png");
	Mat noise;
	Mat dest;
	Mat dest2;
	float sigma = 10.f;
	addNoise(src, noise, sigma);
	int iteration = 1;

	//cout << YPSNR(src, noise) << endl;
	RedundantDXTDenoise dctDenoise;

	{
		CalcTime t("4x4 DCT");
		for (int i = 0; i < iteration; i++)
			dctDenoise(noise, dest, sigma, Size(4, 4));
	}
	cout << YPSNR(src, dest) << endl;
	{
		CalcTime t("8x8 DCT");
		for (int i = 0; i < iteration; i++)
			dctDenoise(noise, dest, sigma, Size(8, 8));
	}
	cout << YPSNR(src, dest) << endl;
	dest.setTo(0);


	{
		CalcTime t("16x16 DCT");
		for (int i = 0; i < iteration; i++)
			dctDenoise(noise, dest, sigma, Size(16, 16));
	}

	cout << YPSNR(src, dest) << endl;

	dctDenoise.isSSE = false;
	{
		CalcTime t("16x16 DCT");
		for (int i = 0; i < iteration; i++)
			dctDenoise(noise, dest2, sigma, Size(16, 16));
	}
	cout << YPSNR(src, dest2) << endl;
	dctDenoise.isSSE = true;
	guiAlphaBlend(dest, dest2);
	//dctDenoise.isSSE = false;
	{
		CalcTime t("4x4 DHT");
		for (int i = 0; i < iteration; i++)
			dctDenoise(noise, dest, sigma, Size(4, 4), RedundantDXTDenoise::BASIS::DHT);
	}
	cout << YPSNR(src, dest) << endl;
	{
		CalcTime t("8x8 DHT");
		for (int i = 0; i < iteration; i++)
			dctDenoise(noise, dest, sigma, Size(8, 8), RedundantDXTDenoise::BASIS::DHT);
	}
	cout << YPSNR(src, dest) << endl;

	{
		CalcTime t("16x16 DHT");
		for (int i = 0; i < iteration; i++)
			dctDenoise(noise, dest, sigma, Size(16, 16), RedundantDXTDenoise::BASIS::DHT);
	}
	cout << YPSNR(src, dest) << endl;

	/*	{
			CalcTime t;
			xphoto::dctDenoising(noise, dest2, sigma);
			}*/

	guiAlphaBlend(src, dest);
	cout << YPSNR(src, dest2) << endl;
	return 0;
}