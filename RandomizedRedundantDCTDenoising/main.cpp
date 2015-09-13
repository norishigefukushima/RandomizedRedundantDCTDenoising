#include "RedundantDXTDenoise.h"

using namespace std;
using namespace cv;
using namespace cp;
using namespace lab;

int main()
{
	Mat src = imread("img/kodim03.png");
	Mat noise;
	Mat dest;
	Mat dest2;
	float sigma = 20.f;
	addNoise(src, noise, sigma);
	int iteration = 10000;

	//cout << YPSNR(src, noise) << endl;
	RedundantDXTDenoise dctDenoise;

	RandomizedRedundantDXTDenoise rrdct;

	for (int i = 0; i < iteration; i++)
	{
		CalcTime t("8x8 R-DCT");
		//rrdct(noise, dest, sigma, Size(8, 8));
		//noise.copyTo(dest);
		dctDenoise(noise, dest, sigma, Size(8, 8));
		cout << YPSNR(src, dest) << endl;
		imshow("test", dest); waitKey(1);
	}
	cout << YPSNR(src, dest) << endl;

	for (int i = 0; i < iteration; i++)
	{
		CalcTime t("4x4 DCT");
		dctDenoise(noise, dest, sigma, Size(4, 4));
	}
	cout << YPSNR(src, dest) << endl;

	for (int i = 0; i < iteration; i++)
	{
		CalcTime t("8x8 DCT");
		dctDenoise(noise, dest, sigma, Size(8, 8));
	}
	cout << YPSNR(src, dest) << endl;
	dest.setTo(0);
	guiAlphaBlend(src, dest);

	for (int i = 0; i < iteration; i++)
	{
		CalcTime t("16x16 DCT");
		dctDenoise(noise, dest, sigma, Size(16, 16));
	}

	cout << YPSNR(src, dest) << endl;

	dctDenoise.isSSE = false;
	for (int i = 0; i < iteration; i++)
	{
		CalcTime t("16x16 DCT");
		dctDenoise(noise, dest2, sigma, Size(16, 16));
	}
	cout << YPSNR(src, dest2) << endl;
	dctDenoise.isSSE = true;
	for (int i = 0; i < iteration; i++)
	{
		CalcTime t("4x4 DHT");
		dctDenoise(noise, dest, sigma, Size(4, 4), RedundantDXTDenoise::BASIS::DHT);
	}
	cout << YPSNR(src, dest) << endl;
	for (int i = 0; i < iteration; i++)
	{
		CalcTime t("8x8 DHT");
		dctDenoise(noise, dest, sigma, Size(8, 8), RedundantDXTDenoise::BASIS::DHT);
	}
	cout << YPSNR(src, dest) << endl;
	for (int i = 0; i < iteration; i++)
	{
		CalcTime t("16x16 DHT");
		dctDenoise(noise, dest, sigma, Size(16, 16), RedundantDXTDenoise::BASIS::DHT);
	}
	cout << YPSNR(src, dest) << endl;

	{
		CalcTime t("4x4 OpenCV");
		xphoto::dctDenoising(noise, dest2, sigma, 4);
	}
	cout << YPSNR(src, dest2) << endl;
	{
		CalcTime t("8x8 OpenCV");
		xphoto::dctDenoising(noise, dest2, sigma, 8);
	}
	cout << YPSNR(src, dest2) << endl;
	{
		CalcTime t("16x16 OpenCV");
		xphoto::dctDenoising(noise, dest2, sigma, 16);
	}
	cout << YPSNR(src, dest2) << endl;

	//guiAlphaBlend(src, dest);
	return 0;
}