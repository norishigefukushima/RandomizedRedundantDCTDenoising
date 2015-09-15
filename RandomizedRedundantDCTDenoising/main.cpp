#include "RedundantDXTDenoise.h"

using namespace std;
using namespace cv;
using namespace cp;
using namespace lab;

void guiDenoise(Mat& src, string wname = "denoise")
{
	namedWindow(wname);
	int sw = 0; createTrackbar("sw", wname, &sw, 4);
	int snoise = 20; createTrackbar("noise", wname, &snoise, 100);
	int r = 2; createTrackbar("r", wname, &r, 20);
	int sigma_s = 50; createTrackbar("ss", wname, &sigma_s, 2500);
	int sigma_c = 180; createTrackbar("sc", wname, &sigma_c, 2500);

	Mat noise;
	Mat dest;
	addNoise(src, noise, snoise);
	int key = 0;
	RedundantDXTDenoise dctDenoise;
	RandomizedRedundantDXTDenoise rrdct;
	rrdct.generateSamplingMaps(src.size(), Size(8, 8), 1, r, RandomizedRedundantDXTDenoise::SAMPLING::LATTICE);
	while (key!='q')
	{
		{
			CalcTime t;
			if (sw == 0) dctDenoise(noise, dest, sigma_c / 10.0, Size(8, 8));
			else if (sw == 1)
			{
				rrdct.generateSamplingMaps(src.size(), Size(8, 8), 1, r, RandomizedRedundantDXTDenoise::SAMPLING::LATTICE);
				rrdct.interlace(noise, dest, sigma_c / 10.0, Size(8, 8));
			}
			else if (sw == 2)
			{
				rrdct.generateSamplingMaps(src.size(), Size(16, 16), 1, r, RandomizedRedundantDXTDenoise::SAMPLING::LATTICE);
				rrdct.interlace(noise, dest, sigma_c / 10.0, Size(16, 16));
			}
			else if (sw == 3)
			{
				nonLocalMeansFilter(noise, dest, 3, 2 * r + 1, 1.4*sigma_c / 10.0);
			}
			//bilateralFilter(noise, dest, 2*r+1, sigma_c / 10.0, sigma_s / 10.0, BORDER_REFLECT);
			//
			
			//rrdct.interlace(noise, dest, sigma_c / 10.0, Size(8, 8));

			
		}
		cout << YPSNR(src, dest) << endl;
		imshow(wname, dest);
		key = waitKey(1);
	}
}

int main()
{
	Mat src_ = imread("img/kodim04.png");
	Mat src;
	resize(src_, src, Size(1024, 1024));
	guiDenoise(src);
	Mat noise;
	Mat dest;
	Mat dest2;
	float sigma = 20.f;
	addNoise(src, noise, sigma);
	int iteration = 100000;

	//cout << YPSNR(src, noise) << endl;
	RedundantDXTDenoise dctDenoise;

	RandomizedRedundantDXTDenoise rrdct;

	//dctDenoise.isSSE = false;
	{
		
		Stat st;
		rrdct.generateSamplingMaps(noise.size(), Size(8,8),1, 3, RandomizedRedundantDXTDenoise::SAMPLING::POISSONDISK);
		CalcTime t("fDCT", 0,false);
		
		for (int i = 0; i < iteration; i++)
		{
			t.start();
			//xphoto::dctDenoising(noise, dest, sigma, 8);
		//	rrdct.interlace(noise, dest, sigma, Size(16, 16));
			//rrdct.interlace(noise, dest, sigma, Size(8, 8));
			
			
			nonLocalMeansFilter(noise, dest, 3, 7, sigma * 0.6);
	
				
			//showMatInfo(dest);
			//noise.copyTo(dest);
			//
			//dctDenoise(noise, dest, sigma, Size(8, 8));
			//dctDenoise(noise, dest, sigma, Size(16,16));
			st.push_back(t.getTime());
			
			//guiAlphaBlend(src, dest);
			cout << YPSNR(src, dest) << endl;
			imshow("test", dest); waitKey(1);
			
			cout<<st.getMedian() << "ms"<<endl;
		}
	}
	//getchar();
	cout << YPSNR(src, dest) << endl;
	/*
	for (int i = 0; i < iteration; i++)
	{
		CalcTime t("4x4 DCT");
		dctDenoise(noise, dest, sigma, Size(4, 4));
	}
	cout << YPSNR(src, dest) << endl;
	*/
	for (int i = 0; i < iteration; i++)
	{
		CalcTime t("8x8 DCT");
		dctDenoise(noise, dest2, sigma, Size(8, 8));
	}
	cout << YPSNR(src, dest2) << endl;
	//dest.setTo(0);
//	guiAlphaBlend(dest, dest2);
	//guiAlphaBlend(src, dest);

	for (int i = 0; i < iteration; i++)
	{
		CalcTime t("16x16 DCT");
		dctDenoise(noise, dest, sigma, Size(16, 16));
	}

	cout << YPSNR(src, dest) << endl;

//	dctDenoise.isSSE = false;
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