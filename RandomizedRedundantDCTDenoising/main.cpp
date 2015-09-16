#include "RedundantDXTDenoise.h"

using namespace std;
using namespace cv;

void guiDenoise(Mat& src, Mat& dest, string wname = "denoise")
{
	namedWindow(wname);

	int sw = 0; createTrackbar("sw", wname, &sw, 3);
	int blksize = 3; createTrackbar("bsize2^n", wname, &blksize, 6);
	int snoise = 20; createTrackbar("noise", wname, &snoise, 100);
	int thresh = 200; createTrackbar("thresh", wname, &thresh, 2000);
	int r = 2; createTrackbar("r", wname, &r, 20);
	int radius = 2; createTrackbar("rad", wname, &radius, 20);
	int h = 50; createTrackbar("h", wname, &h, 2500);
	int h_c = 180; createTrackbar("hc", wname, &h_c, 2500);

	Mat noise;
	addNoise(src, noise, snoise);
	int key = 0;
	RedundantDXTDenoise dctDenoise;
	RRDXTDenoise rrdct;
	rrdct.generateSamplingMaps(src.size(), Size(16, 16), 1, r, RRDXTDenoise::SAMPLING::LATTICE);
	while (key != 'q')
	{
		int bsize = (int)pow(2.0, blksize);
		Size block = Size(bsize, bsize);
		if (key == 'n') addNoise(src, noise, snoise);
		{
			CalcTime t;
			if (sw == 0) dctDenoise(noise, dest, thresh / 10.f, block);
			else if (sw == 1)
			{
				rrdct.generateSamplingMaps(src.size(), block, 1, r, RRDXTDenoise::SAMPLING::LATTICE);
				rrdct(noise, dest, thresh / 10.f, block);
			}
			else if (sw == 2)
			{
				rrdct.generateSamplingMaps(src.size(), block, 1, r, RRDXTDenoise::SAMPLING::LATTICE);
				rrdct(noise, dest, thresh / 10.f, block, RRDXTDenoise::BASIS::DHT);
			}
			else if (sw == 3)
			{
				fastNlMeansDenoisingColored(noise, dest, h / 10.f, h_c / 10.f, 3, 2 * radius + 1);
			}
		}
		cout << YPSNR(src, dest) << " dB" << endl;
		imshow(wname, dest);
		key = waitKey(1);
	}
}

int main(int argc, const char *argv[])
{
	{
		Mat src_ = imread("img/kodim07.png");
		Mat src, dest;
		resize(src_, src, Size(1024, 1024));
		guiDenoise(src, dest);
	}

	const String keys =
	{
		"{help h|   | print this message}"
		"{@src  |   |src image}"
		"{@dest |   |dest image}"
		"{@noise|   |noise stddev}"
		"{b     |DCT|basis DCT or DHT}"
		"{bw    |8  |block width      }"
		"{bh    |0  |block height. if(bh==0) bh=bw}"
		"{g gui |   |call interactive denoising}"
		"{d     |0  |minimum d for sampling. if(d==0) auto}"
	};
	cv::CommandLineParser parser(argc, argv, keys);
	if (argc == 1)
	{
		parser.printMessage();
		return 0;
	}

	Mat src = imread(parser.get<string>(0));
	Mat dest;
	if (src.empty())
	{
		cout << "file path:" << parser.get<string>(0) << "is not correct." << endl;
		return 0;
	}

	if (parser.has("gui"))
	{
		//call gui demo
		guiDenoise(src, dest);
	}
	else
	{
		string basis = parser.get<string>("b");
		int bw = parser.get<int>("bw");
		int bh = parser.get<int>("bh");
		bh = (bh == 0) ? bw : bh;
		Size psize = Size(bw, bh);
		int d = parser.get<int>("d");
		if (d == 0) d = min(bw, bh) / 3;
		float sigma = parser.get<float>(2);

		RedundantDXTDenoise::BASIS dxtbasis;
		if (basis == "DCT") dxtbasis = RedundantDXTDenoise::BASIS::DCT;
		else if (basis == "DHT") dxtbasis = RedundantDXTDenoise::BASIS::DHT;

		RRDXTDenoise rrdxt;
		rrdxt.generateSamplingMaps(src.size(), psize, 1, d, RRDXTDenoise::SAMPLING::LATTICE);
		rrdxt(src, dest, sigma, psize, dxtbasis);
	}

	imwrite(parser.get<string>(1), dest);
	return 0;
}