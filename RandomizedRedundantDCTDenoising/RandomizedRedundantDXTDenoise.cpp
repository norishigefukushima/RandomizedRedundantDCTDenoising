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
	return sqrt((pt1.x - pt2.x)*(pt1.x - pt2.x) + (pt1.y - pt2.y)*(pt1.y - pt2.y));
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
	int hd = d / 2;
	for (int j = d/2; j < dest.rows -d/2; j += d)
	{
		uchar* m = dest.ptr(j);
		for (int i = d/2; i < dest.cols -d/2; i += d)
		{
			int x = rng.uniform(-hd, hd);
			int y = dest.cols*rng.uniform(-hd, hd);
			m[y+x+i] = 1;
		}
	}
}

void RandomizedRedundantDXTDenoise::setSampling(SAMPLING samplingType, int d)
{
	RNG rng;
	switch (samplingType)
	{
	default:
	case FULL:
		sampleMap = Mat::ones(size, CV_8U);
		break;

	case LATTICE:
		sampleMap = Mat::zeros(size, CV_8U);
		setLattice(sampleMap, d, rng);
		break;
	case POISSONDISK:
		sampleMap = Mat::zeros(size, CV_8U);
		setPoissonDisk(sampleMap, d, rng);
		break;
	}
	//cp::showMatInfo(sampleMap);
}

void RandomizedRedundantDXTDenoise::operator()(Mat& src, Mat& dest, float sigma, Size psize, BASIS transform_basis)
{
	int numThreads = getNumThreads();
	Mat temp;
	if (src.depth() != CV_32F) src.convertTo(temp, CV_MAKETYPE(CV_32F, src.channels()));
	else temp = src.clone();

	basis = transform_basis;
	if (src.size() != size || src.channels() != channel || psize != patch_size) init(src.size(), src.channels(), psize);

	int w = src.cols + 2 * psize.width;
	w = ((4 - w % 4) % 4);
	int h = src.rows;
	h = ((numThreads - h % numThreads) % numThreads);

	copyMakeBorder(temp, im, psize.height, 3*psize.height+h, psize.width, psize.width + w, cv::BORDER_REPLICATE);
	int imwidth = im.cols;
	int imheight = im.rows;
	int imarea = imwidth * imheight;
	int wstep = imwidth - psize.width + 1;
	int hstep = imheight - psize.height + 1;
	channel = im.channels();
	float th = 3.f * sigma;

	// pre processing
	if (channel == 1)
		im.copyTo(buff);
	else cvtColorBGR2PLANE(im, buff);
	
	if (channel == 3)	decorrelateColorForward(buff.ptr<float>(0), buff.ptr<float>(0), imwidth, imheight);

	// body
//	setLUT();
	//setSampling(SAMPLING::FULL, 0);
	setSampling(SAMPLING::LATTICE, psize.width/2-1);
	

	if (sum.size() != buff.size())sum = Mat::zeros(buff.size(), CV_32F);
	else sum.setTo(0);

	const int slice_real_height = (src.rows+h) / numThreads;
	const int hstep2 = slice_real_height + psize.height - 1;
	const Size clip_size(imwidth, slice_real_height + 2 * psize.height);

	for (int l = 0; l<channel; l++)
	{
		float* s = buff.ptr<float>(l*imheight);
		float* d = sum.ptr<float>(l*imheight);
		RandomizedRDCTDenoise8x8_Invoker body(s, d, sampleMap, th, clip_size, patch_size, imwidth, imheight, wstep, hstep2, slice_real_height);
		parallel_for_(Range(0, numThreads), body);
	}

	if (channel == 3)
	{
		decorrelateColorInvert(sum.ptr<float>(0), sum.ptr<float>(0), imwidth, imheight);
		cvtColorPLANE2BGR(sum, im);
	}
	else
	{
		im = sum;
	}

	if (src.depth() != CV_32F) im.convertTo(temp, src.type());
	else temp = im;


	
	temp(Rect(psize.width, 0, src.cols, src.rows)).copyTo(dest);

	//cp::guiAlphaBlend(dest, dest);
	//cp::guiAlphaBlend(temp, temp);
}
