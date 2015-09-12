# RandomizedRedundantDCTDenoising

The code is written for demonstration of the paper [1].
In this paper, the DCT-based denoising [2] is accelerated by using a randomized algorithm.
Some modifications improve denoising performance in term of PSNR.

The code is 100x faster than the OpenCV's implementation for the paper [2] (cv::xphoto::dctDenoising).
Optionally, we can use DHT (discrete Walsh–Hadamard fransform) for fast computation.

* [1] S. Fujita, N. Fukushima, M. Kimura, and Y. Ishibashi, "Randomized Redundant DCT: An Efficient Denoising by using Random Subsampling of DCT Patches," Proc. Siggraph Asia, Technical Brief, Nov. 2015.
* [2] Guoshen Yu, and Guillermo Sapiro, DCT image denoising: a simple and effective image denoising algorithm, Image Processing On Line, 1 (2011). http://dx.doi.org/10.5201/ipol.2011.ys-dct
http://www.ipol.im/pub/art/2011/ys-dct/





