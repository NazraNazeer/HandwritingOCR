// main.cpp
// [tbd]
// Author: Johan Brusa, Kyle Farmer, Tyler Hilde

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;



// main - [tbd]
// precondition: [tbd]
// postconditions: [tbd]
int main(int argc, char* argv[])
{	
		Mat image = imread("test.jpg");	
		namedWindow("Original Image");
		imshow("Original Image", image);
		waitKey(0);

		return 0;
}
