/* Segments.h
*
* These functions handle the segmenting of the characters in the image. 
* It will create the horizontal and vertical segments, create pairs for these segments, and then
* use these pairs to create rectangles bounding the characters. These rectangles are shrunk to better fit
* the characters and allow for more accurate classification and identification.
*/

#ifndef SEGMENTS_H
#define SEGMENTS_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

// rectangle struct valuable for sorting by id
struct Rectangle {
	int id; // position in the string
	int x;
	int y;
	int width;
	int height;
};

void initArray(int* a, int length);

/* functions to create and draw segments */
int* horizontalSegments(Mat& src);
int* verticalSegments(Mat& src);
Mat drawHorizontalSegments(int* seg, int rows, int cols);
Mat drawVerticalSegments(int* seg, int rows, int cols);

/* functions to manipulate segments */
vector<pair<int, int> > createSegmentPairs(int* seg, int segSize);
vector<Rectangle> getRectangles(vector<pair<int, int> > vPairs, vector<pair<int, int> > hPairs);
vector<Rectangle> shrinkRectangles(Mat& im, vector<Rectangle> r);
vector<Rectangle> takeRectangles(vector<Rectangle> r, int number);
void drawRectangles(Mat& im, vector<Rectangle> r);

void classify(Mat &image, Mat& trainingData, Mat& trainingClasses, vector<Rectangle> r);
vector<Rectangle> segmentation(Mat& img, int numSquares);

#endif
