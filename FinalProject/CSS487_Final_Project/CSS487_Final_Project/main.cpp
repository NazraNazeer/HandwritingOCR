#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <stdlib.h>

#include "Segments.h"
#include "main.h"

const int RESIZE_FACTOR = 2;

int main(int argc, char* argv[]) {

	Ptr<ml::KNearest> kNN(ml::KNearest::create());
	Mat trainingClasses, trainingData;

	for (int i = 0; i < 9; i++)
	{
		string imgName = "abc(" + to_string(i) + ").png";
		Mat sourceImage = imread(imgName);

		cvtColor(sourceImage, sourceImage, COLOR_BGR2GRAY);

		// Is it loaded?
		if (!sourceImage.data)
			return -1;

		// Resize the image by resize factor, don't need for our data set
		//if (i > 3)
		//	resize(sourceImage, sourceImage, Size(sourceImage.cols * RESIZE_FACTOR, sourceImage.rows * RESIZE_FACTOR));

		// Define our final image
		Mat trainImage = sourceImage.clone();


		// Apply adaptive threshold
		//GaussianBlur(trainImage, trainImage, Size(3, 3), 0);
		adaptiveThreshold(trainImage, trainImage, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 11, 2);


		// Use the thresholded image as a mask, used for certain captchas
		/*Mat tmp;
		sourceImage.copyTo(tmp, trainImage);
		tmp.copyTo(finalImage);
		tmp.release();
		*/

		// Apply binary threshold, used for certain captchas and training data
		/*threshold(trainImage, trainImage, 120, 255, THRESH_BINARY);

		imshow("final Image", trainImage);
		waitKey(0);*/

		// Morphological closing - reduce noise in the letters
		Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
		dilate(trainImage, trainImage, element);
		erode(trainImage, trainImage, element);

		imshow("final Image", trainImage);
		waitKey(0);

		int* segH = horizontalSegments(trainImage);
		int* segV = verticalSegments(trainImage);

		Mat segHImage = drawHorizontalSegments(segH, trainImage.rows, trainImage.cols);
		Mat segVImage = drawVerticalSegments(segV, trainImage.rows, trainImage.cols);

		// Create pairs
		vector<pair<int, int> > verticalPairs = createSegmentPairs(segV, trainImage.rows);
		vector<pair<int, int> > horizontalPairs = createSegmentPairs(segH, trainImage.cols);

		// Get segment squares
		vector<Rectangle> squares = takeRectangles(shrinkRectangles(trainImage, getRectangles(verticalPairs, horizontalPairs)), 30);

		classify(sourceImage, trainingData, trainingClasses, squares);

		// Let's draw the rectangles
		drawRectangles(trainImage, squares);
		drawRectangles(sourceImage, squares);


		// Display the images if necessary

		/*imshow("Final image", trainImage);
		imshow("Source image", sourceImage);
		imshow("HSeg", segHImage);
		imshow("VSeg", segVImage);
		waitKey(0);
*/

		sourceImage.release();
		trainImage.release();
	}
	
	kNN->train(trainingData, ml::ROW_SAMPLE, trainingClasses);


	Mat testImage = imread("t_test(4).png");

	string output;

	cvtColor(testImage, testImage, COLOR_BGR2GRAY);

	// Is it loaded?
	if (!testImage.data)
		return -1;

	imshow("test Image", testImage);
	waitKey(0);

	// Resize the image by resize factor, don't need for our data set
	//resize(sourceImage, sourceImage, Size(sourceImage.cols * RESIZE_FACTOR, sourceImage.rows * RESIZE_FACTOR));

	// Define our final image
	Mat finalImage = testImage.clone();

	/*imshow("final Image", finalImage);
	waitKey(0);*/

	// Apply adaptive threshold
	//GaussianBlur(finalImage, finalImage, Size(3, 3), 0);
	adaptiveThreshold(finalImage, finalImage, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 11, 2);

	imshow("final Image", finalImage);
	waitKey(0);


	// Use the thresholded image as a mask, used for certain captchas
	/*Mat tmp;
	sourceImage.copyTo(tmp, trainImage);
	tmp.copyTo(finalImage);
	tmp.release();
	*/

	// Apply binary threshold, used for certain captchas and training data
	/*threshold(trainImage, trainImage, 120, 255, THRESH_BINARY);

	imshow("final Image", trainImage);
	waitKey(0);*/

	// Morphological closing - reduce noise in the letters
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	dilate(finalImage, finalImage, element);
	erode(finalImage, finalImage, element);

	imshow("final Image", finalImage);
	waitKey(0);

	int* segH = horizontalSegments(finalImage);
	int* segV = verticalSegments(finalImage);

	Mat segHImage = drawHorizontalSegments(segH, finalImage.rows, finalImage.cols);
	Mat segVImage = drawVerticalSegments(segV, finalImage.rows, finalImage.cols);

	// Create pairs
	vector<pair<int, int> > verticalPairs = createSegmentPairs(segV, finalImage.rows);
	vector<pair<int, int> > horizontalPairs = createSegmentPairs(segH, finalImage.cols);

	// Get segment squares
	vector<Rectangle> rects = takeRectangles(shrinkRectangles(finalImage, getRectangles(verticalPairs, horizontalPairs)), 90);

	// Let's draw the rectangles
	drawRectangles(finalImage, rects);
	drawRectangles(testImage, rects);


	// Display the images if necessary

	imshow("Final image", finalImage);
	imshow("Source image", testImage);
	imshow("HSeg", segHImage);
	imshow("VSeg", segVImage);
	waitKey(0);

	for (int i = 0; i < rects.size(); i++)
	{
		rectangle(testImage, Point(rects[i].x, rects[i].y), Point(rects[i].x + rects[i].width, rects[i].y + rects[i].height), cv::Scalar(0, 0, 255), 1);

		Mat ROI = testImage(Rect(rects[i].x, rects[i].y, rects[i].width, rects[i].height));
		Mat tmp = ROI.clone(); // temp of our region of interest, used for resizing and recognition

		resize(tmp, tmp, Size(32, 48));

		Mat matROIFloat;
		tmp.convertTo(matROIFloat, CV_32FC1);             // convert Mat to float, necessary for call to find_nearest

		Mat ROIFlattenedFloat = matROIFloat.reshape(1, 1);

		Mat currentChar(0, 0, CV_32F);

		kNN->findNearest(ROIFlattenedFloat, 1, currentChar);

		float fltCurrentChar = (float)currentChar.at<float>(0, 0);

		output = output + char(int(fltCurrentChar));        // append current char to full string
	}
	

	cout << output << endl;

	waitKey(0);

	return 0;
}
