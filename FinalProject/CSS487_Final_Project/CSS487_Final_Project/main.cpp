/* main.cpp
*
* Created by Kyle Farmer, Johan Brusa, and Tyler Hilde
*
* The purpose of this class is to handle the flow of the program.
* First, training images are loaded, manipulated, and classified
* into contours. Then, a test image is loaded, thresholded, eroded,
* and dilated. The program then uses k-nearest neighbors to match
* contours in the test image with the training data. Finally,
* the result is outputted to the console as a string.
*/

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <stdlib.h>

#include "Segments.h"
#include "Preprocess.h"

//const int RESIZE_FACTOR = 1.4;

int main(int argc, char* argv[]) {

	Ptr<ml::KNearest> kNN(ml::KNearest::create());
	Mat trainingClasses, trainingData; // float images used to train our k nearest object

   	for (int i = 0; i < 4; i++)
	{
		string imgName = "abc(" + to_string(i) + ").png";
		Mat sourceImage = imread(imgName);

		Mat trainImage = preprocessImage(sourceImage);

		imshow("Train Image thresholded", trainImage);
		waitKey(0);

		// segment our training image and return the vector of squares for each segement, 30 refers to max number of squares
		vector<Rectangle> squares = segmentation(trainImage, 30);

		classify(sourceImage, trainingData, trainingClasses, squares);

		sourceImage.release();
		trainImage.release();
	}
	
	// trainingData and classes has been constructed so we can call train on our kNN object
	kNN->train(trainingData, ml::ROW_SAMPLE, trainingClasses);


	Mat testImage = imread("t_test(4).png");

	imshow("test Image", testImage);
	waitKey(0);

	string output;

	Mat finalImage = preprocessImage(testImage);

	imshow("Final Image thresholded", finalImage);
	waitKey(0);

	// 90 hard coded for now, if we know the length of the string before hand we can determine a more realistic max for squares
	vector<Rectangle> rects = segmentation(finalImage, 90);

	// for each rectangle found
	for (int i = 0; i < rects.size(); i++)
	{
		// get the region of interest from our rectangle region in the test image, 
		// extend the width and height slightly to achieve better results
		Mat ROI = testImage(Rect(rects[i].x, rects[i].y, rects[i].width+4, rects[i].height+4));
		Mat tmp = ROI.clone(); // temp of our region of interest, used for resizing and recognition

		// resize for consistency in matching
		resize(tmp, tmp, Size(32, 48));

		// used for testing
		/*rectangle(testImage, Point(rects[i].x, rects[i].y), Point(rects[i].x + rects[i].width, rects[i].y + rects[i].height), cv::Scalar(0, 0, 255), 1);

		imshow("testImage", testImage);
		imshow("tmp", tmp);
		waitKey(0);*/

		Mat ROIFloat;
		tmp.convertTo(ROIFloat, CV_32FC1); // convert Mat to float, necessary for call to findNearest

		Mat ROIFlattenedFloat = ROIFloat.reshape(1, 1);

		Mat currentChar(0, 0, CV_32F);

		kNN->findNearest(ROIFlattenedFloat, 1, currentChar); // k = 1

		float fltCurrentChar = (float)currentChar.at<float>(0, 0);

		output = output + char(int(fltCurrentChar)); // append current char to full string
	}
	
	cout << output << endl;

	waitKey(0);

	return 0;
}
