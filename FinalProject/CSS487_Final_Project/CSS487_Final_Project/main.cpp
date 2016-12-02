#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <stdlib.h>

#include "GenData.h"


class ContourWithData {
	public:
		// member variables 
		std::vector<cv::Point> ptContour;           // contour
		cv::Rect boundingRect;                      // bounding rect for contour
		float fltArea;                              // area of contour

		// probably not the best function for testing if a contour is valid											
		bool checkIfContourIsValid() {                              
			if (fltArea < MIN_CONTOUR_AREA) return false;           
			return true;                                            
		}

		// allows sorting of contours from left to right
		static bool sortByBoundingRectXPosition(const ContourWithData& cwdLeft, const ContourWithData& cwdRight) {    
			return(cwdLeft.boundingRect.x < cwdRight.boundingRect.x);                                                   
		}

};

int main(int argc, char* argv[]) {

	Mat train = imread("handwriting.png"), train_thresh, test = imread("test.png"), test_thresh;
	Mat classificationInts;

	vector<vector<Point>> ptContours = getContours(train, train_thresh); // contours returned from getContours

	Mat trainingImagesAsFlattenedFloats = classify(classificationInts, train, train_thresh, ptContours);

	std::cout << "training complete\n\n";

	// make kNN object
	Ptr<ml::KNearest>  kNearest(ml::KNearest::create());
	kNearest->train(trainingImagesAsFlattenedFloats, ml::ROW_SAMPLE, classificationInts);

	vector<ContourWithData> allContoursWithData;           // declare empty contour vectors
	vector<ContourWithData> validContoursWithData;

	ptContours = getContours(test, test_thresh);

	// for every contour
	for (int i = 0; i < ptContours.size(); i++) {               
		ContourWithData contourWithData; 

		// assign contour to contour with data
		contourWithData.ptContour = ptContours[i];  

		// get bounding rectangle
		contourWithData.boundingRect = boundingRect(contourWithData.ptContour); 

		// calculate the contour area
		contourWithData.fltArea = (float)contourArea(contourWithData.ptContour);

		// add contour with data object to list of all contours with data
		allContoursWithData.push_back(contourWithData);                                     
	}

	for (int i = 0; i < allContoursWithData.size(); i++) {                      // for all contours
		if (allContoursWithData[i].checkIfContourIsValid()) {                   // check if valid
			validContoursWithData.push_back(allContoursWithData[i]);            // if so, append to valid contour list
		}
	}
	// sort contours from left to right
	std::sort(validContoursWithData.begin(), validContoursWithData.end(), ContourWithData::sortByBoundingRectXPosition);

	string strFinalString;     // final string for console output

	for (int i = 0; i < validContoursWithData.size(); i++) {            

		// for each contour draw a green rect around the current char															
		rectangle(test, validContoursWithData[i].boundingRect, Scalar(0, 255, 0), 2);                                           

		Mat matROI = test_thresh(validContoursWithData[i].boundingRect);          // get ROI image of bounding rect

		Mat matROIResized;
		// resize image, this will be more consistent for recognition and storage
		resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));

		Mat matROIFloat;
		matROIResized.convertTo(matROIFloat, CV_32FC1);             // convert Mat to float, necessary for call to find_nearest

		Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1);

		Mat matCurrentChar(0, 0, CV_32F);

		kNearest->findNearest(matROIFlattenedFloat, 1, matCurrentChar);     

		float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);

		strFinalString = strFinalString + char(int(fltCurrentChar));        // append current char to full string
	}

	std::cout << "\n\n" << "numbers read = " << strFinalString << "\n\n";       // show the full string

	imshow("test", test);     // show input image with green boxes drawn around found digits

	waitKey(0);                                         

	return 0;
}
