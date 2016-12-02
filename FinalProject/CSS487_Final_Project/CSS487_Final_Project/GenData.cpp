#include "GenData.h"


vector<vector<Point> > getContours(const Mat& img, Mat& thresh)
{
	Mat gray_img, blur_img, threshCopy;
	vector<Vec4i> v4iHierarchy;                    // declare contours hierarchy
	vector<vector<Point>> ptContours;

	cvtColor(img, gray_img, CV_BGR2GRAY);        // convert to grayscale
	GaussianBlur(gray_img, blur_img, Size(5, 5), 0); // blur
	adaptiveThreshold(blur_img, thresh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 11, 2); // apply threshold

	imshow("imgThresh", thresh);         // show threshold image for reference

	threshCopy = thresh.clone();          // make a copy of the thresh image, this is necessary b/c findContours modifies the image

	findContours(threshCopy, ptContours, v4iHierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE); 

	return ptContours;
}

Mat classify(Mat& classificationInts, Mat& img, const Mat& img_thresh, const vector<vector<Point> > &ptContours)
{
	Mat matTrainingImagesAsFlattenedFloats;
	
	// possible chars interested in are digits 0 through 9 and capital letters A through Z for now
	vector<int> intValidChars = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
		'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
		'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
		'U', 'V', 'W', 'X', 'Y', 'Z' };

	for (int i = 0; i < ptContours.size(); i++) {                           // for each contour
		if (contourArea(ptContours[i]) > MIN_CONTOUR_AREA) {                // if contour is big enough to consider
			Rect boundingRect = cv::boundingRect(ptContours[i]);                // get the bounding rect

			rectangle(img, boundingRect, cv::Scalar(0, 0, 255), 2);      // draw red rectangle around each contour as we ask user for input

			Mat matROI = img_thresh(boundingRect);           // get ROI image of bounding rect

			Mat matROIResized;
			resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));     // resize image, this will be more consistent for recognition and storage

			imshow("matROI", matROI);                               // show ROI image for reference
			imshow("matROIResized", matROIResized);                 // show resized ROI image for reference
			imshow("imgTrainingNumbers", img);       // show training numbers image, this will now have red rectangles drawn on it

			int intChar = waitKey(0);           // get key press

			if (std::find(intValidChars.begin(), intValidChars.end(), intChar) != intValidChars.end()) {     // else if the char is in the list of chars we are looking for . . .

				classificationInts.push_back(intChar);       // append classification char to integer list of chars

				Mat matImageFloat;                          // now add the training image (some conversion is necessary first) . . .
				matROIResized.convertTo(matImageFloat, CV_32FC1);       // convert Mat to float

				Mat matImageFlattenedFloat = matImageFloat.reshape(1, 1);       // flatten

				matTrainingImagesAsFlattenedFloats.push_back(matImageFlattenedFloat);       // add to Mat as though it was a vector, this is necessary due to the
																							// data types that KNearest.train accepts
			}
		}
	}

	return matTrainingImagesAsFlattenedFloats;
}