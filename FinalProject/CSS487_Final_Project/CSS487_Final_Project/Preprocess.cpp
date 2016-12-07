#include "Preprocess.h"

// Preconditions: sourceImage is a raw input image that is read into the system by imread
// Postconditions: sourceImage is converted to graycolor. Returns a new 
//                 image (starts as a copy of source) that has been thresholded,
//				   dilated, and eroded.				   
Mat preprocessImage(Mat& sourceImage)
{
	cvtColor(sourceImage, sourceImage, COLOR_BGR2GRAY); // convert to grayscale

	// Resize the image by a resize factor, don't need for our data set
	//resize(sourceImage, sourceImage, Size(sourceImage.cols * 1.4, sourceImage.rows * 1.4));
	// Define our output image, copy source
	Mat outImage = sourceImage.clone();

	// Apply adaptive threshold
	adaptiveThreshold(outImage, outImage, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 11, 2);

	// Morphological closing - reduce noise in the letters, allows for better segmentation
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	dilate(outImage, outImage, element);
	erode(outImage, outImage, element);

	return outImage; // return final image (thresholded, dilated, eroded)
}