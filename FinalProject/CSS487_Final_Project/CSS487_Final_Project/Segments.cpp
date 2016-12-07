#include "Segments.h"


bool compareSquaresBySize(Rectangle a, Rectangle b) 
{
	return a.width * a.height > b.width * b.height;
}

bool compareSquaresById(Rectangle a, Rectangle b) 
{
	return a.id < b.id;
}

// Preconditions: Takes an initilizaed array and the length
// Postconditions: Returns an initialized array, all indices set to 0
void initArray(int* a, int length)
{
	for (int i = 0; i < length; i++)
		a[i] = 0;
}

// Preconditions: Grayscale image of the text to be decoded
// Postconditions: Segment array for the columns in the image.
// For each column in the image, add 1 to the segment array everytime a pixel is encountered that is not 0. 
// Each pixel is 0 to 255, so >0 indicates text or a line, anything above indicates space. 
// This segment array will tell us which columns in the image have letters by
// having values in their respective columns.
int* horizontalSegments(Mat& src) 
{
	int* seg = new int[src.cols]; // make an integer array sized the image columns
	initArray(seg, src.cols); // initialize array

	for (int i = 0; i < src.cols; i++) {
 		for (int k = 0; k < src.rows; k++) 
		{
			// extract pixels by treating the data array like a 1d array 
			uchar pixel = src.data[src.cols * k + i]; 
			// if pixel is above 0, add 1 to the array index
			if (pixel)
				seg[i] += 1;
 		}
	}

	return seg;
}

// Preconditions: Takes the segment array, and the amount of rows and columns in the image
// Postconditions: Creates an 8 bit segment image that shows the amount of white pixels for each column
Mat drawHorizontalSegments(int* seg, int rows, int cols) 
{
	Mat segImage = Mat(rows, cols, CV_8U, Scalar(0));

	for (int i = 0; i < cols; i++) {
		if (seg[i] > 0)
			line(segImage, Point(i, rows - 1), Point(i, rows - 1 - seg[i]), Scalar(255), 1);
	}

	return segImage;
}


// Preconditions: Grayscale image of the text to be decoded.
// Postconditions: Segment array for the rows in the image.
// Same logic as horizontal segments but for the rows in the image.
int* verticalSegments(Mat& src) 
{
	int* seg = new int[src.rows];
	initArray(seg, src.rows);

	for (int i = 0; i < src.rows; i++) {
		for (int k = 0; k < src.cols; k++) 
		{
			uchar pixel = src.data[src.cols * i + k];

			if (pixel)
				seg[i]++;
		}
	}

	return seg;
}

/// Preconditions: Takes the segment array, and the amount of rows and columns in the image
// Postconditions: Creates an 8 bit segment image that shows the amount of white pixels for each row
Mat drawVerticalSegments(int* seg, int rows, int cols) 
{
	Mat segImage = Mat(rows, cols, CV_8U, Scalar(0));

	for (int i = 0; i < rows; i++) {
		if (seg[i] > 0)
			line(segImage, Point(0, i), Point(seg[i], i), Scalar(255), 1);
	}

	return segImage;
}

// Preconditions: Segment array and the length of the direction of the segment array, horizontal - cols, vertical - rows
// Postconditions: Returns a vector of pairs in the horizontal or vertical direction. Pairs are found when the flag is set to false
//		and the top and bottom integers are greater than 0.
vector<pair<int, int> > createSegmentPairs(int* seg, int segSize)
{
	int top = 0, bottom = 0; // top is the first integer in the pair, bottom is the last integer of the pair
	bool flag = false; // used for finding the zeros in the segment array

	vector<pair<int, int> > pairs;

	// iterate through the length of the segment array
	for (int i = 0; i < segSize; i++)
	{
		// if the segment array has a value greater than 0, set the flag
		if (seg[i])
			flag = true;
		// else set the flag to zero and check if there is a pair present
		else
			flag = false;

		if (flag)
		{
			// check if the top has been set
			if (!top)
			{
				// if not, set both top and bottom to the current index
				top = i;
				bottom = i;
			}
			else
			{
				// top is set so keep incrementing the bottom pair
				bottom = i;
			}

			//corner case
			if (i == segSize - 1)
			{
				pairs.push_back(make_pair(top, bottom));
				top = bottom = 0;
			}
		}
		else
		{
			// if the top and bottom are set and the flag is not
			if (top && bottom)
			{
				// found a pair, push back to the vector
				pairs.push_back(make_pair(top, bottom));
				// reset top and bottom to 0
				top = bottom = 0;
			}
		}

	}
	
	return pairs;
}

// Preconditions: Takes the vertical and horizontal pairs after creating the segments
// Postconditions: Returns a vector of rectangles that will bound each character found in the image based on the segment arrays.
// These rectangles will initially be too large to give good results, but shrinkRectangles will give better results for the bounding
// rectangles for each character.
vector<Rectangle> getRectangles(vector<pair<int, int> > verticalPairs, vector<pair<int, int> > horizontalPairs) {
	vector<Rectangle> squares;
	int id = 0;

	// iterate through the horizontal and vertical pairs
	for (vector<pair<int, int> >::iterator itV = verticalPairs.begin(); itV != verticalPairs.end(); itV++) {
		for (vector<pair<int, int> >::iterator itH = horizontalPairs.begin(); itH != horizontalPairs.end(); itH++) {
			// use the vertical and horizontal pairs to make rectangles
			Rectangle r = { id++, itH->first, itV->first, itH->second - itH->first, itV->second - itV->first };
			// push back to the vector of squares
			squares.push_back(r);
		}
	}

	return squares;
}

// Preconditions: Takes the image and the vector of rectangles returned by getRectangles
// Postconditions: Returns the vector of new squares that has the rectangles fitted to characters found
vector<Rectangle> shrinkRectangles(Mat& image, vector<Rectangle> squares) {
	vector<Rectangle> new_squares;

	// for each the rectangles found
	for (int i = 0; i < squares.size(); i++) {
		int top = -1, bottom = 0, left = 9999, right = -1; // set initial values to extremes
		// create a tmp image of the rectangle from the vector
		Mat tmp2 = image(Rect(squares[i].x, squares[i].y, squares[i].width, squares[i].height));
		Mat tmp = tmp2.clone();

		for (int y = 0; y < tmp.rows; y++) {
			for (int x = 0; x < tmp.cols; x++) {
				int pixel = tmp.data[x + y * tmp.cols];
				// if a pixel value above 0 is found, we know this is a spot where
				if (pixel) {
 					// set data to max uchar value
					tmp.data[x + y * tmp.cols] = 255;
					if (top == -1) 
						top = y; // store the first value it comes across
					if (left > x) // if left is greater than the current column
						left = x; // set left to the current column
					bottom = y; // y will be always incremented, just store the highest value
					if (right < x) // if right is less than the current column
						right = x; // set right to the current column
				}
			}
		}
		// adjust each out a pixel so there isnt overlap on the character
		top -= 1;
		left -= 1;
		bottom += 1;
		right += 1;

		// create the rectangle and push back
		Rectangle r = { squares[i].id, squares[i].x + left, squares[i].y + top, right - left, bottom - top };
		new_squares.push_back(r);
		tmp.release();
		tmp2.release();
	}

	return new_squares;
}

// Preconditions: Takes the rectangles after shrinking, and the number of squares specified
// Postconditions: Returns a vector of squares sorted by the id in which they were created - should be in order from left to right
vector<Rectangle> takeRectangles(vector<Rectangle> squares, int number) {
	vector<Rectangle> new_squares;

	// sort the squares by size
	sort(squares.begin(), squares.end(), compareSquaresBySize);
	// min between the size of the vector or the number of rectangles specified for segmentation
	// we pass a high number in for number since in most cases we want the most amount of rectangles found
	int min = MIN(number, (int)squares.size());

	// add the rectangles in order of size to the new vector
	for (int i = 0; i < min; i++) {
		new_squares.push_back(squares[i]);
	}

	// resort the squares by the order in which they were created from the left to right
	sort(new_squares.begin(), new_squares.end(), compareSquaresById);

	return new_squares;
}

// Preconditions: Takes the source image of training, the two 
// Postconditions: Returns the image with rectangles drawn on it based on the rectangles in the vector r.
// trainingClasses has all of the ASCII values found pushed into a single row. trainingData has the flattened float
// images of the regions of interest found from the rectangles pushed back onto its image.
void classify(Mat &image, Mat& trainingData, Mat& trainingClasses, vector<Rectangle> r)
 {
	// for each rectangle in the vector
	for (int i = 0; i < r.size(); i++)
	{
		// make a temp image of the area inside the rectangle from the vector
		// extend height and width slightly for better results
		Mat tmp = image(Rect(r[i].x, r[i].y, r[i].width + 4, r[i].height + 4));
		Mat tmp2 = tmp.clone();

		// set a small bound on the limit of columns and rows so there arent unnecessary rectangles
		if (tmp2.cols > 4 && tmp2.rows > 4)
		{
			// draw the rectangle on the passed in image
			rectangle(image, Point(r[i].x, r[i].y), Point(r[i].x + r[i].width, r[i].y + r[i].height), cv::Scalar(0, 0, 255), 1);

			// resize the extracted rectangle for more consistent matching
			resize(tmp2, tmp2, Size(32, 48));

			//imshow("TrainingROI", tmp2);
			//imshow("TrainingImg", image);
			//waitKey(0);

			// i+97 for starting at lowercase 'a' ASCII value
			trainingClasses.push_back(i + 97);       // add char to our floating point labels image

			Mat floatImg;
			// convert the training region of interest to a float
			tmp2.convertTo(floatImg, CV_32FC1);

			// flatten
			Mat flattenedFloatImg = floatImg.reshape(1, 1);

			trainingData.push_back(flattenedFloatImg);
		}

		tmp.release();
		tmp2.release();
	}
}

// Preconditions: Takes the image the vector of rectangles
// Postconditions: Returns the image with the rectangles drawn around each letter found
void drawRectangles(Mat& im, vector<Rectangle> r)
{
	for (int i = 0; i < r.size(); i++)
		rectangle(im, Point(r[i].x, r[i].y), Point(r[i].x + r[i].width, r[i].y + r[i].height), Scalar(255));
}

// Preconditions: Takes the image and the number of specified squares to be drawn in the image
// Postconditions: Returns the vector of rectangles bounding each character in the image
vector<Rectangle> segmentation(Mat& img, int numSquares)
{
	int* segH = horizontalSegments(img);
	int* segV = verticalSegments(img);

	Mat segHImage = drawHorizontalSegments(segH, img.rows, img.cols);
	Mat segVImage = drawVerticalSegments(segV, img.rows, img.cols);

	// create pairs
	vector<pair<int, int> > verticalPairs = createSegmentPairs(segV, img.rows);
	vector<pair<int, int> > horizontalPairs = createSegmentPairs(segH, img.cols);

	// get segment rectangles
	vector<Rectangle> rects = takeRectangles(shrinkRectangles(img, getRectangles(verticalPairs, horizontalPairs)), numSquares);

	// draw the rectangles for output
	//drawRectangles(img, rects);
	//// Display the images if necessary
	//imshow("Image rectangles", img);
	//imshow("Horizontal Segments", segHImage);
	//imshow("Vertical Segments", segVImage);
	//waitKey(0);
	

	return rects;
}