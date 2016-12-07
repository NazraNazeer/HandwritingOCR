#include "Segments.h"


bool compareSquaresBySize(Rectangle a, Rectangle b) 
{
	return a.width * a.height > b.width * b.height;
}

bool compareSquaresById(Rectangle a, Rectangle b) 
{
	return a.id < b.id;
}

void initArray(int* a, int length)
{
	for (int i = 0; i < length; i++)
		a[i] = 0;
}

// Preconditions: Grayscale image of the text to be decoded
// Postconditions: Segment array
// For each column in the image, add 1 to the segment array everytime a pixel is encountered that is not 0. 
// Each pixel is 0 to 255, so 0 indicates text or a line, anything above indicates space. 
// This segment array will tell us which columns in the image have letters by
// having a lower resulting sum than the other columns.
int* horizontalSegments(Mat& src) 
{
	int* seg = new int[src.cols];
	initArray(seg, src.cols);

	for (int i = 0; i < src.cols; i++) {
 		for (int k = 0; k < src.rows; k++) 
		{
			// extract pixels by treating the data array like a 1d array 
			uchar pixel = src.data[src.cols * k + i]; 
			// if pixel is above 0, add 1 to the bin
			if (pixel)
				seg[i] += 1;
 		}
	}

	return seg;
}

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
// Postconditions: Segment array for each row in the image indicating the columns
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

Mat drawVerticalSegments(int* seg, int rows, int cols) 
{
	Mat segImage = Mat(rows, cols, CV_8U, Scalar(0));

	for (int i = 0; i < rows; i++) {
		if (seg[i] > 0)
			line(segImage, Point(0, i), Point(seg[i], i), Scalar(255), 1);
	}

	return segImage;
}

vector<pair<int, int> > createSegmentPairs(int* seg, int segSize)
{
	int top = 0, bottom = 0;
	bool in = false;

	vector<pair<int, int> > pairs;

	for (int i = 0; i < segSize; i++)
	{
		if (seg[i] )
			in = true;
		else
			in = false;

		if (in)
		{
			if (!top)
			{
				top = i;
				bottom = i;
			}
			else
			{
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
			if (top && bottom)
			{
				pairs.push_back(make_pair(top, bottom));
				top = bottom = 0;
			}
		}

	}
	
	return pairs;
}


vector<Rectangle> getRectangles(vector<pair<int, int> > verticalPairs, vector<pair<int, int> > horizontalPairs) {
	vector<Rectangle> squares;
	int id = 0;

	for (vector<pair<int, int> >::iterator itV = verticalPairs.begin(); itV != verticalPairs.end(); itV++) {
		for (vector<pair<int, int> >::iterator itH = horizontalPairs.begin(); itH != horizontalPairs.end(); itH++) {
			Rectangle r = { id++, itH->first, itV->first, itH->second - itH->first, itV->second - itV->first };
			squares.push_back(r);
		}
	}

	return squares;
}

vector<Rectangle> shrinkRectangles(Mat& image, vector<Rectangle> squares) {
	vector<Rectangle> new_squares;

	for (int i = 0; i < squares.size(); i++) {
		int top = -1, bottom = 0, left = 9999, right = -1;
		Mat tmp2 = image(Rect(squares[i].x, squares[i].y, squares[i].width, squares[i].height));
		Mat tmp = tmp2.clone();

		for (int y = 0; y < tmp.rows; y++) {
			for (int x = 0; x < tmp.cols; x++) {
				int pixel = tmp.data[x + y * tmp.cols];
				
				if (pixel) {
					tmp.data[x + y * tmp.cols] = 127;
					if (top == -1) 
						top = y; // store the lowest value
					if (left > x) 
						left = x;
					bottom = y; // y will be always incremented, just store the highest value
					if (right < x)
						right = x;
				}
			}
		}

		top -= 1;
		left -= 1;
		bottom += 1;
		right += 1;

		Rectangle r = { squares[i].id, squares[i].x + left, squares[i].y + top, right - left, bottom - top };
		new_squares.push_back(r);
		tmp.release();
		tmp2.release();
	}

	return new_squares;
}

vector<Rectangle> takeRectangles(vector<Rectangle> squares, int number) {
	vector<Rectangle> new_squares;

	sort(squares.begin(), squares.end(), compareSquaresBySize);
	int min = MIN(number, (int)squares.size());

	for (int i = 0; i < min; i++) {
		new_squares.push_back(squares[i]);
	}

	sort(new_squares.begin(), new_squares.end(), compareSquaresById);

	return new_squares;
}

void classify(Mat &image, Mat& trainingData, Mat& trainingClasses, vector<Rectangle> r)
{
	for (int i = 0; i < r.size(); i++)
	{
		Mat tmp = image(Rect(r[i].x, r[i].y, r[i].width + 4, r[i].height + 4));
		Mat tmp2 = tmp.clone();
		if (tmp2.cols > 3 && tmp2.rows > 5)
		{
			rectangle(image, Point(r[i].x, r[i].y), Point(r[i].x + r[i].width, r[i].y + r[i].height), cv::Scalar(0, 0, 255), 1);

			resize(tmp2, tmp2, Size(32, 48));

			//imshow("TrainingROI", tmp2);
			////imshow("TrainingImg", image);
			//waitKey(0);

			// i+97 for starting at lowercase 'a' ASCII value
			trainingClasses.push_back(i + 97);       // add char to our floating point labels image

			Mat matImageFloat;
			// convert the training region of interest to a float
			tmp2.convertTo(matImageFloat, CV_32FC1);

			// flatten
			Mat matImageFlattenedFloat = matImageFloat.reshape(1, 1);

			trainingData.push_back(matImageFlattenedFloat);
		}

		tmp.release();
		tmp2.release();
	}
}

void drawRectangles(Mat& im, vector<Rectangle> r)
{
	for (int i = 0; i < r.size(); i++)
		rectangle(im, Point(r[i].x, r[i].y), Point(r[i].x + r[i].width, r[i].y + r[i].height), Scalar(255));
}

vector<Rectangle> segmentation(Mat& img, int numSquares)
{
	int* segH = horizontalSegments(img);
	int* segV = verticalSegments(img);

	Mat segHImage = drawHorizontalSegments(segH, img.rows, img.cols);
	Mat segVImage = drawVerticalSegments(segV, img.rows, img.cols);

	// Create pairs
	vector<pair<int, int> > verticalPairs = createSegmentPairs(segV, img.rows);
	vector<pair<int, int> > horizontalPairs = createSegmentPairs(segH, img.cols);

	// Get segment squares
	vector<Rectangle> rects = takeRectangles(shrinkRectangles(img, getRectangles(verticalPairs, horizontalPairs)), numSquares);

	// Display the images if necessare
	/*imshow("Image rectangles", img);
	imshow("Horizontal Segments", segHImage);
	imshow("Vertical Segments", segVImage);
	waitKey(0);
*/
	return rects;
}