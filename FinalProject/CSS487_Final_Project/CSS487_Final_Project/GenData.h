#ifndef GENDATA_H
#define GENDATA_H


#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>

#include<iostream>
#include<vector>

using namespace std;
using namespace cv;

// global variables 
const int MIN_CONTOUR_AREA = 80;

const int RESIZED_IMAGE_WIDTH = 20;
const int RESIZED_IMAGE_HEIGHT = 30;

vector<vector<Point> > getContours(const Mat& img, Mat& thresh);
Mat classify(Mat& classificationInts, Mat& img, const Mat& img_thresh, const vector<vector<Point> > &ptContours);

#endif
