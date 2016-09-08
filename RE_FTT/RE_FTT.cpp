// RE_FTT.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

using namespace std;
using namespace cv;

void load_calibrate_result(Mat &mtx1, Mat &mtx2, Mat &RT1, Mat &RT2, Mat &dist1, Mat &dist2);
void simpleBlob_para1_init(SimpleBlobDetector::Params& params);

int main(void) {
	
	VideoCapture cap1(1);
	VideoCapture cap2(0);

	if (!cap1.isOpened()&!cap2.isOpened()) return -1;

	Mat raw_src1, raw_src2, frame1, frame2, frame1_with_led;
	
	SimpleBlobDetector::Params lower_jaw_p;
	simpleBlob_para1_init(lower_jaw_p);

	Ptr<SimpleBlobDetector> lower_jaw_detector = SimpleBlobDetector::create(lower_jaw_p);
	vector<KeyPoint> four_leds;


	for (;;)
	{
		cap1 >> raw_src1;
		cap2 >> raw_src2;

		cvtColor(raw_src1, frame1, COLOR_RGB2GRAY);
		cvtColor(raw_src2, frame2, COLOR_RGB2GRAY);

		lower_jaw_detector->detect(frame1, four_leds);
		drawKeypoints(frame1, four_leds, frame1_with_led, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

		imshow("frame", frame1_with_led);
		imshow("frame2", frame2);
		if (waitKey(30) >= 0) break;
	}
	system("pause");

	return 0;
}

void load_calibrate_result(Mat &mtx1, Mat &mtx2, Mat &RT1, Mat &RT2, Mat &dist1, Mat &dist2) {

}
void simpleBlob_para1_init(SimpleBlobDetector::Params& params) {
	params.minThreshold = 10;
	params.maxThreshold = 200;
	params.filterByColor = true;
	params.blobColor = 255;
	params.filterByColor = true;
	params.filterByArea = true;
	params.minArea = 100;
	params.maxArea = 340;
	params.filterByCircularity = false;
	params.minCircularity = 0.5;
	params.filterByConvexity = false;
	params.minConvexity = 0.5;
	params.filterByInertia = false;
	params.minInertiaRatio = 0.5;
}
