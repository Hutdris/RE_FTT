// RE_FTT.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

using namespace std;
using namespace cv;

void load_calibrate_result(Mat &mtx1, Mat &mtx2, Mat &RT1, Mat &RT2, Mat &dist1, Mat &dist2, Mat &fund_mat);
void simpleBlob_para1_init(SimpleBlobDetector::Params& params);
bool pt_compare_by_x(Point2d p1, Point2d p2) {
	return p1.x < p2.x;
}
int main(void) {
	
	VideoCapture cap1(1);
	VideoCapture cap2(0);

	if (!cap1.isOpened()&!cap2.isOpened()) return -1;
//Initalization
	Mat dist1, dist2, mtx1, mtx2, RT1, RT2, fund_mat;
	load_calibrate_result(mtx1, mtx2, RT1, RT2, dist1, dist2, fund_mat);
	Mat project1 = mtx1*RT1;
	Mat project2 = mtx2*RT2;
	Mat raw_src1, raw_src2, frame1, frame2, frame1_with_led, frame2_with_led;
	
	SimpleBlobDetector::Params lower_jaw_p;
	simpleBlob_para1_init(lower_jaw_p);

	Ptr<SimpleBlobDetector> lower_jaw_detector = SimpleBlobDetector::create(lower_jaw_p);
	vector<KeyPoint> four_leds1, four_leds2;
	Mat fourD_points;
	for (;;)
	{
		cap1 >> raw_src1;
		cap2 >> raw_src2;

		cvtColor(raw_src1, frame1, COLOR_RGB2GRAY);
		cvtColor(raw_src2, frame2, COLOR_RGB2GRAY);

		lower_jaw_detector->detect(frame1, four_leds1);
		lower_jaw_detector->detect(frame2, four_leds2);
		if ((four_leds1.size() == four_leds2.size())&(four_leds1.size() >= 4)) {
			//traiangluate eat '2*N' array not '1*N' points vector
			//Tranlate keypoints to sorted vector<Point2d>
			int N = int(four_leds1.size());
			vector<Point2d> f_leds1, f_leds2;
			for (int i = 0; i < N; i++) {
				f_leds1.push_back(four_leds1[i].pt);
				f_leds2.push_back(four_leds2[i].pt);
			}
			sort(f_leds1.begin(), f_leds1.end(), pt_compare_by_x);
			sort(f_leds2.begin(), f_leds2.end(), pt_compare_by_x);
//			Mat f_leds1 = Mat(2, N, CV_64F, &four_leds1); //converting vector to array trick
//			Mat f_leds2 = Mat(2, N, CV_64F, &four_leds2);
			fourD_points = Mat(4, N, CV_64F);
			triangulatePoints(project1, project2, f_leds1, f_leds2, fourD_points);
			cout << fourD_points << endl;
		}
		else cout << "Fail" << endl;
		drawKeypoints(raw_src1, four_leds1, frame1_with_led, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		drawKeypoints(raw_src2, four_leds2, frame2_with_led, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);


		imshow("frame1", frame1_with_led);
		imshow("frame2", frame2_with_led);
		if (waitKey(30) >= 0) break;
	}
	cap1.release();
	cap2.release();
//	system("pause");

	return 0;
}

void simpleBlob_para1_init(SimpleBlobDetector::Params& params) {
	params.minThreshold = 110;
	params.maxThreshold = 400;
	params.filterByColor = false;
	params.blobColor = 200;
	params.filterByArea = true;
	params.minArea = 100;
	params.maxArea = 340;
	params.filterByCircularity = true;
	params.minCircularity = 0.5;
	params.filterByConvexity = true;
	params.minConvexity = 0.5;
	params.filterByInertia = true;
	params.minInertiaRatio = 0.5;
}

void load_calibrate_result(Mat &mtx1, Mat &mtx2, Mat &RT1, Mat &RT2, Mat &dist1, Mat &dist2, Mat &fund_mat) {
	double m1[3][3] = { {2441.347367, 0.000000, 661.608922}
		, {0.000000, 2437.002900, 509.925022}
		, {0, 0, 1} };
	double m2[3][3] = {
	{ 2412.136458, 0.000000, 742.768353 },
	{ 0.000000, 2409.743589, 569.840267 },
	{ 0, 0, 1 }
	};
	double d1[5] = { -0.418326, 12.064726, 0.006454, 0.006439, 0.0 };
	double d2[5] = { 0.013682, -4.051145, 0.005566, 0.004979, 0.0 };
	double rt1[3][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}};
	double rt2[3][4] = { {0.695238, -0.000966, 0.718779, -236.973392},
		{0.012590, 0.999862, -0.010835, -2.722954},
		{-0.718670, 0.016582, 0.695154, 99.387476} };
	dist1 = Mat(1, 5, CV_64F, d1);
	dist2 = Mat(1, 5, CV_64F, d2);
	RT1 = Mat(3, 4, CV_64F, rt1);
	RT2 = Mat(3, 4, CV_64F, rt2);
	mtx1 = Mat(3, 3, CV_64F, m1);
	mtx2 = Mat(3, 3, CV_64F, m2);
}
