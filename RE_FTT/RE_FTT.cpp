// RE_FTT.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

using namespace std;
using namespace cv;

void stereo_cam_calibrate();
void load_calibrate_result(Mat &mtx1, Mat &mtx2, Mat &RT1, Mat &RT2, Mat &dist1, Mat &dist2, Mat &fund_mat);
void chess_calibrate(VideoCapture &capture1, VideoCapture &capture2, Mat &mtx1, Mat &mtx2, Mat &RT1, Mat &RT2, Mat &dist1, Mat &dist2, Mat &fund_mat);

void simpleBlob_para1_init(SimpleBlobDetector::Params& params);
inline bool pt_compare_by_x(Point2d p1, Point2d p2) { return p1.x < p2.x;}
int main(void) {
//Calibrate Start

	int numBoards = 3;
	int board_w = 10;
	int board_h = 7;

	Size board_sz = Size(board_w, board_h);
	int board_n = board_w*board_h;

	vector<vector<Point3f> > object_points;
	vector<vector<Point2f> > imagePoints1, imagePoints2;
	vector<Point2f> corners1, corners2;

	vector<Point3f> obj;
	for (int j = 0; j<board_n; j++)
	{
		obj.push_back(Point3f(j / board_w, j%board_w, 0.0f));
	}

	Mat img1, img2, gray1, gray2;

	int success = 0, k = 0;
	bool found1 = false, found2 = false;

	VideoCapture cap1 = VideoCapture(0);
	VideoCapture cap2 = VideoCapture(1);
	while (success < numBoards)
	{
		cap1 >> img1;
		cap2 >> img2;
		//resize(img1, img1, Size(320, 280));
		//resize(img2, img2, Size(320, 280));
		cvtColor(img1, gray1, CV_BGR2GRAY);
		cvtColor(img2, gray2, CV_BGR2GRAY);

		found1 = findChessboardCorners(img1, board_sz, corners1, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
		found2 = findChessboardCorners(img2, board_sz, corners2, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

		if (found1)
		{
			cornerSubPix(gray1, corners1, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
			drawChessboardCorners(gray1, board_sz, corners1, found1);
		}

		if (found2)
		{
			cornerSubPix(gray2, corners2, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
			drawChessboardCorners(gray2, board_sz, corners2, found2);
		}

		imshow("image1", gray1);
		imshow("image2", gray2);

		k = waitKey(10);
		if (found1 && found2)
		{
			k = waitKey(0);
		}
		if (k == 27)
		{
			break;
		}
		if (k == ' ' && found1 != 0 && found2 != 0)
		{
			imagePoints1.push_back(corners1);
			imagePoints2.push_back(corners2);
			object_points.push_back(obj);
			printf("Corners stored\n");
			success++;

			if (success >= numBoards)
			{
				break;
			}
		}
	}

	destroyAllWindows();
	printf("Starting Calibration\n");
	Mat CM1 = Mat(3, 3, CV_64FC1);
	Mat CM2 = Mat(3, 3, CV_64FC1);
	Mat D1, D2;
	Mat R, T, E, F;
	stereoCalibrate(object_points, imagePoints1, imagePoints2,
		CM1, D1, CM2, D2, img1.size(), R, T, E, F
	);
	/*
		cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 100, 1e-5),
		CV_CALIB_SAME_FOCAL_LENGTH | CV_CALIB_ZERO_TANGENT_DIST
	*/
	FileStorage fs1("mystereocalib.yml", FileStorage::WRITE);
	fs1 << "CM1" << CM1;
	fs1 << "CM2" << CM2;
	fs1 << "D1" << D1;
	fs1 << "D2" << D2;
	fs1 << "R" << R;
	fs1 << "T" << T;
	fs1 << "E" << E;
	fs1 << "F" << F;

	printf("Done Calibration\n");

	printf("Starting Rectification\n");

	Mat R1, R2, P1, P2, Q;
	stereoRectify(CM1, D1, CM2, D2, img1.size(), R, T, R1, R2, P1, P2, Q);
	fs1 << "R1" << R1;
	fs1 << "R2" << R2;
	fs1 << "P1" << P1;
	fs1 << "P2" << P2;
	fs1 << "Q" << Q;

	printf("Done Rectification\n");

	printf("Applying Undistort\n");

	Mat map1x, map1y, map2x, map2y;
	Mat imgU1, imgU2;

	initUndistortRectifyMap(CM1, D1, R1, P1, img1.size(), CV_32FC1, map1x, map1y);
	initUndistortRectifyMap(CM2, D2, R2, P2, img2.size(), CV_32FC1, map2x, map2y);

	printf("Undistort complete\n");

	while (1)
	{
		cap1 >> img1;
		cap2 >> img2;

		remap(img1, imgU1, map1x, map1y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
		remap(img2, imgU2, map2x, map2y, INTER_LINEAR, BORDER_CONSTANT, Scalar());

		imshow("image1", imgU1);
		imshow("image2", imgU2);

		k = waitKey(5);

		if (k == 27)
		{
			break;
		}
	}

//	cap1.release();
//	cap2.release();

	//Calibrate Done


	
//	VideoCapture cap1 = VideoCapture(0);
//	VideoCapture cap2 = VideoCapture(1);

	if (!cap1.isOpened()&!cap2.isOpened()) return -1;
//Initalization
	Mat dist1, dist2, mtx1, mtx2, RT1, RT2, fund_mat;
	load_calibrate_result(mtx1, mtx2, RT1, RT2, dist1, dist2, fund_mat);
	chess_calibrate(cap1, cap2, mtx1, mtx2, RT1, RT2, dist1, dist2, fund_mat);
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
		CvSize chessboard_size = cvSize(3, 3);
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

void stereo_cam_calibrate()
{
	int numBoards = 20;
	int board_w = 10;
	int board_h = 7;

	Size board_sz = Size(board_w, board_h);
	int board_n = board_w*board_h;

	vector<vector<Point3f> > object_points;
	vector<vector<Point2f> > imagePoints1, imagePoints2;
	vector<Point2f> corners1, corners2;

	vector<Point3f> obj;
	for (int j = 0; j<board_n; j++)
	{
		obj.push_back(Point3f(j / board_w, j%board_w, 0.0f));
	}

	Mat img1, img2, gray1, gray2;

	int success = 0, k = 0;
	bool found1 = false, found2 = false;

	VideoCapture cap1 = VideoCapture(0);
	VideoCapture cap2 = VideoCapture(1);
	while (success < numBoards)
	{
		cap1 >> img1;
		cap2 >> img2;
		//resize(img1, img1, Size(320, 280));
		//resize(img2, img2, Size(320, 280));
		cvtColor(img1, gray1, CV_BGR2GRAY);
		cvtColor(img2, gray2, CV_BGR2GRAY);

		found1 = findChessboardCorners(img1, board_sz, corners1, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
		found2 = findChessboardCorners(img2, board_sz, corners2, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

		if (found1)
		{
			cornerSubPix(gray1, corners1, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
			drawChessboardCorners(gray1, board_sz, corners1, found1);
		}

		if (found2)
		{
			cornerSubPix(gray2, corners2, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
			drawChessboardCorners(gray2, board_sz, corners2, found2);
		}

		imshow("image1", gray1);
		imshow("image2", gray2);

		k = waitKey(10);
		if (found1 && found2)
		{
			k = waitKey(0);
		}
		if (k == 27)
		{
			break;
		}
		if (k == ' ' && found1 != 0 && found2 != 0)
		{
			imagePoints1.push_back(corners1);
			imagePoints2.push_back(corners2);
			object_points.push_back(obj);
			printf("Corners stored\n");
			success++;

			if (success >= numBoards)
			{
				break;
			}
		}
	}

	destroyAllWindows();
	printf("Starting Calibration\n");
	Mat CM1 = Mat(3, 3, CV_64FC1);
	Mat CM2 = Mat(3, 3, CV_64FC1);
	Mat D1, D2;
	Mat R, T, E, F;
	stereoCalibrate(object_points, imagePoints1, imagePoints2,
		CM1, D1, CM2, D2, img1.size(), R, T, E, F
	);
	/*
		cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 100, 1e-5),
		CV_CALIB_SAME_FOCAL_LENGTH | CV_CALIB_ZERO_TANGENT_DIST
	*/
	FileStorage fs1("mystereocalib.yml", FileStorage::WRITE);
	fs1 << "CM1" << CM1;
	fs1 << "CM2" << CM2;
	fs1 << "D1" << D1;
	fs1 << "D2" << D2;
	fs1 << "R" << R;
	fs1 << "T" << T;
	fs1 << "E" << E;
	fs1 << "F" << F;

	printf("Done Calibration\n");

	printf("Starting Rectification\n");

	Mat R1, R2, P1, P2, Q;
	stereoRectify(CM1, D1, CM2, D2, img1.size(), R, T, R1, R2, P1, P2, Q);
	fs1 << "R1" << R1;
	fs1 << "R2" << R2;
	fs1 << "P1" << P1;
	fs1 << "P2" << P2;
	fs1 << "Q" << Q;

	printf("Done Rectification\n");

	printf("Applying Undistort\n");

	Mat map1x, map1y, map2x, map2y;
	Mat imgU1, imgU2;

	initUndistortRectifyMap(CM1, D1, R1, P1, img1.size(), CV_32FC1, map1x, map1y);
	initUndistortRectifyMap(CM2, D2, R2, P2, img2.size(), CV_32FC1, map2x, map2y);

	printf("Undistort complete\n");

	while (1)
	{
		cap1 >> img1;
		cap2 >> img2;

		remap(img1, imgU1, map1x, map1y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
		remap(img2, imgU2, map2x, map2y, INTER_LINEAR, BORDER_CONSTANT, Scalar());

		imshow("image1", imgU1);
		imshow("image2", imgU2);

		k = waitKey(5);

		if (k == 27)
		{
			break;
		}
	}

	cap1.release();
	cap2.release();

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

void chess_calibrate(VideoCapture &capture1, VideoCapture &capture2, Mat &mtx1, Mat &mtx2, Mat &RT1, Mat &RT2, Mat &dist1, Mat &dist2, Mat &fund_mat) {


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
