/*************************************************************************
    > File Name: camera_calibration.h
    > Author: Yang Zhou
    > Mail: zhouyang@shanghaitech.edu.cn 
    > Created Time: Mon 02 Jul 2018 01:33:26 PM EDT
 ************************************************************************/
#ifndef __CAMERA_CALIBRATION__
#define __CAMERA_CALIBRATION__
//#include <ros/ros.h>
//#include <image_transport/image_transport.h>
//#include <cv_bridge/cv_bridge.h>
//#include <sensor_msgs/image_encodings.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp> 
//#include <camera_calibration_parsers/parse.h>
//#include <geometry_msgs/Point.h>
#include <string>
#include <vector>
#include <iostream>
using namespace std;
using namespace cv;
enum Pattern { CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID };
class camera_calibrator{
	private:
		//ros::NodeHandle _node_handle;
		vector<cv::KeyPoint> _keypoints;
		bool calibration_done;
		//sensor_msgs::CameraInfo camera_info;
	public:
		camera_calibrator(float _square_size,int _rows_num, int _cols_num, int _flags, const string& _output_filename, Pattern _pattern, double _aspectRatio, bool _writeExtrinsics, bool _writePoints);
		cv::Mat	camera_matrix;
		cv::Mat dist_coeff;
		cv::Size image_size;
		cv::Size boardSize;
		float square_size;
		int rows_num;
		int cols_num;
		int flags;
		string output_filename;

		Pattern pattern;
		double aspectRatio;
		bool writeExtrinsics;
		bool writePoints;

		int image_num;
		vector<vector<cv::Point2f> > camera_points;
		vector<vector<cv::Point3f> > world_points;
		vector<cv::Mat> rvec,tvec;


		void load_images(int _image_num, const string& path);
		void calibrate();
		void undistort(const string& path, const string& undistort_path);
		
		static double computeReprojectionErrors(
	        const vector<vector<Point3f> >& objectPoints,
	        const vector<vector<Point2f> >& imagePoints,
	        const vector<Mat>& rvecs, const vector<Mat>& tvecs,
	        const Mat& cameraMatrix, const Mat& distCoeffs,
	        vector<float>& perViewErrors );



		static void calcChessboardCorners(Size boardSize, float squareSize, vector<Point3f>& corners, Pattern patternType = CHESSBOARD);
		static bool runCalibration( vector<vector<Point2f> > imagePoints,
                    Size imageSize, Size boardSize, Pattern patternType,
                    float squareSize, float aspectRatio,
                    int flags, Mat& cameraMatrix, Mat& distCoeffs,
                    vector<Mat>& rvecs, vector<Mat>& tvecs,
                    vector<float>& reprojErrs,
                    double& totalAvgErr);
		static void saveCameraParams( const string& filename,
                       Size imageSize, Size boardSize,
                       float squareSize, float aspectRatio, int flags,
                       const Mat& cameraMatrix, const Mat& distCoeffs,
                       const vector<Mat>& rvecs, const vector<Mat>& tvecs,
                       const vector<float>& reprojErrs,
                       const vector<vector<Point2f> >& imagePoints,
                       double totalAvgErr );

		static bool runAndSave(const string& outputFilename,
                const vector<vector<Point2f> >& imagePoints,
                Size imageSize, Size boardSize, Pattern patternType, float squareSize,
                float aspectRatio, int flags, Mat& cameraMatrix,
                Mat& distCoeffs, bool writeExtrinsics, bool writePoints , vector<Mat>& rvecs, vector<Mat>& tvecs);
                                         
};


#endif /* ifndef __CAMERA_CALIBRATION__*/
