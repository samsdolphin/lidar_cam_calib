#include <string>
#include <iostream>
#include <fstream>
#include <ros/ros.h>
#include "camera_calibration.h"

using namespace std;
string dir = "/home/sam/images_small_685/";
string output_filename = "/home/sam/catkin_ws/src/lidar_cam_calib/marker_dector/log";

int main(int argc, char** argv)
{
    ros::init(argc, argv, "checker_board");
    ros::NodeHandle nd("~");

    Pattern pattern = CHESSBOARD;
    double square_size = 68.5 / 1000;
    int height = 6;
    int width = 9;
    int flags = 0;
    double aspectRatio = 1;
    bool writeExtrinsics = true;
    bool writePoints = true;
    int num_intrinsic = 393;

    camera_calibrator cam_calib(square_size, height, width, flags, output_filename,
                                pattern, aspectRatio, writeExtrinsics, writePoints);

	cam_calib.load_images(num_intrinsic, dir);
	cam_calib.calibrate();
	// cam_calib.undistort(dir, dir + "undistorted/");

    ros::spin();
}