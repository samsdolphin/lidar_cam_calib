#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp> 
#include <ros/ros.h>

using namespace std;
using namespace cv;

string path = "/home/sam/catkin_ws/src/lidar_cam_calib/marker_dector/fig/";

int main(int argc, char** argv)
{
    ros::init(argc, argv, "checker_detect");
    ros::NodeHandle nh;
    int rows_num = 9;
    int cols_num = 7;

    string filename = path + "image.png";
    cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);
    cv::Size image_size = image.size();
    cv::Mat gray_img;
    cv::cvtColor(image, gray_img, CV_BGR2GRAY);
    cv::Size boardSize;
    boardSize.height = rows_num;
    boardSize.width = cols_num;
    vector<cv::Point2f> checkerboard_corners;
    vector<cv::Point2f> pointbuf;

    bool found = findChessboardCorners(image, boardSize, pointbuf,
        CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
    
    if (found)
        cornerSubPix(gray_img, pointbuf, Size(5,5),
            Size(-1,-1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.01));

    if (found)
    {
        drawChessboardCorners(image, boardSize, Mat(pointbuf), found);
        imshow("found", image);
        // camera_points.push_back(pointbuf);
    }
    else
        cout<<"not found"<<endl;
    

    ros::Rate loop_rate(1);
    while (ros::ok())
    {
        ros::spinOnce();
        loop_rate.sleep();
    }
}