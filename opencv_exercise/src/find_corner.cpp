#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <stdio.h>
#include <opencv2/imgproc/imgproc.hpp>
// #include "MvCameraControl.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

cv::Mat gray_img, src_img;
int max_corners = 500;
cv::RNG  random_number_generator;
void SubPixel_Demo(int, void*);

int main(int argc, char **argv)
{
    ros::init(argc, argv, "find_corner");
    ros::NodeHandle n;
    
    string filename;
    cout << "Give the filename" << endl;
    cin >> filename;
    filename = "/home/sam/catkin_ws/src/lidar_cam_calib/opencv_exercise/pic1121/" + filename +".bmp";
    src_img = cv::imread(filename);

    if(src_img.empty())
    {
        printf("No Picture\n");
        return 0;
    }

    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    cameraMatrix.at<double>(0, 0) = 1.730735067136013e+03;
    cameraMatrix.at<double>(0, 1) = -0.000682525720977;
    cameraMatrix.at<double>(0, 2) = 1.515012142085100e+03;
    cameraMatrix.at<double>(1, 1) = 1.730530820356212e+03;
    cameraMatrix.at<double>(1, 2) = 1.044575428820981e+03;

    cv::Mat distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
    distCoeffs.at<double>(0, 0) = -0.095982349277083;
    distCoeffs.at<double>(1, 0) = 0.090204555257461;
    distCoeffs.at<double>(2, 0) = 0.001075320356832;
    distCoeffs.at<double>(3, 0) = -0.001243809361172;
    distCoeffs.at<double>(4, 0) = 0;

    cv::Mat view, rview, map1, map2;
    cv::Size imageSize = src_img.size();
    cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(),cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0), imageSize, CV_16SC2, map1, map2);

    cv::remap(src_img, src_img, map1, map2, cv::INTER_LINEAR);

    cv::namedWindow("source", CV_WINDOW_KEEPRATIO);
    cv::imshow("source", src_img);
    cv::waitKey(0);

    cv::cvtColor(src_img, gray_img, cv::COLOR_BGR2GRAY);
    cv::namedWindow("output", CV_WINDOW_KEEPRATIO);
    // cv::createTrackbar("Number of corner", "output", &max_corners, 1000, SubPixel_Demo);
    // SubPixel_Demo(0, 0);
    
    vector<cv::Point2f> corners;
    cout << "Give the corner coordinate" << endl;
    while(1)
    {
        cv::Point2f p;
        cin >> p.x >> p.y;
        if(p.x < 0.1 && p.y < 0.1)
        {
            break;
        }
        corners.push_back(p);
    }
    cv::Size winSize = cv::Size(5, 5);
	cv::Size zerozone = cv::Size(-1, -1);
    cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 40, 0.001);

    cv::cornerSubPix(gray_img, corners, winSize, zerozone, criteria);

    cv::Mat result_img = src_img.clone();
    for(uint t=0; t<corners.size(); t++)
    {
        cv::circle(result_img, corners[t], 3, cv::Scalar(random_number_generator.uniform(0, 255), random_number_generator.uniform(0, 255), random_number_generator.uniform(0, 255)), 1, 8, 0);
        printf("(%.3f %.3f)", corners[t].x, corners[t].y);
    }
    printf("\n");
    imshow("output", result_img);
    cv::waitKey(0);
    return 0;
}

void SubPixel_Demo(int, void*)
{
    if(max_corners < 5)
    {
        max_corners = 5;
    }
    vector<cv::Point2f> corners;
    double qualityLevel = 0.01;
    double minDistance = 10;
    int blockSize = 3;
    double k = 0.04;
    cout << "***********************" << endl;

    cv::goodFeaturesToTrack(gray_img, corners, max_corners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);
    cout << "corner number: " << corners.size() << endl;
    cv::Mat result_img = src_img.clone();
    // for(uint t=0; t<corners.size(); t++)
    // {
    //     circle(result_img, corners[t], 2, cv::Scalar(random_number_generator.uniform(0, 255), 
	// 		random_number_generator.uniform(0, 255), random_number_generator.uniform(0, 255)), 2, 8, 0);
    // }
    // imshow("output", result_img);

    cv::Size winSize = cv::Size(5, 5);
	cv::Size zerozone = cv::Size(-1, -1);
    cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 40, 0.001);
    cv::cornerSubPix(gray_img, corners, winSize, zerozone, criteria);

    for(uint t=0; t<corners.size(); t++)
    {
        cv::circle(result_img, corners[t], 3, cv::Scalar(random_number_generator.uniform(0, 255), random_number_generator.uniform(0, 255), random_number_generator.uniform(0, 255)), 1, 8, 0);
        printf("(%.3f, %.3f)", corners[t].x, corners[t].y);
    }
    printf("\n");
    imshow("output", result_img);

}







