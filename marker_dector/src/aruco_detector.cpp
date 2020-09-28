#include <iostream>
#include <ros/ros.h>
#include <ros/console.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/PoseStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <nav_msgs/Odometry.h>
#include <aruco/cvdrawingutils.h>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <Eigen/Eigen>
#include <Eigen/SVD>

using namespace std;
using namespace cv;
using namespace Eigen;

Quaterniond Q_mg;
cv::Mat K, D;
ros::Publisher pose_pub;

void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    try
    {
        Mat InImage = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8)->image;
        Mat UndisImg, VinsImg, TagImg, Mask;
        undistort(InImage, UndisImg, K, D);

        vector<int> markerIds; 
        vector<vector<Point2f> > markerCorners;
        cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);

        cv::aruco::detectMarkers(UndisImg, dictionary, markerCorners, markerIds);
        cv::aruco::drawDetectedMarkers(UndisImg, markerCorners, markerIds);

        vector<Vec3d> rvecs, tvecs;

        if (markerIds.size() > 0)
        {
            cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.14, K, D, rvecs, tvecs);
            cv::aruco::drawDetectedMarkers(UndisImg, markerCorners);
        }

        for (unsigned int i = 0; i < markerIds.size(); i++)
        {
            Vec3d rvec = rvecs[i];
            Vec3d tvec = tvecs[i];
            Mat R;
            cv::Rodrigues(rvec, R);

            Matrix3d R_eigen;
            for(int j = 0; j < 3; j++)
                for(int k = 0; k < 3; k++)
                    R_eigen(j, k) = R.at<double>(j, k);

            Vector3d T_eigen;
            for (int j = 0; j < 3; j++)
                T_eigen(j) = tvec(j);

            Quaterniond Q_cm;
            Q_cm = R_eigen;
            Q_cm.normalize();

            cout<<"T_eigen: "<<T_eigen(0)<<" "<<T_eigen(1)<<" "<<T_eigen(2)<<endl;
            //cout<<"camera_position: "<<camera_position.transpose()<<endl;
            Vector3d temp_marker_pos = T_eigen;
            Quaterniond temp_marker_ori = Q_cm*Q_mg;

            if (sqrt(pow(T_eigen(0), 2) + pow(T_eigen(1), 2)) <= 0.3)
            {
                // cout<<"WITH IN RANGE"<<endl;
                geometry_msgs::PoseStamped marker_pose;

                marker_pose.header = img_msg->header;
                marker_pose.header.frame_id = "world";

                marker_pose.pose.position.x = temp_marker_pos.x();
                marker_pose.pose.position.y = temp_marker_pos.y();
                marker_pose.pose.position.z = temp_marker_pos.z();

                marker_pose.pose.orientation.w = temp_marker_ori.w();
                marker_pose.pose.orientation.x = temp_marker_ori.x();
                marker_pose.pose.orientation.y = temp_marker_ori.y();
                marker_pose.pose.orientation.z = temp_marker_ori.z();

                pose_pub.publish(marker_pose);
            }
        }
        imshow("view", UndisImg);
        waitKey(1);
    }
    catch(cv_bridge::Exception& ex)
    {
        ROS_ERROR("'%s'", ex.what());
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "aruco_detector");
    ros::NodeHandle nh("~");
    namedWindow("view", 0); // 1:autosize, 0:resize
    resizeWindow("view", 1920, 1080);

    pose_pub = nh.advertise<geometry_msgs::PoseStamped>("marker_pose", 10);
    ros::Subscriber img_sub = nh.subscribe("image_raw", 30, img_callback);

    string cam_cal;
    nh.getParam("cam_cal_file", cam_cal);
    cv::FileStorage param_reader(cam_cal, cv::FileStorage::READ);

    param_reader["camera_matrix"] >> K;
    param_reader["distortion_coefficients"] >> D;

    Matrix3d R_mg;
    R_mg << 0, -1, 0,
            0, 0, 1,
            -1, 0, 0;
    Q_mg = R_mg;
    Q_mg.normalize();

    ros::spin();
    destroyWindow("view");
}