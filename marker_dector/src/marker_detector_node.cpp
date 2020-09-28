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
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace message_filters;

ros::Publisher pose_pub;

Quaterniond Q_mg;
//Vector3d verr(0, 0, 0);
//Vector3d vavg(2.6363, -0.0557088, 1.1583);
//double err = 0.0;
//int cnt = 0;
cv::Mat K, D;

void img_callback(const sensor_msgs::ImageConstPtr &img_msg, const nav_msgs::OdometryConstPtr &camera_pose_msg)
{
    try
    {
        //cout<<"IMG_CALLBACK"<<endl;
        Mat InImage = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8)->image;
        Mat UndisImg, VinsImg, TagImg, Mask;
        undistort(InImage, UndisImg, K, D);
        vector<Point3f> pts_3;
        vector<Point2f> pts_2;

        Vector3d camera_position(camera_pose_msg->pose.pose.position.x, camera_pose_msg->pose.pose.position.y, camera_pose_msg->pose.pose.position.z);
        Quaterniond camera_orientation(camera_pose_msg->pose.pose.orientation.w, camera_pose_msg->pose.pose.orientation.x, camera_pose_msg->pose.pose.orientation.y, camera_pose_msg->pose.pose.orientation.z);
        camera_orientation.normalize();

        vector<int> markerIds; 
        vector<vector<Point2f> > markerCorners;
        cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);

        cv::aruco::detectMarkers(UndisImg, dictionary, markerCorners, markerIds);
        cv::aruco::drawDetectedMarkers(UndisImg, markerCorners, markerIds);

        vector<Vec3d> rvecs, tvecs;

        if (markerIds.size()>0)
        {
            cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.12, K, D, rvecs, tvecs);
            cv::aruco::drawDetectedMarkers(UndisImg, markerCorners);
        }

        for (unsigned int i=0; i<markerIds.size(); i++)
        {
            Vec3d rvec = rvecs[i];
            Vec3d tvec = tvecs[i];
            Mat R;
            cv::Rodrigues(rvec, R);

            Matrix3d R_eigen;
            for(int j=0; j<3; j++)
                for(int k=0; k<3; k++)
                    R_eigen(j, k) = R.at<double>(j, k);

            Vector3d T_eigen;
            for (int j=0; j<3; j++)
                T_eigen(j) = tvec(j);

            Quaterniond Q_cm;
            Q_cm = R_eigen;
            Q_cm.normalize();

            cout<<"T_eigen: "<<T_eigen.transpose()<<endl;
            //cout<<"camera_position: "<<camera_position.transpose()<<endl;
            Vector3d temp_marker_pos = camera_orientation.toRotationMatrix()*T_eigen+camera_position;
            Quaterniond temp_marker_ori = camera_orientation*Q_cm*Q_mg;

            if (sqrt(pow(T_eigen(0), 2) + pow(T_eigen(1), 2)) <= 0.3)
            {
                cout<<"WITH IN RANGE"<<endl;
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
/*
                verr += temp_marker_pos;
                cout<<"tag: "<<T_eigen.transpose()<<endl;
                cnt++;
                cout<<"avg: "<<verr.transpose()/cnt<<endl;
                Vector3d dif = temp_marker_pos-vavg;
                err += sqrt(pow(dif(0),2)+pow(dif(1),2)+pow(dif(2),2));
                cout<<"cnt: "<<cnt<<", err: "<<err<<endl;
                */
            }
            //std::cout<<"marker_ori:\n"<<temp_marker_ori.toRotationMatrix()<<endl;
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
    ros::init(argc, argv, "marker_detector");
    ros::NodeHandle nh("~");
    namedWindow("view", 0); // 1:autosize, 0:resize
    resizeWindow("view", 960, 480);

    pose_pub = nh.advertise<geometry_msgs::PoseStamped>("marker_pose", 10);

    //img_sub = nh.subscribe("image_raw", 30, img_callback);
    //cam_pose_sub = nh.subscribe("camera_pose", 30, cam_pose_callback);

    message_filters::Subscriber<sensor_msgs::Image> sub_img(nh, "image_raw", 30);
    message_filters::Subscriber<nav_msgs::Odometry> sub_cam_pose(nh, "camera_pose", 100);

    typedef sync_policies::ApproximateTime<sensor_msgs::Image, nav_msgs::Odometry> MySyncPolicy;
    Synchronizer<MySyncPolicy> sync(MySyncPolicy(100), sub_img, sub_cam_pose);
    sync.registerCallback(boost::bind(&img_callback, _1, _2));

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