#include <iostream>
#include <fstream>
#include <string>

#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <opencv2/core/core.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>
#include <ceres/ceres.h>
#include <ros/ros.h>

using namespace std;
using namespace Eigen;
using namespace cv;

int main(int argc, char** argv)
{
    ros::init(argc, argv, "calibrate");
    ros::NodeHandle nh("~");

    vector<Vector3d, aligned_allocator<Vector3d>> cam_nors, lidar_nors;

    Vector3d n1_c(0.528802, 0.151168, 0.835175);
    Vector3d n2_c(0.00391974, 0.817007, 0.576615);
    Vector3d n3_c(-0.728075, 0.310619, 0.611083);
    // n2_c -= n1_c * n1_c.dot(n2_c);
    // n3_c -= n1_c * n1_c.dot(n3_c);
    // n3_c -= n2_c * n2_c.dot(n3_c);
    cam_nors.push_back(n1_c);
    cam_nors.push_back(n2_c);
    cam_nors.push_back(n3_c);

    Vector3d n1_l(0.8672, -0.475618, -0.147487);
    Vector3d n2_l(0.555614, -0.00203993, -0.83143);
    Vector3d n3_l(0.601813, 0.733008, -0.317051);
    // n2_l -= n1_l * n1_l.dot(n2_l);
    // n3_l -= n1_l * n1_l.dot(n3_l);
    // n3_l -= n2_l * n2_l.dot(n3_l);
    lidar_nors.push_back(n1_l);
    lidar_nors.push_back(n2_l);
    lidar_nors.push_back(n3_l);

    Matrix3d n_l, n_c, R, R_gt;
    for (int i = 0; i < 3; i++)
    {
        n_c.col(i) = cam_nors[i];
        n_l.col(i) = lidar_nors[i];
    }
    R = n_c * n_l.inverse();
    Quaterniond q(R);
    q.normalized();
    R_gt << 0, -1, 0, 0, 0, -1, 1, 0, 0;
    Quaterniond q_gt(R_gt);
    cout<<"q "<<q.w()<<" "<<q.x()<<" "<<q.y()<<" "<<q.z()<<endl;
    // cout<<"angular distance "<<q_gt.angularDistance(q)<<endl;

    Vector3d d_c(3.49818, 2.5973, 3.35969);
    Vector3d d_l(3.65261, 2.47498, 3.33524);

    Vector3d p0_l = n_l.transpose().inverse() * (d_l);
    Vector3d p0_c = n_c.transpose().inverse() * (d_c);
    // cout<<"P0_l "<<p0_l(0)<<" "<<p0_l(1)<<" "<<p0_l(2)<<endl;
    // cout<<"P0_c "<<p0_c(0)<<" "<<p0_c(1)<<" "<<p0_c(2)<<endl;
    Vector3d t = p0_c - q * p0_l;
    cout<<"t "<<t(0)<<" "<<t(1)<<" "<<t(2)<<endl;

    ros::Rate loop_rate(1);
    while (ros::ok())
    {
        ros::spinOnce();
        loop_rate.sleep();
    }
}