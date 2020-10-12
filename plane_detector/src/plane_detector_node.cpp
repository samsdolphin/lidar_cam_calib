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
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>
#include <ceres/ceres.h>
#include <ros/ros.h>

using namespace std;
using namespace Eigen;

typedef pcl::PointXYZI PointType;
pcl::PointCloud<PointType>::Ptr pc_surf(new pcl::PointCloud<PointType>);
string address = "/home/sam/catkin_ws/src/lidar_cam_calib/plane_detector/mid40/hkust/s2.json";

#define PI 3.14159265
int max_svd_it = 30;
double inlier_ratio = 0.95;
double max_svd_val = 0.9;
double max_radius = 0.2;

pcl::PointCloud<PointType> read_pointcloud(std::string path)
{
	pcl::PointCloud<PointType> pc;
	pc.points.resize(1e8);
	std::fstream file;
	file.open(path);
	size_t cnt = 0;
	float x, y, z;
	while (!file.eof())
	{
		file >> x >> y >> z;
		pc.points[cnt].x = x;
		pc.points[cnt].y = y;
		pc.points[cnt].z = z;
		cnt++;
	}
	file.close();
	pc.points.resize(cnt);
	return pc;
}

pcl::PointCloud<PointType>::Ptr append_cloud(pcl::PointCloud<PointType>::Ptr pc1, pcl::PointCloud<PointType> pc2)
{
    size_t size1 = pc1->points.size();
    size_t size2 = pc2.points.size();
    pc1->points.resize(size1 + size2);
    for (size_t i = size1; i < size1 + size2; i++)
    {
        pc1->points[i].x = pc2.points[i - size1].x;
        pc1->points[i].y = pc2.points[i - size1].y;
        pc1->points[i].z = pc2.points[i - size1].z;
    }
    return pc1;
}

void surf_callback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
    pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
    pcl::fromROSMsg(*msg, *pc);
    pc_surf = append_cloud(pc_surf, *pc);

    ofstream file_w;
    file_w.open(address, std::ofstream::app);
    for (size_t i = 0; i < pc->points.size(); i++)
        file_w << pc->points[i].x << "\t" << pc->points[i].y << "\t" << pc->points[i].z << "\n";
    file_w.close();
}

double compute_inlier(std::vector<double> residuals, double ratio)
{
    std::sort(residuals.begin(), residuals.end());
    // cout<<"inlier "<<residuals[floor(ratio * residuals.size())]<<endl;
    // cout<<"max "<<residuals[residuals.size()-1]<<endl;
    return residuals[floor(ratio * residuals.size())];
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "plane_detector");
    ros::NodeHandle nh;
    ros::Publisher pub_origin = nh.advertise<sensor_msgs::PointCloud2>("/origin_surf", 10000);
    ros::Publisher pub_rough = nh.advertise<sensor_msgs::PointCloud2>("/rough_surf", 10000);
    ros::Publisher pub_filt = nh.advertise<sensor_msgs::PointCloud2>("/filt_surf", 10000);
    ros::Publisher pub_nor = nh.advertise<visualization_msgs::MarkerArray>("svd_nor", 0);
    ros::Subscriber sub_surf = nh.subscribe<sensor_msgs::PointCloud2>("/pc2_surfaceN", 10000, surf_callback);

    pcl::PointCloud<PointType>::Ptr pc_src(new pcl::PointCloud<PointType>);
    *pc_src = read_pointcloud("/home/sam/catkin_ws/src/lidar_cam_calib/plane_detector/mid40/hkust/s2.json");
    pcl::PointCloud<PointType>::Ptr pc_rough(new pcl::PointCloud<PointType>);
    pc_rough->points.resize(1e8);
    size_t cnt = 0;

    for (size_t i = 0; i < pc_src->points.size(); i++)
    {
        if (pc_src->points[i].z > -0.8 && pc_src->points[i].z < 0 &&
            pc_src->points[i].y > 0.3 && pc_src->points[i].y < 1.3 &&
            pc_src->points[i].x > 3 && pc_src->points[i].x < 5)
            {
                pc_rough->points[cnt].x = pc_src->points[i].x;
                pc_rough->points[cnt].y = pc_src->points[i].y;
                pc_rough->points[cnt].z = pc_src->points[i].z;
                cnt++;
            }
    }
    pc_rough->points.resize(cnt);

    std::vector<Eigen::Vector3d> candidates, new_can;
    std::vector<double> residuals;
    Eigen::Vector3d center;
    for (size_t i = 0; i < pc_rough->points.size(); i++)
    {
        Eigen::Vector3d tmp(pc_rough->points[i].x, pc_rough->points[i].y, pc_rough->points[i].z);
        candidates.push_back(tmp);
    }

    for (int it = 0; it < max_svd_it; it++)
    {
        center.setZero();
        size_t pt_size = candidates.size();

        for (size_t i = 0; i < pt_size; i++)
            center += candidates[i];
        center /= pt_size;

        Eigen::MatrixXd A(pt_size, 3);

        for (size_t i = 0; i < pt_size; i++)
            A.row(i) = (candidates[i] - center).transpose();
        
        JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
        Eigen::Vector3d svd_nor = svd.matrixV().col(2);
        double sig_val = svd.singularValues()[2];
        cout<<"candidate size "<<pt_size<<", singular value "<<sig_val<<endl;

        for (size_t i = 0; i < pt_size; i++)
        {
            double tmp = svd_nor.dot(candidates[i] - center);
            residuals.push_back(abs(tmp));
        }
        double rej_val = compute_inlier(residuals, inlier_ratio);

        for (size_t i = 0; i < pt_size; i++)
        {
            double tmp = svd_nor.dot(candidates[i] - center);
            if (abs(tmp) < rej_val)
                new_can.push_back(candidates[i]);
        }

        candidates.clear();
        candidates = new_can;
        new_can.clear();

        visualization_msgs::MarkerArray marker_array;
        visualization_msgs::Marker marker;
        marker.header.frame_id = "/camera_init";
        marker.header.stamp = ros::Time();
        marker.ns = "my_namespace";
        marker.id = it;
        marker.type = visualization_msgs::Marker::ARROW;
        marker.action = visualization_msgs::Marker::ADD;
        marker.scale.x = 0.02;
        marker.scale.y = 0.05;
        marker.scale.z = 0.1;
        marker.color.a = 1.0;
        marker.color.r = cos(PI*it/max_svd_it);
        marker.color.g = sin(PI*it/max_svd_it);
        marker.color.b = 0.0;
        geometry_msgs::Point apoint;
        apoint.x = center(0);
        apoint.y = center(1);
        apoint.z = center(2);
        marker.points.push_back(apoint);
        apoint.x += svd_nor(0);
        apoint.y += svd_nor(1);
        apoint.z += svd_nor(2);
        marker.points.push_back(apoint);
        marker_array.markers.push_back(marker);        
        pub_nor.publish(marker_array);

        if (sig_val < max_svd_val)
        {
            for (size_t i = 0; i < pt_size; i++)
                if ((candidates[i] - center).norm() < max_radius)
                    new_can.push_back(candidates[i]);

            candidates.clear();
            candidates = new_can;
            new_can.clear();
            break;
        }
    }

    center.setZero();
    size_t pt_size = candidates.size();

    for (size_t i = 0; i < pt_size; i++)
        center += candidates[i];
    center /= pt_size;

    Eigen::MatrixXd A(pt_size, 3);

    for (size_t i = 0; i < pt_size; i++)
        A.row(i) = (candidates[i] - center).transpose();
    
    JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
    Eigen::Vector3d svd_nor = svd.matrixV().col(2);
    double sig_val = svd.singularValues()[2];
    cout<<"final point size "<<pt_size<<", singular value "<<sig_val<<endl;

    visualization_msgs::MarkerArray marker_array;
    visualization_msgs::Marker marker;
    marker.header.frame_id = "/camera_init";
    marker.header.stamp = ros::Time();
    marker.ns = "my_namespace";
    marker.id = max_svd_it;
    marker.type = visualization_msgs::Marker::ARROW;
    marker.action = visualization_msgs::Marker::ADD;
    marker.scale.x = 0.02;
    marker.scale.y = 0.05;
    marker.scale.z = 0.1;
    marker.color.a = 1.0;
    marker.color.r = 0.0;
    marker.color.g = sin(PI/2);
    marker.color.b = 0.0;
    geometry_msgs::Point apoint;
    apoint.x = center(0);
    apoint.y = center(1);
    apoint.z = center(2);
    marker.points.push_back(apoint);
    apoint.x += svd_nor(0);
    apoint.y += svd_nor(1);
    apoint.z += svd_nor(2);
    marker.points.push_back(apoint);
    marker_array.markers.push_back(marker);        
    pub_nor.publish(marker_array);

    pcl::PointCloud<PointType>::Ptr pc_filt(new pcl::PointCloud<PointType>);
    pc_filt->points.resize(pt_size);
    for (size_t i = 0; i < pt_size; i++)
    {
        pc_filt->points[i].x = candidates[i](0);
        pc_filt->points[i].y = candidates[i](1);
        pc_filt->points[i].z = candidates[i](2);
    }

    sensor_msgs::PointCloud2 laserCloudMsg;
    pcl::toROSMsg(*pc_src, laserCloudMsg);
    laserCloudMsg.header.stamp = ros::Time::now();
    laserCloudMsg.header.frame_id = "/camera_init";
    pub_origin.publish(laserCloudMsg);

    pcl::toROSMsg(*pc_rough, laserCloudMsg);
    laserCloudMsg.header.stamp = ros::Time::now();
    laserCloudMsg.header.frame_id = "/camera_init";
    pub_rough.publish(laserCloudMsg);

    pcl::toROSMsg(*pc_filt, laserCloudMsg);
    laserCloudMsg.header.stamp = ros::Time::now();
    laserCloudMsg.header.frame_id = "/camera_init";
    pub_filt.publish(laserCloudMsg);

	ros::Rate loop_rate(1);
    while (ros::ok())
    {
        ros::spinOnce();
        loop_rate.sleep();
    }
}