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

typedef pcl::PointXYZI PointType;
#define PI 3.14159265

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

void write_pointcloud(pcl::PointCloud<PointType>::Ptr pc, string path)
{
    std::ofstream file;
    file.open(path, std::ofstream::trunc);
    for (size_t i = 0; i < pc->points.size(); i++)
    {
        file << pc->points[i].x << " "
             << pc->points[i].y << " "
             << pc->points[i].z << "\n";
    }
    file.close();
}

double compute_inlier(std::vector<double> residuals, double ratio)
{
    std::sort(residuals.begin(), residuals.end());
    return residuals[floor(ratio * residuals.size())];
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "plane_detector");
    ros::NodeHandle nh("~");
    ros::Publisher pub_origin = nh.advertise<sensor_msgs::PointCloud2>("/origin_surf", 10000);
    ros::Publisher pub_rough = nh.advertise<sensor_msgs::PointCloud2>("/rough_surf", 10000);
    ros::Publisher pub_filt = nh.advertise<sensor_msgs::PointCloud2>("/filt_surf", 10000);
    ros::Publisher pub_nor = nh.advertise<visualization_msgs::MarkerArray>("svd_nor", 10000);

    string pointcloud_path, write_path, boundary_param;
    cv::Mat boundary;
    int max_SVD_iteration;
    double inlier_SVD_ratio, max_SVD_value, max_circle_radius;
    
    nh.getParam("pointcloud_path", pointcloud_path);
    nh.getParam("write_path", write_path);
    nh.getParam("boundary_param", boundary_param);
    nh.getParam("max_SVD_iteration", max_SVD_iteration);
    nh.getParam("inlier_SVD_ratio", inlier_SVD_ratio);
    nh.getParam("max_SVD_value", max_SVD_value);
    nh.getParam("max_circle_radius", max_circle_radius);

    cv::FileStorage param_reader(boundary_param, cv::FileStorage::READ);
    param_reader["boundary"] >> boundary;

    pcl::PointCloud<PointType>::Ptr pc_rough(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr pc_src(new pcl::PointCloud<PointType>);
    *pc_src = read_pointcloud(pointcloud_path);
    
    pc_rough->points.resize(1e8);
    size_t cnt = 0;

    for (size_t i = 0; i < pc_src->points.size(); i++)
    {
        if (
            pc_src->points[i].x > boundary.at<double>(0, 0) &&
            pc_src->points[i].x < boundary.at<double>(0, 1) &&
            pc_src->points[i].y > boundary.at<double>(1, 0) &&
            pc_src->points[i].y < boundary.at<double>(1, 1) &&
            pc_src->points[i].z > boundary.at<double>(2, 0) &&
            pc_src->points[i].z < boundary.at<double>(2, 1)
            )
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

    for (int it = 0; it < max_SVD_iteration; it++)
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
        if (svd_nor(0) < 0)
            svd_nor *= -1;
        double sig_val = svd.singularValues()[2];
        cout<<"candidate size "<<pt_size<<", singular value "<<sig_val<<endl;

        for (size_t i = 0; i < pt_size; i++)
        {
            double tmp = svd_nor.dot(candidates[i] - center);
            residuals.push_back(abs(tmp));
        }
        double rej_val = compute_inlier(residuals, inlier_SVD_ratio);

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
        marker.color.r = cos(PI*it/max_SVD_iteration);
        marker.color.g = sin(PI*it/max_SVD_iteration);
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

        if (sig_val < max_SVD_value)
        {
            for (size_t i = 0; i < pt_size; i++)
                if ((candidates[i] - center).norm() < max_circle_radius)
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
    if (svd_nor(0) < 0)
        svd_nor *= -1;
    double sig_val = svd.singularValues()[2];
    cout<<"final point size "<<pt_size<<", singular value "<<sig_val<<endl;
    cout<<"SVD "<<svd_nor(0)<<" "<<svd_nor(1)<<" "<<svd_nor(2)<<endl;
    double tmp = svd_nor.dot(center);
    cout<<"d "<<tmp<<endl;

    visualization_msgs::MarkerArray marker_array;
    visualization_msgs::Marker marker;
    marker.header.frame_id = "/camera_init";
    marker.header.stamp = ros::Time();
    marker.ns = "my_namespace";
    marker.id = max_SVD_iteration;
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
    write_pointcloud(pc_filt, write_path);

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