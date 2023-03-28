#include <iostream>
#include <fstream>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <ros/ros.h>

using namespace std;
using namespace Eigen;

typedef pcl::PointXYZI PointType;
pcl::PointCloud<PointType>::Ptr pc_surf(new pcl::PointCloud<PointType>);
string data_path;

pcl::PointCloud<PointType>::Ptr append_cloud(pcl::PointCloud<PointType>::Ptr pc1,
                                             pcl::PointCloud<PointType> pc2)
{
  size_t size1 = pc1->points.size();
  size_t size2 = pc2.points.size();
  pc1->points.resize(size1 + size2);
  for(size_t i = size1; i < size1 + size2; i++)
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
  file_w.open(data_path, std::ofstream::app);
  for(size_t i = 0; i < pc->points.size(); i++)
      file_w << pc->points[i].x << "\t" << pc->points[i].y << "\t" << pc->points[i].z << "\n";
  file_w.close();
}

double compute_inlier(std::vector<double> residuals, double ratio)
{
  std::sort(residuals.begin(), residuals.end());
  return residuals[floor(ratio * residuals.size())];
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "plane_extractor");
  ros::NodeHandle nh("~");
  ros::Subscriber sub_surf = nh.subscribe<sensor_msgs::PointCloud2>("/pc2_surfN", 10000, surf_callback);

  nh.getParam("data_write_path", data_path);

  ros::Rate loop_rate(1);
  while (ros::ok())
  {
    ros::spinOnce();
    loop_rate.sleep();
  }
}