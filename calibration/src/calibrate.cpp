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
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp> 
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>
#include <ceres/ceres.h>
#include <ros/ros.h>

typedef pcl::PointXYZRGB PointType;
using namespace std;
using namespace Eigen;
using namespace cv;

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

int main(int argc, char** argv)
{
    ros::init(argc, argv, "calibrate");
    ros::NodeHandle nh("~");

    ros::Publisher pub_out = nh.advertise<sensor_msgs::PointCloud2>("/color_pt", 10000);

    string camera_param = "/home/sam/catkin_ws/src/lidar_cam_calib/marker_dector/config/left.yaml";
    string filename = "/home/sam/catkin_ws/src/lidar_cam_calib/marker_dector/fig/hkust/large/1.png";
    cv::FileStorage param_reader(camera_param, cv::FileStorage::READ);
    cv::Mat camera_matrix, dist_coeff;
    param_reader["camera_matrix"] >> camera_matrix;
    param_reader["distortion_coefficients"] >> dist_coeff;
    cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);

    vector<Vector3d, aligned_allocator<Vector3d>> cam_nors, lidar_nors;

    Vector3d n1_c(0.586455, 0.207063, 0.783068);
    Vector3d n2_c(-0.0134726, 0.901135, 0.433329);
    Vector3d n3_c(-0.824312, 0.269763, 0.497733);
    cam_nors.push_back(n1_c);
    cam_nors.push_back(n2_c);
    cam_nors.push_back(n3_c);

    Vector3d n1_l(0.786101, -0.584967, -0.199649);
    Vector3d n2_l(0.432133, 0.0192129, -0.901605);
    Vector3d n3_l(0.491403, 0.828714, -0.267874);
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
    R = q.toRotationMatrix();
    R_gt << 0, -1, 0, 0, 0, -1, 1, 0, 0;
    Quaterniond q_gt(R_gt);
    cout<<"q "<<q.w()<<" "<<q.x()<<" "<<q.y()<<" "<<q.z()<<endl;
    // cout<<"angular distance "<<q_gt.angularDistance(q)<<endl;

    Vector3d d_c(3.77202, 2.73181, 3.20342);
    Vector3d d_l(3.72805, 2.65214, 3.18523);

    Vector3d p0_l = n_l.transpose().inverse() * (d_l);
    Vector3d p0_c = n_c.transpose().inverse() * (d_c);
    Vector3d t = p0_c - q * p0_l;
    cout<<"t "<<t(0)<<" "<<t(1)<<" "<<t(2)<<endl;

    cv::Vec3d rvec, tvec;
    cv::Mat R_mat;
    float data_r = 
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            R_mat.at<double>(i, j) = R(i, j);
cout<<"GOOD"<<endl;
    cv::Rodrigues(R_mat, rvec);
    for (int i = 0; i < 3; i++)
        tvec(i) = t(i);
cout<<"GOOD"<<endl;
    pcl::PointCloud<PointType>::Ptr pc_src(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr pc_out(new pcl::PointCloud<PointType>);
    string pointcloud_path =
        "/home/sam/catkin_ws/src/lidar_cam_calib/plane_detector/data/hkust/large/middle.json";
    *pc_src = read_pointcloud(pointcloud_path);
    size_t pc_size = pc_src->points.size();
cout<<"GOOD"<<endl;
    vector<cv::Point3f> world_pts;
    vector<cv::Point2f> image_pts;
    for (size_t i = 0; i < pc_size; i++)
    {
        Point3f p(pc_src->points[i].x, pc_src->points[i].y, pc_src->points[i].z);
        world_pts.push_back(p);
        projectPoints(Mat(world_pts), Mat(rvec), Mat(tvec), camera_matrix, dist_coeff, image_pts);
        int r = image_pts[0].x;
        int c = image_pts[0].y;
        if (r >= 1080 || c >= 1920)
            continue;
        cout<<"r "<<r<<" c "<<c<<endl;
        Vec3b pixel = image.at<Vec3b>(r, c);
        // uchar blue = pixel[0];
        // uchar green = pixel[1];
        // uchar red = pixel[2];
        PointType point;
        point.x = float (pc_src->points[i].x); 
        point.y = float (pc_src->points[i].y); 
        point.z = float (pc_src->points[i].z);
        point.r = uint8_t (pixel[2]);
        point.g = uint8_t (pixel[1]);
        point.b = uint8_t (pixel[0]);
        pc_out->push_back (point);
    }

    sensor_msgs::PointCloud2 laserCloudMsg;
    pcl::toROSMsg(*pc_out, laserCloudMsg);
    laserCloudMsg.header.stamp = ros::Time::now();
    laserCloudMsg.header.frame_id = "/camera_init";
    pub_out.publish(laserCloudMsg);

    ros::Rate loop_rate(1);
    while (ros::ok())
    {
        pub_out.publish(laserCloudMsg);
        ros::spinOnce();
        loop_rate.sleep();
    }
}