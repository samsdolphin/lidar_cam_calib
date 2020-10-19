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
#include "calibrate.hpp"

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

extrin_calib::extrin_calib()
{
    loss_function = new ceres::HuberLoss(0.1);
    local_parameterization = new ceres::EigenQuaternionParameterization();
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 1;
    options.minimizer_progress_to_stdout = false;
    options.check_gradients = false;
    options.gradient_check_relative_precision = 1e-10;
    for (int j = 0; j < 7; j++)
        buffer[j] = 0;
    buffer[3] = 1;
}

void extrin_calib::add_parameterblock()
{
    problem.AddParameterBlock(buffer, 4, local_parameterization);
    problem.AddParameterBlock(buffer + 4, 3);
}

void extrin_calib::init(Quaterniond q, Vector3d t)
{
    buffer[0] = q.x();
    buffer[1] = q.y();
    buffer[2] = q.z();
    buffer[3] = q.w();
    buffer[4] = t(0);
    buffer[5] = t(1);
    buffer[6] = t(2);
}

void extrin_calib::add_residualblock(pcl::PointCloud<PointType>::Ptr pc,
                                     Vector3d p0_c,
                                     Vector3d n,
                                     double d)
{
    size_t pt_size = pc->points.size();
    for (size_t i = 0; i < pt_size; i++)
    {
        Vector3d p_l(pc->points[i].x, pc->points[i].y, pc->points[i].z);
        ceres::CostFunction* cost_func;
        cost_func = p2p::Create(p_l, p0_c, n, d);
        block_id = problem.AddResidualBlock(cost_func, loss_function, buffer, buffer + 4);
        residual_block_ids.push_back(block_id);
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "calibrate");
    ros::NodeHandle nh("~");

    ros::Publisher pub_out = nh.advertise<sensor_msgs::PointCloud2>("/color_pt", 10000);

    string pointcloud_path, camera_param, image_path;
    nh.getParam("pointcloud_path", pointcloud_path);
    nh.getParam("camera_param", camera_param);
    nh.getParam("image_path", image_path);
    cv::FileStorage param_reader(camera_param, cv::FileStorage::READ);
    cv::Mat camera_matrix, dist_coeff;
    param_reader["camera_matrix"] >> camera_matrix;
    param_reader["distortion_coefficients"] >> dist_coeff;
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);

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
    cout<<"angular distance "<<q_gt.angularDistance(q)<<endl;

    Vector3d d_c(3.77202, 2.73181, 3.20342);
    Vector3d d_l(3.72805, 2.65214, 3.18523);

    Vector3d p0_l = n_l.transpose().inverse() * (d_l);
    Vector3d p0_c = n_c.transpose().inverse() * (d_c);
    Vector3d t = p0_c - q * p0_l;
    cout<<"t "<<t(0)<<" "<<t(1)<<" "<<t(2)<<endl;

    cv::Vec3d rvec, tvec;
    cv::Mat R_mat = cv::Mat_<double>(3, 3);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            R_mat.at<double>(i, j) = R(i, j);

    cv::Rodrigues(R_mat, rvec);
    for (int i = 0; i < 3; i++)
        tvec(i) = t(i);

    pcl::PointCloud<PointType>::Ptr pc_src(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr pc_out(new pcl::PointCloud<PointType>);
    *pc_src = read_pointcloud(pointcloud_path);
    size_t pc_size = pc_src->points.size();

    vector<cv::Point3f> world_pts;
    vector<cv::Point2f> image_pts;
    for (size_t i = 0; i < pc_size; i++)
    {
        Point3f p(pc_src->points[i].x, pc_src->points[i].y, pc_src->points[i].z);
        world_pts.push_back(p);
        projectPoints(Mat(world_pts), Mat(rvec), Mat(tvec), camera_matrix, dist_coeff, image_pts);
        world_pts.clear();
        int c = image_pts[0].x;
        int r = image_pts[0].y;
        if (r >= image.size().height || c >= image.size().width)
            continue;
        
        Vec3b pixel = image.at<Vec3b>(r, c);
        PointType point;
        point.x = float (pc_src->points[i].x); 
        point.y = float (pc_src->points[i].y); 
        point.z = float (pc_src->points[i].z);
        point.r = uint8_t (pixel[2]);
        point.g = uint8_t (pixel[1]);
        point.b = uint8_t (pixel[0]);
        pc_out->push_back(point);
    }

    sensor_msgs::PointCloud2 laserCloudMsg;
    pcl::toROSMsg(*pc_out, laserCloudMsg);
    laserCloudMsg.header.stamp = ros::Time::now();
    laserCloudMsg.header.frame_id = "/camera_init";
    pub_out.publish(laserCloudMsg);

    extrin_calib calib;
    calib.add_parameterblock();
    calib.init(q, t);
    pcl::PointCloud<PointType>::Ptr pc_(new pcl::PointCloud<PointType>);
    *pc_ = read_pointcloud("/home/sam/catkin_ws/src/lidar_cam_calib/plane_detector/data/hkust/large/left_.json");
    calib.add_residualblock(pc_, p0_c, n1_c, d_c(0));
    *pc_ = read_pointcloud("/home/sam/catkin_ws/src/lidar_cam_calib/plane_detector/data/hkust/large/middle_.json");
    calib.add_residualblock(pc_, p0_c, n2_c, d_c(1));
    *pc_ = read_pointcloud("/home/sam/catkin_ws/src/lidar_cam_calib/plane_detector/data/hkust/large/right_.json");
    calib.add_residualblock(pc_, p0_c, n3_c, d_c(2));
    ceres::Solve(calib.options, &(calib.problem), &(calib.summary));

    Eigen::Map<Eigen::Quaterniond> q_ = Eigen::Map<Eigen::Quaterniond>(calib.buffer);
    Eigen::Map<Eigen::Vector3d> t_ = Eigen::Map<Eigen::Vector3d>(calib.buffer + 4);
    q.w() = q_.w();
    q.x() = q_.x();
    q.y() = q_.y();
    q.z() = q_.z();
    t(0) = t_(0);
    t(1) = t_(1);
    t(2) = t_(2);

    cout<<"q "<<q.w()<<" "<<q.x()<<" "<<q.y()<<" "<<q.z()<<endl;
    cout<<"angular distance "<<q_gt.angularDistance(q)<<endl;
    cout<<"t "<<t(0)<<" "<<t(1)<<" "<<t(2)<<endl;

    R = q.toRotationMatrix();
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            R_mat.at<double>(i, j) = R(i, j);

    cv::Rodrigues(R_mat, rvec);
    for (int i = 0; i < 3; i++)
        tvec(i) = t(i);

    for (size_t i = 0; i < pc_size; i++)
    {
        Point3f p(pc_src->points[i].x, pc_src->points[i].y, pc_src->points[i].z);
        world_pts.push_back(p);
        projectPoints(Mat(world_pts), Mat(rvec), Mat(tvec), camera_matrix, dist_coeff, image_pts);
        world_pts.clear();
        int c = image_pts[0].x;
        int r = image_pts[0].y;
        if (r >= image.size().height || c >= image.size().width)
            continue;
        
        Vec3b pixel = image.at<Vec3b>(r, c);
        PointType point;
        point.x = float (pc_src->points[i].x); 
        point.y = float (pc_src->points[i].y); 
        point.z = float (pc_src->points[i].z);
        point.r = uint8_t (pixel[2]);
        point.g = uint8_t (pixel[1]);
        point.b = uint8_t (pixel[0]);
        pc_out->push_back(point);
    }

    pcl::toROSMsg(*pc_out, laserCloudMsg);
    laserCloudMsg.header.stamp = ros::Time::now();
    laserCloudMsg.header.frame_id = "/camera_init";
    pub_out.publish(laserCloudMsg);

    ros::Rate loop_rate(1);
    while (ros::ok())
    {
        ros::spinOnce();
        loop_rate.sleep();
    }
}