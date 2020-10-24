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

double compute_inlier(std::vector<double> residuals, double ratio)
{
    std::sort(residuals.begin(), residuals.end());
    return residuals[floor(ratio * residuals.size())];
}

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
    loss_function = new ceres::HuberLoss(0.5);
    local_parameterization = new ceres::EigenQuaternionParameterization();
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 100;
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
                                     Quaterniond q,
                                     Vector3d t)
{
    size_t pt_size = pc->points.size();
    double pro_err = 0;
    for (size_t i = 0; i < pt_size; i++)
    {
        Vector3d p_l(pc->points[i].x, pc->points[i].y, pc->points[i].z);
        ceres::CostFunction* cost_func;
        cost_func = p2p::Create(p_l, p0_c, n);
        block_id = problem.AddResidualBlock(cost_func, loss_function, buffer, buffer + 4);
        residual_block_ids.push_back(block_id);
        pro_err += abs(n.dot(q * p_l + t - p0_c));
    }
    cout<<"average projection error "<<pro_err/pt_size<<endl;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "calibrate");
    ros::NodeHandle nh("~");

    ros::Publisher pub_out = nh.advertise<sensor_msgs::PointCloud2>("/color_pt", 10000);

    string pointcloud_path, camera_param, image_path, normal_path, scene_pointcloud;
    nh.getParam("pointcloud_path", pointcloud_path);
    nh.getParam("scene_pointcloud", scene_pointcloud);
    nh.getParam("camera_param", camera_param);
    nh.getParam("image_path", image_path);
    nh.getParam("normal_path", normal_path);
    cv::FileStorage param_reader(camera_param, cv::FileStorage::READ);
    cv::Mat camera_matrix, dist_coeff;
    param_reader["camera_matrix"] >> camera_matrix;
    param_reader["distortion_coefficients"] >> dist_coeff;
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);

    vector<Vector3d, aligned_allocator<Vector3d>> cam_nors, lidar_nors;

    Vector3d n1_c(-0.272656, 0.723249, 0.634483);
    Vector3d n2_c(0.352557, 0.770933, 0.53044);
    Vector3d n3_c(0.00794265, 0.774883, 0.632055);
    cam_nors.push_back(n1_c);
    cam_nors.push_back(n2_c);
    cam_nors.push_back(n3_c);

    Vector3d n1_l(0.639109, 0.266962, -0.721298);
    Vector3d n2_l(0.52839, -0.34726, -0.774735);
    Vector3d n3_l(0.629665, -0.00584401, -0.776845);
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
    cout<<"angular distance "<<q_gt.angularDistance(q)<<endl;

    Vector3d d_c(4.33844, 3.79384, 4.41193);
    Vector3d d_l(4.29866, 3.67984, 4.30644);

    Vector3d p0_l = n_l.transpose().inverse() * (d_l);
    Vector3d p0_c = n_c.transpose().inverse() * (d_c);
    Vector3d t = p0_c - q * p0_l;

    Vector3d t_gt(0.0175, 0.0548, 0);
    cout<<"linear distance "<<(t - t_gt).norm()<<endl;

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
    *pc_src = read_pointcloud(scene_pointcloud);
    size_t pc_size = pc_src->points.size();

    for (int iter = 0; iter < 1; iter++)
    {
        extrin_calib calib;
        calib.add_parameterblock();
        calib.init(q, t);

        std::fstream file;
        file.open(normal_path);
        vector<Vector3d, aligned_allocator<Vector3d>> n_cam, center_cam;
        vector<int> valid_n;
        int num;
        double nx, ny, nz, cx, cy, cz, d;

        while (!file.eof())
        {
            file >> num >> nx >> ny >> nz >> cx >> cy >> cz >> d;
            valid_n.push_back(num);
            n_cam.push_back(Vector3d(nx, ny, nz));
            center_cam.push_back(Vector3d(cx, cy, cz));
        }
        file.close();
        valid_n.pop_back();
        n_cam.pop_back();
        center_cam.pop_back();

        for (size_t i = 0; i < valid_n.size(); i++)
        {
            pcl::PointCloud<PointType>::Ptr pc_(new pcl::PointCloud<PointType>);
            *pc_ = read_pointcloud(pointcloud_path + to_string(valid_n[i]) + ".json");
            calib.add_residualblock(pc_, center_cam[i], n_cam[i], q, t);
        }
        ceres::Solve(calib.options, &(calib.problem), &(calib.summary));
        cout<<calib.summary.BriefReport()<<endl;

        Eigen::Map<Eigen::Quaterniond> q_ = Eigen::Map<Eigen::Quaterniond>(calib.buffer);
        Eigen::Map<Eigen::Vector3d> t_ = Eigen::Map<Eigen::Vector3d>(calib.buffer + 4);
        q.w() = q_.w();
        q.x() = q_.x();
        q.y() = q_.y();
        q.z() = q_.z();
        t(0) = t_(0);
        t(1) = t_(1);
        t(2) = t_(2);
        cout<<"angular distance "<<q_gt.angularDistance(q)<<endl;
        cout<<"linear distance "<<(t - t_gt).norm()<<endl;
        cout<<t(0)<<" "<<t(1)<<" "<<t(2)<<endl;
    }

    vector<cv::Point3f> world_pts;
    vector<cv::Point2f> image_pts;

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

    sensor_msgs::PointCloud2 laserCloudMsg;
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