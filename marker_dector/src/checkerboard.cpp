#include <string>
#include <iostream>
#include <fstream>
#include <ros/ros.h>
#include <Eigen/Eigen>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;
using namespace Eigen;

int main(int argc, char** argv)
{
    ros::init(argc, argv, "checker_board");
    ros::NodeHandle nh("~");

    string camera_param, image_path, normal_write_path;
    int corner_height, corner_width, image_num;
    double square_size, reproject_err;
    nh.getParam("camera_param_config", camera_param);
    nh.getParam("image_path", image_path);
    nh.getParam("normal_write_path", normal_write_path);
    nh.getParam("corner_height", corner_height);
    nh.getParam("corner_width", corner_width);
    nh.getParam("square_size", square_size);
    nh.getParam("image_num", image_num);
    nh.getParam("reproject_err", reproject_err);

    cv::FileStorage param_reader(camera_param, cv::FileStorage::READ);
    cv::Mat camera_matrix, dist_coeff;
    param_reader["camera_matrix"] >> camera_matrix;
    param_reader["distortion_coefficients"] >> dist_coeff;

    vector<Vector3d, aligned_allocator<Vector3d>> n_cam, center_cam;
    vector<int> valid_n;
    vector<double> dis;

    for (int k = 100; k <= image_num; k++)
    {
        string filename = image_path + to_string(k) + ".png";
        cout << "processing image " << k <<endl;
		cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);
        cv::Mat gray_img;
		cv::cvtColor(image, gray_img, CV_BGR2GRAY);
        cv::Size boardSize(corner_width, corner_height);
		vector<cv::Point3f> world_corners;
		vector<cv::Point2f> img_corners, repro_corners;
        
		cv::Vec3d rvec, tvec;

		bool found = findChessboardCorners(image, boardSize, img_corners,
            CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
        
		if (found)
            cornerSubPix(gray_img, img_corners, Size(5, 5), Size(-1, -1),
                TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.01));
        else
            continue;

        drawChessboardCorners(image, boardSize, Mat(img_corners), found);
        filename = image_path + "gray/" + "IMG_" + to_string(k) + ".png";
        imwrite(filename, gray_img);

        for (int i = 0; i < boardSize.height; i++)
            for (int j = 0; j < boardSize.width; j++)
                world_corners.push_back(Point3f(float(j * square_size),
                                                float(i * square_size), 0));

        solvePnP(world_corners, img_corners, camera_matrix, dist_coeff, rvec, tvec);
        projectPoints(Mat(world_corners), Mat(rvec), Mat(tvec),
                      camera_matrix, dist_coeff, repro_corners);
        double err = norm(Mat(img_corners), Mat(repro_corners), NORM_L2);
        err = sqrt(err * err / world_corners.size());
        if (err < reproject_err)
        {
            cout << "reprojected error " << err << endl;
            valid_n.push_back(k);
        }

        cv::Mat R;
        cv::Rodrigues(rvec, R);
        Matrix3d R_eigen;
        Vector3d t_eigen;

        for (int i = 0; i < 3; i++)
        {
            t_eigen(i) = tvec(i);
            for (int j = 0; j < 3; j++)
                R_eigen(i, j) = R.at<double>(i, j);
        }
        
        vector<Vector3d> candidates;
        Vector3d center;
        
        for (size_t i = 0; i < img_corners.size(); i++)
        {
            Vector3d p(world_corners[i].x, world_corners[i].y, world_corners[i].z);
            p = R_eigen * p + t_eigen;
            candidates.push_back(p);
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
        if (svd_nor(2) < 0)
            svd_nor *= -1;
        n_cam.push_back(svd_nor);
        center_cam.push_back(center);
        dis.push_back(abs(svd_nor.dot(center)));
    }

    std::ofstream file;
    file.open(normal_write_path, std::ofstream::trunc);
    for (size_t i = 0; i < valid_n.size(); i++)
    {
        file << valid_n[i] << " "
             << n_cam[i](0) << " "
             << n_cam[i](1) << " "
             << n_cam[i](2) << " "
             << center_cam[i](0) << " "
             << center_cam[i](1) << " "
             << center_cam[i](2) << " "
             << dis[i] << "\n";
    }
    file.close();

    cout << "complete!" << endl;

    ros::spin();
}