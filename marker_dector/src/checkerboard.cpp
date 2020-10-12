#include <string>
#include <iostream>
#include <fstream>
#include <ros/ros.h>
#include <Eigen/Eigen>
#include "camera_calibration.h"

using namespace std;
using namespace cv;
using namespace Eigen;

// string dir = "/home/sam/images_small_685/reorder/";
// string output_filename = "/home/sam/catkin_ws/src/lidar_cam_calib/marker_dector/log";
string path = "/home/sam/catkin_ws/src/lidar_cam_calib/marker_dector/fig/";
string cam_cal = "/home/sam/catkin_ws/src/lidar_cam_calib/marker_dector/config/left.yaml";

int main(int argc, char** argv)
{
    ros::init(argc, argv, "checker_board");
    ros::NodeHandle nd("~");

    Pattern pattern = CHESSBOARD;
    double square_size = 0.02;
    int height = 6;
    int width = 9;
    int flags = 0;
    double aspectRatio = 1;
    bool writeExtrinsics = true;
    bool writePoints = true;
    int num_intrinsic = 361;
    cv::FileStorage param_reader(cam_cal, cv::FileStorage::READ);
    cv::Mat camera_matrix, dist_coeff;

    param_reader["camera_matrix"] >> camera_matrix;
    param_reader["distortion_coefficients"] >> dist_coeff;

    // int count = 0;
    // for (int i = 0; i < 400; i++)
    // {
    //     string filename = dir + "IMG_" + std::to_string(i) +".jpg";
    //     cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    //     if (image.size().height)
    //     {
    //         string rewrite_filename = dir + "reorder/" + std::to_string(count) + ".jpg";
    //         cv::imwrite(rewrite_filename, image);
    //         count++;
    //     }
    //     cout<<"processing "<<i<<endl;
    // }

    // camera_calibrator cam_calib(square_size, height, width, flags, output_filename,
    //                             pattern, aspectRatio, writeExtrinsics, writePoints);
	// cam_calib.load_images(num_intrinsic, dir);
	// cam_calib.calibrate();
	// cam_calib.undistort(dir, dir + "undistorted/");

    vector<vector<cv::Point2f>> camera_points;
	vector<vector<cv::Point3f>> world_points;
    for (int k = 3; k <= 3; k++)
    {
        // string filename = path + "s" + std::to_string(k) +".png";
        string filename = path + "image.png";
		cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);
        cv::Mat gray_img, draw_image;
		cv::cvtColor(image, gray_img, CV_BGR2GRAY);
		cv::Size image_size = image.size();
        cv::Size boardSize;
		boardSize.height = 7;
		boardSize.width = 9;
		vector<cv::Point3f> world_corners;
		vector<cv::Point2f> corners;
        
		Vec3d rvec, tvec;

		bool found = findChessboardCorners(image, boardSize, corners,
            CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
        
		if (found)
            cornerSubPix(gray_img, corners, Size(5, 5), Size(-1, -1),
                TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.01));

        if (found)
        {
            camera_points.push_back(corners);
            for(int i = 0; i < boardSize.height; i++)
                for(int j = 0; j < boardSize.width; j++)
                    world_corners.push_back(Point3f(float(j * square_size),
                                                    float(i * square_size), 0));
            world_points.push_back(world_corners);
            world_points.resize(camera_points.size(), world_points[0]);
            // drawChessboardCorners(image, boardSize, Mat(corners), found);
        }

        // calibrateCamera(world_points, camera_points, image_size, camera_matrix,
        //                 dist_coeff, rvec, tvec, flags | CALIB_FIX_K4 | CALIB_FIX_K5);
        solvePnP(world_corners, corners, camera_matrix, dist_coeff, rvec, tvec);

        Mat R;
        cv::Rodrigues(rvec, R);

        Matrix3d R_eigen;
        for(int i = 0; i < 3; i++)
            for(int j = 0; j < 3; j++)
                R_eigen(i, j) = R.at<double>(i, j);

        Vector3d t_eigen;
        for (int i = 0; i < 3; i++)
            t_eigen(i) = tvec(i);

        std::vector<Eigen::Vector3d> candidates;
        Eigen::Vector3d center;
        
        for (int i = 0; i < corners.size(); i++)
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
        // cout<<"svd_nor "<<svd_nor(0)<<" "<<svd_nor(1)<<" "<<svd_nor(2)<<endl;

        cv::Point2f po, px, py, pz;
        Point3f world_po(0, 0, 0);
        Point3f world_px((boardSize.width - 1) * square_size, 0, 0);
        Point3f world_py(0, (boardSize.height - 1) * square_size, 0);
        Point3f world_pz(0, 0, (boardSize.height - 1) * square_size);
        std::vector<cv::Point3f> world_pts;
        std::vector<cv::Point2f> image_pts;
        world_pts.push_back(world_po);
        world_pts.push_back(world_px);
        world_pts.push_back(world_py);
        world_pts.push_back(world_pz);
        projectPoints(Mat(world_pts), Mat(rvec), Mat(tvec), camera_matrix, dist_coeff, image_pts);
        po = image_pts[0];
        px = image_pts[1];
        py = image_pts[2];
        pz = image_pts[3];

        cv::line(image, po, px, cv::Scalar(0, 0, 255), 2);
        cv::line(image, po, py, cv::Scalar(255, 0, 0), 2);
        cv::line(image, po, pz, cv::Scalar(0, 255, 0), 2);
        cv::undistort(image, draw_image, camera_matrix, dist_coeff);
        string axis_filename = path + "after_image.png";
        cv::imwrite(axis_filename, draw_image);

        // vector<Point2f> imagePoints2;
        // projectPoints(Mat(world_corners), rvec, tvec, camera_matrix, dist_coeff, imagePoints2);
        // for (int i = 0; i < corners.size(); i++)
        // {
        //     cout<<corners[i].x<<" "<<corners[i].y<<endl;
        //     cout<<imagePoints2[i].x<<" "<<imagePoints2[i].y<<endl;
        // }
    }

    ros::spin();
}