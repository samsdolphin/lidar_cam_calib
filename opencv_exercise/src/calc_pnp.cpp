#include <ros/ros.h>
#include <iostream>
#include <ceres/ceres.h>
#include <Eigen/Core>
#include <fstream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>
#include <pcl_conversions/pcl_conversions.h>


typedef pcl::PointXYZRGB PointType;
using namespace std;

#define DATA_SIZE 11*4

struct PnPData
{
    double x, y, z, u, v;
};

struct myPoint
{
  float x;
  float y;
  float z;
  unsigned char ref;
};


double is_number(string str);

class cali_model
{
public:
    cali_model(PnPData p)
    {
        pd = p;
    }

    template <typename T>
    bool operator()(const T* const m, T* residuals) const
    {
        // residuals[0] = T(pd.x)*m[0] + T(pd.y)*m[1] + T(pd.z)*m[2] + m[3] 
        // - T(pd.u) * (T(pd.x)*m[8] + T(pd.y)*m[9] + T(pd.z)*m[10] + m[11]);
        // residuals[1] = T(pd.x)*m[4] + T(pd.y)*m[5] + T(pd.z)*m[6] + m[7] 
        // - T(pd.v) * (T(pd.x)*m[8] + T(pd.y)*m[9] + T(pd.z)*m[10] + m[11]) ;

        residuals[0] = (T(pd.x)*m[0] + T(pd.y)*m[1] + T(pd.z)*m[2] + m[3]) / (T(pd.x)*m[8] + T(pd.y)*m[9] + T(pd.z)*m[10] + m[11]) - T(pd.u);
        residuals[1] = (T(pd.x)*m[4] + T(pd.y)*m[5] + T(pd.z)*m[6] + m[7]) / (T(pd.x)*m[8] + T(pd.y)*m[9] + T(pd.z)*m[10] + m[11]) - T(pd.v);

        return true;
    }

    static ceres::CostFunction* Create(PnPData p)
    {
        return (new ceres::AutoDiffCostFunction<cali_model, 2, 12>(new cali_model(p)));
    }

private:
    PnPData pd;

};

cv::Mat image;

int main(int argc, char **argv)
{
    ros::init(argc, argv, "calc_pnp");
    ros::NodeHandle n;
    ros::Publisher pub_cloud = n.advertise<sensor_msgs::PointCloud2>("pub_pointcloud2", 10);
    PnPData pdata[DATA_SIZE];
    double m[12];

    image = cv::imread("/home/dji/catkin_ws/src/opencv_exercise/pic1121/w2.bmp");
    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    cameraMatrix.at<double>(0, 0) = 1.730735067136013e+03;
    cameraMatrix.at<double>(0, 1) = -0.000682525720977;
    cameraMatrix.at<double>(0, 2) = 1.515012142085100e+03;
    cameraMatrix.at<double>(1, 1) = 1.730530820356212e+03;
    cameraMatrix.at<double>(1, 2) = 1.044575428820981e+03;

    cv::Mat distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
    distCoeffs.at<double>(0, 0) = -0.095982349277083;
    distCoeffs.at<double>(1, 0) = 0.090204555257461;
    distCoeffs.at<double>(2, 0) = 0.001075320356832;
    distCoeffs.at<double>(3, 0) = -0.001243809361172;
    distCoeffs.at<double>(4, 0) = 0;
    cv::Mat view, rview, map1, map2;
    cv::Size imageSize = image.size();
    cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(),cv::getOptimalNewCameraMatrix(cameraMatrix,distCoeffs, imageSize, 1, imageSize, 0), imageSize, CV_16SC2, map1, map2);
    cv::remap(image, image, map1, map2, cv::INTER_LINEAR);

    ifstream inFile;
    inFile.open("/home/dji/catkin_ws/src/opencv_exercise/bag_pic_corner.txt");
    string lineStr;

    getline(inFile, lineStr);
    int count = 0;
    while(getline(inFile, lineStr) && count<DATA_SIZE)
    {
        for(int i=0; i<4; i++)
        {
            if(getline(inFile, lineStr))
            {
                stringstream line(lineStr);
                string str;
                
                line >> str;
                pdata[count].x = is_number(str);
                // cout << pdata[count].x << " ";

                line >> str;
                pdata[count].y = is_number(str);
                // cout << pdata[count].y << " ";

                line >> str;
                pdata[count].z = is_number(str);
                // cout << pdata[count].z << " ";

                line >> str;
                pdata[count].u = is_number(str);
                // cout << pdata[count].u << " ";

                line >> str;
                pdata[count].v = is_number(str);
                // cout << pdata[count].v << " ";
                
                // cout << endl;
            }
            count++;
        }
        // cout << endl;
    }

    inFile.close();
    Eigen::Matrix4d inner, extrin, m_matrix;
    // inner << 1.730735067136013e+03, -0.000682525720977, 1.515012142085100e+03, 0,
    // 0,  1.730530820356212e+03, 1.044575428820981e+03, 0,
    // 0, 0, 1, 0, 
    // 0, 0, 0, 1;
    inner << 1.730735067136013e+03, -0.000682525720977, 1.515012142085100e+3, 0,
    0,  1.730530820356212e+03, 1.044575428820981e+03, 0,
    0, 0, 1, 0, 
    0, 0, 0, 1;
    extrin << 0, -1, 0, 0, 
    0, 0, -1, 0.04, 
    1, 0, 0, 0, 
    0, 0, 0, 1;
    
    m_matrix = inner * extrin;

    // m_matrix << -1498.43, 1923.47, -36.2926, -685.951, 
    // -1095.27, 38.3737, 1793.61, -388.505, 
    // -1.03112, 0.0373611, -0.0236389, -0.395082, 
    // 0, 0, 0, 1;  
    // cout << m_matrix << endl << endl;

    for(int i=0; i<3; i++)
    {
        for(int j=0; j<4; j++)
        {
            m[i*4 + j] = m_matrix(i, j);
        }
    }
    
    ceres::Problem problem;

    for(int i=0; i<DATA_SIZE; i++)
    {
        ceres::CostFunction *cost_function;
        cost_function = cali_model::Create(pdata[i]);
        problem.AddResidualBlock(cost_function, NULL, m);
    }


    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    // options.max_num_iterations = 1;

    ceres::Solver::Summary summary;
    cout << "start" << endl;
    ceres::Solve(options, &problem, &summary);
    cout << summary.BriefReport() << endl;

    for(int i=0; i<12; i++)
    {
        cout << m[i] << " ";
        if((i+1)%4 == 0)
        {
            cout << endl;
        }
    }
    
    m_matrix << m[0], m[1], m[2], m[3],
    m[4], m[5], m[6], m[7],
    m[8], m[9], m[10], m[11],
    0, 0, 0, 1;
    cout << m_matrix * inner.inverse() << endl;


    // PnPData pd;
    // double re = 0;
    // for(int i=0; i<DATA_SIZE; i++)
    // {
    //     pd = pdata[i];
    //     re += fabs( (pd.x*m[0] + pd.y*m[1] + pd.z*m[2] + m[3]) / (pd.x*m[8] + pd.y*m[9] + pd.z*m[10] + m[11]) - pd.u );
    //     re += fabs( (pd.x*m[4] + pd.y*m[5] + pd.z*m[6] + m[7]) / (pd.x*m[8] + pd.y*m[9] + pd.z*m[10] + m[11]) - pd.v );
    //     if((i+1)%4 == 0)
    //     {
    //         cout << re << endl;
    //         re = 0;
    //     }
    // }


    inFile.open("/home/dji/catkin_qt/src/replay/datas/1121_2.dat", ios::in | ios::binary);
    if(!inFile.is_open())
    {
        cout << "Can't Open" << endl;
        return 0;
    }
    uint num;
    pcl::PointCloud<PointType> pointcl;

    while(inFile.read((char*)&num, sizeof(num)))
    {
        myPoint mp;
        PointType pp;
        for(uint i=0; i<num; i++)
        {
            inFile.read((char*)&mp, sizeof(mp));
            pp.x = mp.x;
            pp.y = mp.y;
            pp.z = mp.z;
            // pp.x = 1;
            // pp.y = mp.y / mp.x;
            // pp.z = mp.z / mp.x;
            int u = round( (pp.x*m[0] + pp.y*m[1] + pp.z*m[2] + m[3]) / (pp.x*m[8] + pp.y*m[9] + pp.z*m[10] + m[11]) );
            int v = round( (pp.x*m[4] + pp.y*m[5] + pp.z*m[6] + m[7]) / (pp.x*m[8] + pp.y*m[9] + pp.z*m[10] + m[11]) );


            if(u<image.size().width && v<image.size().height && u>0 && v>0)
            {
                pp.r = image.at<cv::Vec3b>(v, u)[0];
                pp.g = image.at<cv::Vec3b>(v, u)[1];
                pp.b = image.at<cv::Vec3b>(v, u)[2];
            }
            else
            {
                pp.r = 255;
                pp.g = 0;
                pp.b = 0;
            }
            pointcl.push_back(pp);
        }
    }
    // cin >> num;
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(pointcl, output);
    output.header.frame_id = "camera_init";
    while(n.ok())
    {
        pub_cloud.publish(output);
        ros::Duration(1).sleep();
    }
    

    ros::spin();
    return 0;

}

double is_number(string str)
{
    double d;
    stringstream sin(str);
    if(sin >> d)
    {
        return d;
    }
    cout << str << endl;
    cout << "huge error." << endl;
    exit(0);
}


