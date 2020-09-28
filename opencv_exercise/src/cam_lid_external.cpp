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
double is_number(string str);
Eigen::Matrix3d inner;

class external_cali
{
public:
  external_cali(PnPData p)
  {
    pd = p;
  }

  template <typename T>
  bool operator()(const T *_q, const T *_t, T *residuals) const
  {
    Eigen::Matrix<T, 3, 3> innerT = inner.cast<T>();
    Eigen::Quaternion<T> q_incre{_q[ 3 ], _q[ 0 ], _q[ 1 ], _q[ 2 ]};
    Eigen::Matrix<T, 3, 1> t_incre{_t[ 0 ], _t[ 1 ], _t[ 2 ]};

    Eigen::Matrix<T, 3, 1> p_l(T(pd.x), T(pd.y), T(pd.z));
    Eigen::Matrix<T, 3, 1> p_c = q_incre.toRotationMatrix() * p_l + t_incre;
    Eigen::Matrix<T, 3, 1> p_2 = innerT * p_c;

    residuals[0] = p_2[0]/p_2[2] - T(pd.u);
    residuals[1] = p_2[1]/p_2[2] - T(pd.v);

    // cout << residuals[0] << " " << residuals[1] << endl;

    return true;
  }

  static ceres::CostFunction *Create(PnPData p)
  {
    return (new ceres::AutoDiffCostFunction<external_cali, 2, 4, 3>(new external_cali(p)));
  }


private:
  PnPData pd;

};


int main(int argc, char **argv)
{
  ros::init(argc, argv, "cam_lid_external");
  ros::NodeHandle n;


  PnPData pdata[DATA_SIZE];

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
  }

  inFile.close();
  Eigen::Matrix4d extrin;

  inner << 1.730735067136013e+03, -0.000682525720977, 1.515012142085100e+3,
  0,  1.730530820356212e+03, 1.044575428820981e+03,
  0, 0, 1;

  Eigen::Matrix3d R;
  R << 0, -1, 0,
  0, 0, -1,
  1, 0, 0;

  Eigen::Quaterniond q(R);
  double ext[7];

  ext[0] = q.x();
  ext[1] = q.y();
  ext[2] = q.z();
  ext[3] = q.w();
  ext[4] = 0;
  ext[5] = 0.04;
  ext[6] = 0;

  Eigen::Map<Eigen::Quaterniond> m_q = Eigen::Map<Eigen::Quaterniond>(ext);
  Eigen::Map<Eigen::Vector3d> m_t = Eigen::Map<Eigen::Vector3d>(ext+4);

  ceres::LocalParameterization * q_parameterization = new ceres::EigenQuaternionParameterization();
  ceres::Problem problem;

  problem.AddParameterBlock(ext, 4, q_parameterization);
  problem.AddParameterBlock(ext+4, 3);
  
  for(int i=0; i<DATA_SIZE; i++)
  {
    ceres::CostFunction *cost_function;
    cost_function = external_cali::Create(pdata[i]);
    problem.AddResidualBlock(cost_function, NULL, ext, ext+4);
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  // options.max_num_iterations = 1;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  cout << summary.BriefReport() << endl;

  cout << m_q.toRotationMatrix() << endl;
  cout << m_t << endl;

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

