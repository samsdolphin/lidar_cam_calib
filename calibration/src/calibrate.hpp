#ifndef CALIBRATE_HPP
#define CALIBRATE_HPP

#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>

class extrin_calib
{
public:
  ceres::Problem problem;
  ceres::LossFunction* loss_function;
  ceres::LocalParameterization* local_parameterization;
  ceres::Solver::Options options;
  ceres::Solver::Summary summary;
  ceres::ResidualBlockId block_id;
  std::vector<ceres::ResidualBlockId> residual_block_ids;
  std::vector<ceres::ResidualBlockId> temp_residual_block_ids;
  double buffer[7];
public:
  extrin_calib();
  ~extrin_calib(){};
  void add_parameterblock();
  void init(Eigen::Quaterniond, Eigen::Vector3d);
  void add_residualblock(pcl::PointCloud<pcl::PointXYZRGB>::Ptr,
                          Eigen::Vector3d,
                          Eigen::Vector3d,
                          Eigen::Quaterniond,
                          Eigen::Vector3d);
};

struct p2p
{
  Eigen::Vector3d _p, _pt, _n;

  p2p(const Eigen::Vector3d& p,
      const Eigen::Vector3d& pt,
      const Eigen::Vector3d& n):
      _p(p),
      _pt(pt),
      _n(n){};
  
  template <typename T>
  bool operator()(const T* _q, const T* _t, T* residual) const
  {
      Eigen::Quaternion<T> q{_q[3], _q[0], _q[1], _q[2]};
      Eigen::Matrix<T, 3, 1> t{_t[0], _t[1], _t[2]};

      Eigen::Matrix<T, 3, 1> p = _p.template cast<T>();
      Eigen::Matrix<T, 3, 1> pt = _pt.template cast<T>();
      Eigen::Matrix<T, 3, 1> n = _n.template cast<T>();
      Eigen::Matrix<T, 3, 1> p_c;
      p_c = q * p + t - pt;
      residual[0] = abs(n.dot(p_c));
      return true;
  };

  static ceres::CostFunction* Create(const Eigen::Vector3d& p,
                                      const Eigen::Vector3d& pt,
                                      const Eigen::Vector3d& n)
  {
      return (new ceres::AutoDiffCostFunction<p2p, 1, 4, 3>(
          new p2p(p, pt, n)));
  }
};

#endif