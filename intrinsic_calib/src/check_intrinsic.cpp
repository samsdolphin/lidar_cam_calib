#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <fstream>
using namespace std;
using namespace cv;

string filepath = "/media/sam/data/huawei_chunran/";

double getDistance(Point2f point1, Point2f point2)
{
  double distance = sqrtf(powf((point1.x - point2.x),2) + powf((point1.y - point2.y),2));
  return distance;
}

int main()
{
  ofstream fout(filepath + "caliberation_result.txt");
  int image_count = 2500;
  double thr = 0.3;
  Size board_size = Size(8, 6);

  vector<Point2f> corners;
  vector<vector<Point2f>> corners_Seq, corners_Seq2;
  vector<Mat> image_Seq;
  int successImageNum = 0;
  
  for(int i = 0; i <= image_count; i++)
  {
    cv::Mat image = imread(filepath + "BR3/" + to_string(i) + ".png");
    if(image.empty())
      continue;
    
    Mat imageGray;
    cvtColor(image, imageGray , CV_RGB2GRAY);
    bool patternfound = findChessboardCorners(image, board_size, corners,
      CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
    if(!patternfound)
    {
      cout<<"no chessboard img "<<i<<endl;
      fout<<i<<endl;
      continue;
    }
    else
    {
      cornerSubPix(imageGray, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
      successImageNum++;
      corners_Seq.push_back(corners);
    }
    cout<<"processed img "<<i<<endl;
    image_Seq.push_back(image);
  }

  float square_size = 0.1015;
	vector<vector<Point3f>> object_Points, object_Points2;

  vector<int> point_counts;                                                         
  for(int t = 0; t < successImageNum; t++)
  {
    vector<Point3f> tempPointSet;
    for(int i = 0; i < board_size.height; i++)
      for(int j = 0; j < board_size.width; j++)
      {
        Point3f tempPoint;
        tempPoint.x = i*square_size;
        tempPoint.y = j*square_size;
        tempPoint.z = 0;
        tempPointSet.push_back(tempPoint);
      }
    object_Points.push_back(tempPointSet);
  }

  for(int i = 0; i< successImageNum; i++)
    point_counts.push_back(board_size.width * board_size.height);

  Size image_size = image_Seq[0].size();
  cv::Matx33d intrinsic_matrix;
  cv::Vec4d distortion_coeffs;
  std::vector<cv::Vec3d> rotation_vectors;
  std::vector<cv::Vec3d> translation_vectors;
  int flags = 0;
  flags |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
  flags |= cv::fisheye::CALIB_CHECK_COND;
  flags |= cv::fisheye::CALIB_FIX_SKEW;
  cout<<"begin 1st calibration..."<<endl;
  fisheye::calibrate(object_Points, corners_Seq, image_size, intrinsic_matrix, distortion_coeffs,
                     rotation_vectors, translation_vectors, flags, cv::TermCriteria(3, 20, 1e-6));
  cout<<"1st calibration completed!"<<endl;

  double total_err = 0.0;
  double err = 0.0;
  vector<Point2f> image_points2; // 3D点重投影得到的2D点
  int valid_cnt = 0;
  for(int i = 0; i < successImageNum; i++) 
  {
    vector<Point3f> tempPointSet = object_Points[i]; // 3D点
    fisheye::projectPoints(tempPointSet, image_points2, rotation_vectors[i], translation_vectors[i],
                           intrinsic_matrix, distortion_coeffs);
    vector<Point2f> tempImagePoint = corners_Seq[i]; // 提取到的2D点
    vector<Point3f> tempPointSet2;
    vector<Point2f> corners2;
    err = 0.0;
    for(size_t j = 0; j < tempImagePoint.size(); j++)
    {
      double tmp = getDistance(tempImagePoint[j], image_points2[j]);
      err += tmp;
      if(tmp <= thr)
      {
        corners2.push_back(corners_Seq[i][j]);
        tempPointSet2.push_back(object_Points[i][j]);
      }
    }
    corners_Seq2.push_back(corners2);
    object_Points2.push_back(tempPointSet2);
    valid_cnt += tempImagePoint.size();
    total_err += err;
  }
  cout<<"using new intrinsic total error "<<total_err/valid_cnt<<endl;
  // fout << intrinsic_matrix << endl;
  // fout << distortion_coeffs << endl;
  cout << intrinsic_matrix << endl;
  cout << distortion_coeffs << endl;

  return 0;
}