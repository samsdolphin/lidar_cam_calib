# lidar_cam_calib

use `ceres-solver 1.14.0`

roslaunch plane_detector extractor.launch

roslaunch plane_detector detector.launch

2023.3.28
增加了鱼眼相机内参标定的代码，使用`check_intrinsic`选取合适标定的照片集，使用`calib_fisheye`标定不同文件夹的照片。

todo
自动检测合适标定图片