#include<iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
using namespace std;
 
void myHough(Mat src, Mat dst)
{
	vector<Vec2f> lines;//用于储存参数空间的交点
	HoughLines(src, lines, 1, CV_PI / 180, 120, 0, 0);//针对不同像素的图片注意调整阈值
	const int alpha = 10;//alpha取得充分大，保证画出贯穿整个图片的直线
	//lines中存储的是边缘直线在极坐标空间下的rho和theta值，在图像空间(直角坐标系下)只能体现出一个点
	//以该点为基准，利用theta与斜率之间的关系，找出该直线上的其他两个点(可能不在图像上)，之后以这两点画出直线
	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		double cs = cos(theta), sn = sin(theta);
		double x = rho * cs, y = rho * sn;
		Point pt1(cvRound(x + alpha * (-sn)), cvRound(y + alpha * cs));
		Point pt2(cvRound(x - alpha * (-sn)), cvRound(y - alpha * cs));
		line(dst, pt1, pt2, Scalar(0, 0, 255), 1, LINE_AA);
	}
}

int main()
{
	Mat mImage = imread("/home/sam/catkin_ws/src/lidar_cam_calib/image.png");
	if (mImage.data == 0)
	{
		cerr << "Image reading error !" << endl;
		system("pause");
	}
	namedWindow("The original image");
	imshow("The original image", mImage);

	Mat mMiddle ;
	cvtColor(mImage, mMiddle,  COLOR_BGR2GRAY);//Canny()只接受单通道8位图像，边缘检测前先将图像转换为灰度图
	Canny(mImage, mMiddle, 50, 150, 3);//边缘检测，检测结果作为霍夫变换的输入
	Mat mResult = mImage.clone();
	myHough(mMiddle, mResult);//将结果展示在原图像上
	namedWindow("The processed image");
	imshow("The processed image", mResult);
	waitKey();
	destroyAllWindows();
	return 0;
}