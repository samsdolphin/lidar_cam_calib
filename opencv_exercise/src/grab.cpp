#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <stdio.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "MvCameraControl.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

bool is_undistorted = true;


unsigned int g_nPayloadSize = 0;

int main(int argc, char **argv)
{
    ros::init(argc, argv, "grab");
    ros::NodeHandle n;
    // printf("%d\n", CV_8UC3);
    // cv::Mat image = cv::imread("/home/dji/a.jpg", cv::IMREAD_COLOR);
    // cv::FileStorage fs("m.xml", cv::FileStorage::READ);
    // cv::Mat image;
    // fs["nn"] >> image;
    // if(image.empty())
    // {
    //     printf("Open Error\n");
    // }
    // cv::imshow("a", image);
    // cv::waitKey(0);
    // cv::destroyWindow("a");
    // cv::FileStorage fs("m.xml", cv::FileStorage::WRITE);
    // fs << "nn" << image;
    // fs.release();

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


    int nRet = MV_OK;
    void *handle = NULL;

    while(1)
    {
        MV_CC_DEVICE_INFO_LIST stDeviceList;
        memset(&stDeviceList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
        nRet = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &stDeviceList);
        if(MV_OK != nRet)
        {
            printf("Enum Devices fail!");
            break;
        }

        if(stDeviceList.nDeviceNum == 0)
        {
            printf("No Camera.\n");
            break;
        }

        nRet = MV_CC_CreateHandle(&handle, stDeviceList.pDeviceInfo[0]);
        if(MV_OK != nRet)
        {
            printf("Create Handle fail");
            break;
        }

        nRet = MV_CC_OpenDevice(handle);
        if(MV_OK != nRet)
        {
            printf("Open Device fail\n");
            break;
        }

        nRet = MV_CC_SetEnumValue(handle, "TriggerMode", 0);
        if(MV_OK != nRet)
        {
            printf("Set Trigger Mode fail\n");
            break;
        }

        // nRet = MV_CC_SetIntValue(handle, "GainAuto", 2);
        // if(nRet != MV_OK)
        // {
        //     // printf("Gain setting can't work.");
        //     // break;
        // }

        nRet = MV_CC_SetFloatValue(handle, "Gain", 18);
        if(nRet != MV_OK)
        {
            printf("Gain setting can't work.");
            // break;
        }

        MVCC_INTVALUE stParam;
        memset(&stParam, 0, sizeof(MVCC_INTVALUE));
        nRet = MV_CC_GetIntValue(handle, "PayloadSize", &stParam);
        if(MV_OK != nRet)
        {
            printf("Get PayloadSize fail\n");
            break;
        }
        g_nPayloadSize = stParam.nCurValue;

        // nRet = MV_CC_SetEnumValue(handle, "PixelFormat", 0x02180014);
        // if(nRet != MV_OK)
        // {
        //     printf("My setting can't work.");
        //     break;
        // }

        nRet = MV_CC_SetEnumValue(handle, "PixelFormat", 0x02180014);
        if(nRet != MV_OK)
        {
            printf("Pixel setting can't work.");
            break;
        }

        nRet = MV_CC_StartGrabbing(handle);
        if(MV_OK != nRet)
        {
            printf("Start Grabbing fail.\n");
            break;
        }


        MV_FRAME_OUT_INFO_EX stImageInfo = {0};
        unsigned char *pData = (unsigned char *)malloc(sizeof(unsigned char) * (g_nPayloadSize));
        cv::namedWindow("camera", CV_WINDOW_KEEPRATIO);
        cv::namedWindow("camera2", CV_WINDOW_KEEPRATIO);

        cout << "Give a headname" << endl;
        string name;
        cin >> name;
        name = "/home/dji/catkin_ws/src/opencv_exercise/pic/" + name;
        // cv::FileStorage fs(name, cv::FileStorage::WRITE);
        int count = 0;

        nRet = MV_CC_GetImageForBGR(handle, pData, g_nPayloadSize, &stImageInfo, 100);
        if(MV_OK != nRet)
        {
            printf("No data");
            std::free(pData);
            pData = NULL;
            break;
        }
        cv::Size imageSize;
        imageSize.height = stImageInfo.nHeight;
        imageSize.width = stImageInfo.nWidth;
        
        cv::Mat view, rview, map1, map2;
        cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(),
        cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
        imageSize, CV_16SC2, map1, map2);

        while(1)
        {
            memset(&stImageInfo, 0, sizeof(MV_FRAME_OUT_INFO_EX));
            if(pData == NULL)
            {
                printf("Allocate memory failed.\n");
                break;
            }
            
            nRet = MV_CC_GetImageForBGR(handle, pData, g_nPayloadSize, &stImageInfo, 100);
            if(MV_OK != nRet)
            {
                printf("No data");
                std::free(pData);
                pData = NULL;
                break;
            }

            cv::Mat srcImage, calibration;
            srcImage = cv::Mat(stImageInfo.nHeight, stImageInfo.nWidth, CV_8UC3, pData);

            if(is_undistorted)
            {
                imshow("camera", srcImage);
                remap(srcImage, srcImage, map1, map2, cv::INTER_LINEAR);
                imshow("camera2", srcImage);
            }
            else
            {
                cv::imshow("camera", srcImage);
            }
            


            

            unsigned char c = cvWaitKey(50);
            if(c == 'q')
            {
                break;
            }
            else if(c == 's')
            {
                // fs << ("n" + to_string(count)) << srcImage;
                if(cvWaitKey(0) == 's')
                {
                    cv::imwrite(name+to_string(count)+".bmp", srcImage);
                    count++;
                }  
            }
            
            srcImage.release();
            
        }
        // fs.release();
        cv::destroyWindow("window");
        free(pData);
        nRet = MV_CC_StopGrabbing(handle);
        nRet = MV_CC_CloseDevice(handle);
        nRet = MV_CC_DestroyHandle(handle);
        break;
    }

    if(nRet != MV_OK)
    {
        if(handle != NULL)
        {
            MV_CC_DestroyHandle(handle);
            handle = NULL;
        }
    }

    return 0;
}




