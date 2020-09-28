#include <ros/ros.h>
#include <string>
#include <std_msgs/String.h>
#include <QDebug>
#include "qnode.h"

using namespace std;

QNode::QNode(int argc, char **argv):init_argc(argc), init_argv(argv){}

void QNode::lidarCbk(const find_corner::CustomMsg::ConstPtr &msg)
{
    pcl::PointCloud<PointType>::Ptr p(new pcl::PointCloud<PointType>);
    p->is_dense = false;
    p->height = 1;
    PointType po;
    for(uint i = 0; i < msg->point_num; i++)
    {
        if(msg->points[i].x > 1)
        {
            po.x = msg->points[i].x;
            po.y = msg->points[i].y;
            po.z = msg->points[i].z;
            int f = msg->points[i].reflectivity;
            int r, g, b;
            if(f < 30)
            {
                // r=0; g=254-4*f; b=255;
                int green = f * 255 / 30;
                r=0; g=green&0xff; b=0xff;
            }
            else if(f < 90)
            {
                // r=0; g=4*f-254; b=510-4*f;
                int blue = (90-f)*255/60;
                r=0; g=0xff; b=blue&0xff;
            }
            else if(f < 150)
            {
                // r=4*f-510; g=255; b=0;
                int red = (f-90)*255/60;
                r=red&0xff; g=0xff; b=0;
            }
            else
            {
                // r=255; g=1022-4*f; b=0;
                int green = (255-f)/(256-150);
                r=0xff; g=green&0xff; b = 0;
            }
            po.r = r;
            po.g = g;
            po.b = b;
            p->push_back(po);
        }
    }
    if(p->size() > 0)
    {
        clouds.push_back(p);
        Q_EMIT update_pointcloud();
    }
}

bool QNode::init()
{
    quit_flag = true;
    ros::init(init_argc, init_argv, "qt_findcorner");
    ros::start();
    ros::NodeHandle n;
    pub = n.advertise<std_msgs::String>("chatter", 1000);
    sub_custom = n.subscribe("/livox/lidar", 1000, &QNode::lidarCbk, this);

    start();
    return true;
}

void QNode::run()
{
    ros::Rate loop_rate(1);
    while(quit_flag && ros::ok())
    {
        ros::spinOnce();
    }
    qDebug() << "Done";
    Q_EMIT rosShutdown();
}

void QNode::qpub()
{
    std_msgs::String msg;
    string ss = "world";
    msg.data = ss;
    pub.publish(msg);
    ros::spinOnce();
    qDebug() << "world";
}