#ifndef QNODE_H
#define QNODE_H

#include <ros/ros.h>
#include <string>
#include <QThread>
#include <QStringListModel>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <find_corner/CustomMsg.h>

typedef pcl::PointXYZRGB PointType;
using namespace std;

class QNode : public QThread
{
Q_OBJECT
public:
    QNode(int argc, char **argv);
    bool init();
    void run();
    void qpub();
    bool quit_flag;
    void lidarCbk(const find_corner::CustomMsg::ConstPtr &msg);
    vector<pcl::PointCloud<PointType>::Ptr> clouds;

Q_SIGNALS:
    void rosShutdown();
    void update_pointcloud();

private:
    int init_argc;
    char **init_argv;
    ros::Publisher pub;
    ros::Subscriber sub_custom;
};

#endif // QNODE_H