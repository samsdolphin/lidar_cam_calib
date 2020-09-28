#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "qnode.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <vtkRenderWindow.h>

namespace Ui 
{
    class MainWindow;
}

class MainWindow : public QMainWindow
{
Q_OBJECT
public:
    explicit MainWindow(int argc, char **argv, QWidget *parent = 0);
    ~MainWindow();
    void closeEvent(QCloseEvent *event);
    void update_cube();

protected:
    pcl::visualization::PCLVisualizer::Ptr viewer;
    pcl::PointCloud<PointType>::Ptr cloud;

private slots:
    void myupdate();
    void on_doubleSpinBox_valueChanged(double arg1);
    void on_doubleSpinBox_2_valueChanged(double arg1);
    void on_doubleSpinBox_3_valueChanged(double arg1);
    void on_doubleSpinBox_4_valueChanged(double arg1);
    void on_doubleSpinBox_5_valueChanged(double arg1);
    void on_spinBox_valueChanged(int arg1);

private:
    Ui::MainWindow *ui;
    QNode qnode;
    ros::Subscriber sub_custom;
    double cube_x, cube_y, cube_z;
    int lineWidth;
};

#endif // MAINWINDOW_H