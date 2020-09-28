#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QtGui>
#include "qnode.h"
#include <ros/ros.h>
#include <string>
#include <std_msgs/String.h>

MainWindow::MainWindow(int argc, char** argv, QWidget *parent):
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    qnode(argc, argv)
{
    ui->setupUi(this);
    lineWidth = ui->spinBox->value();

    cloud.reset(new pcl::PointCloud<PointType>);
    cloud->points.resize(200);

    for(std::size_t i = 0; i < cloud->points.size(); ++i)
    {
        cloud->points[i].x = 1024 * rand () / (RAND_MAX + 1.0f);
        cloud->points[i].y = 1024 * rand () / (RAND_MAX + 1.0f);
        cloud->points[i].z = 1024 * rand () / (RAND_MAX + 1.0f);
        
        cloud->points[i].r = 255 *(1024 * rand () / (RAND_MAX + 1.0f));
        cloud->points[i].g = 255 *(1024 * rand () / (RAND_MAX + 1.0f));
        cloud->points[i].b = 255 *(1024 * rand () / (RAND_MAX + 1.0f));
    }
    viewer.reset(new pcl::visualization::PCLVisualizer("viewer", false));
    ui->qvtkWidget->SetRenderWindow (viewer->getRenderWindow());
    viewer->setupInteractor(ui->qvtkWidget->GetInteractor(), ui->qvtkWidget->GetRenderWindow());
    ui->qvtkWidget->update();

    viewer->addCoordinateSystem(1.0);    

    ui->qvtkWidget->update();
    viewer->resetCamera();
    ui->qvtkWidget->update();

    qnode.init();
    connect(&qnode, SIGNAL(rosShutdown()), this, SLOT(close()));
    connect(&qnode, SIGNAL(update_pointcloud()), this, SLOT(myupdate()));
    ui->doubleSpinBox->setSingleStep(ui->doubleSpinBox_4->value());
    ui->doubleSpinBox_2->setSingleStep(ui->doubleSpinBox_4->value());
    ui->doubleSpinBox_3->setSingleStep(ui->doubleSpinBox_4->value());

    cube_x = 0; cube_y = 0; cube_z = 0;
    update_cube();
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    qnode.quit_flag = false;
    qnode.wait();
    event->accept();
}

void MainWindow::myupdate()
{
    string name = "cloud";
    name = name + to_string(qnode.clouds.size());
    viewer->addPointCloud(qnode.clouds[qnode.clouds.size()-1], name);
    ui->qvtkWidget->update();
}


void MainWindow::on_doubleSpinBox_valueChanged(double arg1)
{
    cube_x = arg1;
    update_cube();
}

void MainWindow::on_doubleSpinBox_2_valueChanged(double arg1)
{
    cube_y = arg1;
    update_cube();
}

void MainWindow::on_doubleSpinBox_3_valueChanged(double arg1)
{
    cube_z = arg1;
    update_cube();
}

void MainWindow::update_cube()
{
    viewer->removeAllShapes();
    double len = ui->doubleSpinBox_5->value() / 2;

    viewer->addCube(cube_x-len, cube_x+len, cube_y-len, cube_y+len, cube_z-len, cube_z+len);
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
                                        pcl::visualization::PCL_VISUALIZER_REPRESENTATION_SURFACE,"cube");
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, lineWidth, "cube");
    ui->qvtkWidget->update();
}

void MainWindow::on_doubleSpinBox_4_valueChanged(double arg1)
{
    ui->doubleSpinBox->setSingleStep(arg1);
    ui->doubleSpinBox_2->setSingleStep(arg1);
    ui->doubleSpinBox_3->setSingleStep(arg1);
}

void MainWindow::on_doubleSpinBox_5_valueChanged(double arg1)
{
    update_cube();
}

void MainWindow::on_spinBox_valueChanged(int arg1)
{
    lineWidth = ui->spinBox->value();
    update_cube();
}