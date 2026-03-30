/*
1.直接点击打印map系坐标
2.按住ctrl再点击,绘制点击点,按z撤销上一个点
3.按住空格点击,发布模拟自瞄目标,按t切换是否追踪,左右键切换装甲板类型
4.按住c点击,绘制点击点关于中心点的对称点
5.按住a、d点击，发布模拟costmap障碍 已删除
6.按住r点击圆中心点，释放时确定半径
7.按下i键，读取并加载点列表
*/
#ifndef WIDGET_H
#define WIDGET_H
#include <yaml-cpp/yaml.h>
#include <QWidget>
#include <QPixmap>
#include <QLabel>
#include <QMouseEvent>
#include <iostream>
#include <QCoreApplication>
#include <QImage>
#include <QDebug>
#include <QFile>
#include <qtimer.h>
#include <QStringList>
#include <QPainter>
#include <iomanip>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

QT_BEGIN_NAMESPACE
namespace Ui
{
    class Widget;
}
QT_END_NAMESPACE

class Widget : public QWidget
{
    Q_OBJECT

public:
    Widget(QWidget *parent = nullptr);
    ~Widget();
    void loadImage(const QString &filePath);
    double target_x;
    double target_y;
    double resolution;
    double origin_x;
    double origin_y;
    double img_w;
    double img_h;
    double bias_x;
    double bias_y;
    int k;
    struct point_real
    {
        double x_real;
        double y_real;
        QPoint point_pix;
        int count;
    };
public slots:
    void spinOnce();

protected:
    void mousePressEvent(QMouseEvent *event) override; // 捕获鼠标点击事件
    void mouseMoveEvent(QMouseEvent *event) override;  // 捕获鼠标释放事件
    void keyPressEvent(QKeyEvent *event) override;     // 捕获键盘按下事件
    void keyReleaseEvent(QKeyEvent *event) override;   // 捕获键盘释放事件
    void paintEvent(QPaintEvent *event) override;      // 绘制事件
    void updatePixmap();

private:
    void init_auto_aim_target();
    void publish_auto_aim_target();
    void publish_temp_costmap(int mode, double x, double y);
    void pixmap_to_real(const QPoint &point, point_real &real_point);
    void real_to_pixmap(point_real &real_point);
    Ui::Widget *ui;
    QTimer *ros_timer;
    QLabel *imageLabel;
    bool ctrlPressed_ = false; // 是否按住 Ctrl 键
    bool spacePressed_ = false;
    bool cPressed_ = false;
    bool aPressed_ = false;
    bool dPressed_ = false;
    bool rPressed_ = false;
    point_real circle_point_;
    double radius_pix_, radius_real_;
    std::vector<point_real> points;
    point_real center_point;
    QPixmap pixmap;
    std::shared_ptr<rclcpp::Node> node_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr auto_aim_target_publisher_;
    std::mutex mutex_;
    struct auto_aim_target
    {
        double x;
        double y;
        bool tracked;
        int armor_type;
    } auto_aim_target_;

    /* ---------------------------------- 绘制点列表 --------------------------------- */
    std::vector<point_real> point_red_;
    std::vector<point_real> point_blue_;
    point_real point_center_;
    bool show_point_list_ = false; // 是否显示点列表
};
#endif // WIDGET_H
