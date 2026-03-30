#include "widget.h"
#include "./ui_widget.h"
#define DELTA_X 0 // 边框10像素
#define DELTA_Y 0
Widget::Widget(QWidget *parent)
    : QWidget(parent), ui(new Ui::Widget)
{
    ui->setupUi(this);
    node_ = std::make_shared<rclcpp::Node>("get_coordinate");
    auto_aim_target_publisher_ = node_->create_publisher<std_msgs::msg::String>("auto_aim_target_pos", 10);
    std::string config_file;
    node_->declare_parameter<std::string>("config_file", "");
    node_->get_parameter("config_file", config_file);
    const auto config_all = YAML::LoadFile(config_file);
    std::string yaml_path = config_all["map_yaml"].as<std::string>();
    std::string map_path = config_all["map_pgm"].as<std::string>();
    std::string point_list_path = config_all["point_list"].as<std::string>();
    k = config_all["scale"].as<double>();

    YAML::Node config = YAML::LoadFile(yaml_path);
    resolution = config["resolution"].as<double>();
    YAML::Node origin_node = config["origin"];
    origin_x = origin_node[0].as<double>();
    origin_y = origin_node[1].as<double>();
    QVBoxLayout *layout = new QVBoxLayout(this);
    imageLabel = new QLabel;
    layout->addWidget(imageLabel);
    QString qmap_path = QString::fromStdString(map_path);
    pixmap = QPixmap(qmap_path);
    img_w = pixmap.size().rwidth();
    img_h = pixmap.size().rheight();
    // 绘制坐标系
    double ox = abs(origin_x) / resolution;
    double oy = img_h - abs(origin_y) / resolution;
    QPainter painter(&pixmap);
    // painter.begin(this);
    painter.setPen(QPen(Qt::red, 1));
    painter.drawLine(ox, oy, ox + 50, oy);
    painter.setPen(QPen(Qt::blue, 1));
    painter.drawLine(ox, oy, ox, oy - 50);
    painter.end();

    bias_x = DELTA_X + k * abs(origin_x) / resolution; // 边框10像素
    bias_y = DELTA_Y + k * (img_h - abs(origin_y) / resolution);
    pixmap = pixmap.scaled(pixmap.width() * k, pixmap.height() * k, Qt::KeepAspectRatio);
    imageLabel->setPixmap(pixmap);
    resize(pixmap.size());
    setWindowTitle("Map");

    /* --------------------------------  读取点列表 -------------------------------- */
    YAML::Node point_list = YAML::LoadFile(point_list_path);
    point_center_.x_real = point_list["center_x"].as<double>();
    point_center_.y_real = point_list["center_y"].as<double>();
    for (const auto &point : point_list["points"])
    {
        point_real p;
        p.x_real = point["x"].as<double>();
        p.y_real = point["y"].as<double>();
        real_to_pixmap(p);
        // std::cout<< "point_pix: " << p.point_pix.x() << ", " << p.point_pix.y() << std::endl;
        if (point_list["base"].as<std::string>() == "red")
        {
            point_red_.push_back(p);
        }
        else if (point_list["base"].as<std::string>() == "blue")
        {
            point_blue_.push_back(p);
        }
        // 计算对称方
        p.x_real = 2 * point_center_.x_real - p.x_real;
        p.y_real = 2 * point_center_.y_real - p.y_real;
        real_to_pixmap(p);
        if (point_list["base"].as<std::string>() == "red")
        {
            point_blue_.push_back(p);
        }
        else if (point_list["base"].as<std::string>() == "blue")
        {
            point_red_.push_back(p);
        }
    }
    /* --------------------------------  读取点列表 -------------------------------- */

    init_auto_aim_target();
    ros_timer = new QTimer(this);
    connect(ros_timer, SIGNAL(timeout()), this, SLOT(spinOnce()));
    ros_timer->start(25); // set the rate to 25ms  You can change this if you want to increase/decrease update rate
}
Widget::~Widget()
{
    delete ui;
}
void Widget::spinOnce()
{
    publish_auto_aim_target();
}
void Widget::init_auto_aim_target()
{
    std::lock_guard<std::mutex> lock(mutex_);
    target_x = 0;
    target_y = 0;
    auto_aim_target_.x = 0.0;
    auto_aim_target_.y = 0.0;
    auto_aim_target_.tracked = 0;
    auto_aim_target_.armor_type = 0;
}
void Widget::publish_auto_aim_target()
{
    std::lock_guard<std::mutex> lock(mutex_);
    std_msgs::msg::String msg;
    double real_target_x = auto_aim_target_.x + (rand() % 200) / 1000.0;
    double real_target_y = auto_aim_target_.y + (rand() % 200) / 1000.0;
    std::string str = std::to_string(real_target_x) + "," + std::to_string(real_target_y) + ",";
    str += std::to_string(auto_aim_target_.tracked) + "," + std::to_string(auto_aim_target_.armor_type);
    msg.data = str;
    auto_aim_target_publisher_->publish(msg);
}

void Widget::pixmap_to_real(const QPoint &point, point_real &real_point)
{
    // 将像素坐标转换为实际坐标
    real_point.x_real = (static_cast<double>(point.x()) - bias_x) * resolution / k;
    real_point.y_real = (-static_cast<double>(point.y()) + bias_y) * resolution / k;
}
void Widget::real_to_pixmap(point_real &real_point)
{
    int x = static_cast<int>(real_point.x_real * k / resolution + bias_x);
    int y = static_cast<int>(-real_point.y_real * k / resolution + bias_y);
    real_point.point_pix = QPoint(x, y);
    // 将实际坐标转换为像素坐标
}
void Widget::mousePressEvent(QMouseEvent *event)
{
    QPoint clickPos = event->pos();
    clickPos -= QPoint(10, 10);
    if (ctrlPressed_)
    {
        // 添加点到列表中
        point_real point;
        QPoint click(clickPos.x(), clickPos.y());
        point.x_real = (static_cast<double>(clickPos.x()) - bias_x) * resolution / k;
        point.y_real = (-static_cast<double>(clickPos.y()) + bias_y) * resolution / k;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "x_real:" << point.x_real << "  y_real:" << point.y_real << std::endl;
        point.point_pix = click;
        points.push_back(point);
        updatePixmap();
        update(); // 刷新窗口
    }
    else if (spacePressed_)
    {
        int x = clickPos.x();
        int y = clickPos.y();
        target_x = x;
        target_y = y;
        double x_double = (static_cast<double>(x) - bias_x) * resolution / k;
        double y_double = (-static_cast<double>(y) + bias_y) * resolution / k;
        std::lock_guard<std::mutex> lock(mutex_);
        auto_aim_target_.x = x_double;
        auto_aim_target_.y = y_double;
        updatePixmap();
        update(); // 刷新窗口
    }
    else if (cPressed_)
    {
        int x = clickPos.x();
        int y = clickPos.y();
        double x_double = (static_cast<double>(x) - bias_x) * resolution / k;
        double y_double = (-static_cast<double>(y) + bias_y) * resolution / k;
        center_point.x_real = x_double;
        center_point.y_real = y_double;
        center_point.point_pix = QPoint(x, y);
        updatePixmap();
        update(); // 刷新窗口
    }
    else if (rPressed_)
    {
        int x = clickPos.x();
        int y = clickPos.y();
        double x_double = (static_cast<double>(x) - bias_x) * resolution / k;
        double y_double = (-static_cast<double>(y) + bias_y) * resolution / k;
        circle_point_.point_pix = QPoint(x, y);
        circle_point_.x_real = x_double;
        circle_point_.y_real = y_double;
    }
    else
    {
        int x = clickPos.x();
        int y = clickPos.y();
        double x_double = (static_cast<double>(x) - bias_x) * resolution / k;
        double y_double = (-static_cast<double>(y) + bias_y) * resolution / k;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "点击坐标：" << x_double << "   " << y_double << std::endl;
    }
}
void Widget::mouseMoveEvent(QMouseEvent *event)
{
    if (rPressed_)
    {
        int x = event->pos().x() - 10;
        int y = event->pos().y() - 10;
        double x_double = (static_cast<double>(x) - bias_x) * resolution / k;
        double y_double = (-static_cast<double>(y) + bias_y) * resolution / k;
        radius_pix_ = sqrt(pow((circle_point_.point_pix.x() - (x)), 2) + pow((circle_point_.point_pix.y() - (y)), 2));
        radius_real_ = sqrt(pow((circle_point_.x_real - x_double), 2) + pow((circle_point_.y_real - y_double), 2));
        std::cout << "circle_radius:" << radius_real_ << std::endl;
        updatePixmap();
        update(); // 刷新窗口
    }
}
void Widget::keyPressEvent(QKeyEvent *event)
{
    if (event->key() == Qt::Key_Control)
    {
        ctrlPressed_ = true; // 标记 Ctrl 键被按下
    }
    else if (event->key() == Qt::Key_Z)
    {
        if (points.size() > 0)
        {
            points.pop_back(); // 移除最后一个点
            updatePixmap();
            update(); // 刷新窗口
        }
    }
    else if (event->key() == Qt::Key_Space)
    {
        spacePressed_ = true;
    }
    else if (event->key() == Qt::Key_T)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto_aim_target_.tracked = !auto_aim_target_.tracked;
    }
    else if (event->key() == Qt::Key_Left)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto_aim_target_.armor_type -= 1;
    }
    else if (event->key() == Qt::Key_Right)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto_aim_target_.armor_type += 1;
    }
    else if (event->key() == Qt::Key_C)
    {
        cPressed_ = true;
    }
    else if (event->key() == Qt::Key_A)
    {
        aPressed_ = true;
    }
    else if (event->key() == Qt::Key_D)
    {
        dPressed_ = true;
    }
    else if (event->key() == Qt::Key_R)
    {
        rPressed_ = true;
    }
    else if (event->key() == Qt::Key_I)
    {
        show_point_list_ = !show_point_list_;
        updatePixmap();
        update(); // 刷新窗口
    }
}

void Widget::keyReleaseEvent(QKeyEvent *event)
{
    if (event->key() == Qt::Key_Control)
    {
        ctrlPressed_ = false; // 标记 Ctrl 键被释放
    }
    else if (event->key() == Qt::Key_Space)
    {
        spacePressed_ = false;
    }
    else if (event->key() == Qt::Key_C)
    {
        cPressed_ = false;
    }
    else if (event->key() == Qt::Key_A)
    {
        aPressed_ = false;
    }
    else if (event->key() == Qt::Key_D)
    {
        dPressed_ = false;
    }
    else if (event->key() == Qt::Key_R)
    {
        rPressed_ = false;
    }
}

void Widget::paintEvent(QPaintEvent *event)
{
    QWidget::paintEvent(event); // 保持原有绘制逻辑
}
void Widget::updatePixmap()
{
    QPixmap pixmap_temp = pixmap;
    if (!pixmap_temp.isNull())
    {
        QPainter painter(&pixmap_temp);
        painter.setPen(QPen(Qt::yellow, 3));                  // 设置画笔颜色和宽度
        painter.setBrush(QBrush(Qt::blue, Qt::SolidPattern)); // 设置填充颜色
        painter.drawEllipse(target_x, target_y, 8, 8);        // 绘制点

        painter.setPen(QPen(Qt::black, 3));                                                  // 设置画笔颜色和宽度
        painter.setBrush(QBrush(Qt::green, Qt::SolidPattern));                               // 设置填充颜色
        painter.drawEllipse(center_point.point_pix.rx(), center_point.point_pix.ry(), 8, 8); // 绘制点

        for (const point_real point : points)
        {
            painter.setPen(QPen(Qt::green, 3));                   // 设置画笔颜色和宽度
            painter.setBrush(QBrush(Qt::blue, Qt::SolidPattern)); // 设置填充颜色
            painter.drawEllipse(point.point_pix, 5, 5);           // 绘制点
                                                                  // 绘制坐标文本，稍微偏移点的位置
            QString coordinateText = QString("(%1, %2)")
                                         .arg(point.x_real, 0, 'f', 2) // 格式化为小数点后3位
                                         .arg(point.y_real, 0, 'f', 2);
            painter.setPen(QPen(Qt::darkRed, 3));
            painter.drawText(point.point_pix + QPoint(10, -10), coordinateText); // 偏移文本位置
        }
        if (center_point.point_pix.rx() != 0 && center_point.point_pix.ry() != 0)
        {
            QString coordinateText = QString("(%1, %2)")
                                         .arg(center_point.x_real, 0, 'f', 2) // 格式化为小数点后3位
                                         .arg(center_point.y_real, 0, 'f', 2);
            painter.setPen(QPen(Qt::darkRed, 3));
            painter.drawText(center_point.point_pix + QPoint(10, -10), coordinateText); // 偏移文本位置

            if (cPressed_)
            {
                // 绘制points关于center_point的对称点
                for (const point_real point : points)
                {
                    double x = 2 * center_point.x_real - point.x_real;
                    double y = 2 * center_point.y_real - point.y_real;
                    QPoint point_pix = QPoint((2 * center_point.point_pix.rx() - point.point_pix.x()),
                                              (2 * center_point.point_pix.ry() - point.point_pix.y()));
                    painter.setPen(QPen(Qt::red, 3));                     // 设置画笔颜色和宽度
                    painter.setBrush(QBrush(Qt::blue, Qt::SolidPattern)); // 设置填充颜色
                    painter.drawEllipse(point_pix, 5, 5);                 // 绘制点
                    QString coordinateText = QString("(%1, %2)")
                                                 .arg(x, 0, 'f', 2) // 格式化为小数点后3位
                                                 .arg(y, 0, 'f', 2);
                    painter.setPen(QPen(Qt::darkRed, 3));
                    painter.drawText(point_pix + QPoint(10, -10), coordinateText); // 偏移文本位置
                }
            }
        }
        // 画圆
        if (radius_pix_ > 0)
        {
            painter.setPen(QPen(Qt::red, 3));                     // 设置画笔颜色和宽度
            painter.setBrush(QBrush(Qt::blue, Qt::SolidPattern)); // 设置填充颜色
            painter.drawEllipse(circle_point_.point_pix, 5, 5);   // 绘制点
            QString coordinateText = QString("(%1, %2)")
                                         .arg(circle_point_.x_real, 0, 'f', 2) // 格式化为小数点后3位
                                         .arg(circle_point_.y_real, 0, 'f', 2);
            painter.setPen(QPen(Qt::darkRed, 3));
            painter.drawText(circle_point_.point_pix + QPoint(10, -10), coordinateText); // 偏移文本位置

            painter.setBrush(QBrush(Qt::blue, Qt::NoBrush));                                                            // 设置无填充
            painter.drawEllipse(circle_point_.point_pix, static_cast<int>(radius_pix_), static_cast<int>(radius_pix_)); // 绘制圆
            QString circleText = QString("r = %1")
                                     .arg(radius_real_, 0, 'f', 2);
            painter.setPen(QPen(Qt::darkRed, 3));
            painter.drawText(circle_point_.point_pix + QPoint(10, 10), circleText); // 偏移文本位置
        }
        if (show_point_list_)
        {
            // 绘制点列表
            painter.setPen(QPen(Qt::red, 3));                      // 设置画笔颜色和宽度
            painter.setBrush(QBrush(Qt::white, Qt::SolidPattern)); // 设置填充颜色
            for (const point_real &point : point_red_)
            {
                painter.drawEllipse(point.point_pix, 5, 5); // 绘制点
                QString pointText = QString(" (%1, %2)")
                                        .arg(point.x_real, 0, 'f', 2) // 格式化为小数点后3位
                                        .arg(point.y_real, 0, 'f', 2);
                // painter.drawText(point.point_pix + QPoint(10, -10), pointText); // 在左上角绘制点坐标
                QPoint text_pos = point.point_pix + QPoint(10, -10);
                // 设置字体（你可以替换为其他更清晰的字体）
                QFont font("Tahoma", 10, QFont::Bold);
                painter.setFont(font);
                QFontMetrics fm(font);
                QRect text_rect = fm.boundingRect(pointText);
                text_rect.moveTopLeft(text_pos);
                int padding = 3;
                text_rect.adjust(-padding, -padding, +padding, +padding);
                // 绘制背景框
                painter.setBrush(QBrush(QColor(255, 255, 255, 230))); // 半透明白底
                painter.setPen(QPen(Qt::black, 1));                   // 黑边框
                painter.drawRect(text_rect);
                // 绘制文字（居中对齐）
                painter.setPen(QPen(Qt::red));
                painter.drawText(text_rect, Qt::AlignLeft | Qt::AlignVCenter, pointText);
            }
            painter.setPen(QPen(Qt::blue, 3));                     // 设置画笔颜色和宽度
            painter.setBrush(QBrush(Qt::white, Qt::SolidPattern)); // 设置填充颜色
            for (const point_real &point : point_blue_)
            {
                painter.drawEllipse(point.point_pix, 5, 5); // 绘制点
                QString pointText = QString(" (%1, %2)")
                                        .arg(point.x_real, 0, 'f', 2) // 格式化为小数点后3位
                                        .arg(point.y_real, 0, 'f', 2);
                // painter.drawText(point.point_pix + QPoint(10, -10), pointText); // 在左上角绘制点坐标
                QPoint text_pos = point.point_pix + QPoint(10, -10);
                // 设置字体（你可以替换为其他更清晰的字体）
                QFont font("Tahoma", 10, QFont::Bold);
                painter.setFont(font);
                QFontMetrics fm(font);
                QRect text_rect = fm.boundingRect(pointText);
                text_rect.moveTopLeft(text_pos);
                int padding = 3;
                text_rect.adjust(-padding, -padding, +padding, +padding);
                // 绘制背景框
                painter.setBrush(QBrush(QColor(255, 255, 255, 230))); // 半透明白底
                painter.setPen(QPen(Qt::black, 1));                   // 黑边框
                painter.drawRect(text_rect);
                // 绘制文字（居中对齐）
                painter.setPen(QPen(Qt::blue));
                painter.drawText(text_rect, Qt::AlignLeft | Qt::AlignVCenter, pointText);
            }
        }
        imageLabel->setPixmap(pixmap_temp);
    }
}
