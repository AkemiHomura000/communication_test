#include <iostream>
#include <string>
#include <vector>
#include <chrono>

// PCL
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

// Scancontext
#include "module/Scancontext/Scancontext.h"

using namespace std;

/**
 * @brief 将 Eigen 矩阵转换为 OpenCV Mat 以进行可视化
 */
cv::Mat visualizeScancontext(Eigen::MatrixXd _sc) {
    int rows = _sc.rows();
    int cols = _sc.cols();
    
    cv::Mat dst(rows, cols, CV_8UC1);
    
    // 归一化到 0-255 以便显示 (Scancontext 通常存储高度值)
    double min_val, max_val;
    max_val = _sc.maxCoeff();
    min_val = _sc.minCoeff();
    
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            float val = _sc(r, c);
            // 线性映射
            float norm_val = (max_val - min_val) > 1e-6 ? (val - min_val) / (max_val - min_val) : 0;
            dst.at<uint8_t>(r, c) = static_cast<uint8_t>(norm_val * 255);
        }
    }
    
    // 应用伪彩色增强可视化效果
    cv::Mat color_mapped;
    cv::applyColorMap(dst, color_mapped, cv::COLORMAP_JET);
    
    // 放大图像以便观察 (Scancontext 默认通常是 20x60)
    cv::Mat resized;
    cv::resize(color_mapped, resized, cv::Size(600, 200), 0, 0, cv::INTER_NEAREST);
    
    return resized;
}

/**
 * @brief 将矩形的 Scancontext 转换为极坐标下的圆形可视化
 */
cv::Mat visualizeScancontextPolar(Eigen::MatrixXd _sc) {
    int num_ring = _sc.rows();
    int num_sector = _sc.cols();
    
    // 1. 先生成基础的归一化矩阵 (和之前类似)
    cv::Mat sc_base(num_ring, num_sector, CV_8UC1);
    double min_val, max_val;
    max_val = _sc.maxCoeff();
    min_val = _sc.minCoeff();
    
    for (int r = 0; r < num_ring; r++) {
        for (int c = 0; c < num_sector; c++) {
            float val = _sc(r, c);
            float norm_val = (max_val - min_val) > 1e-6 ? (val - min_val) / (max_val - min_val) : 0;
            sc_base.at<uint8_t>(r, c) = static_cast<uint8_t>(norm_val * 255);
        }
    }

    // 2. 转换为极坐标圆形
    // 设定输出图像大小
    int out_size = 800;
    cv::Mat polar_img(out_size, out_size, CV_8UC1, cv::Scalar(0));
    cv::Point2f center(out_size / 2.0f, out_size / 2.0f);
    double max_radius = out_size / 2.0;

    // 使用 WarpPolar 的逆变换：LinearPolar
    // 注意：Scancontext 的行(row)是半径，列(col)是角度
    // OpenCV 的 LinearPolar 期望输入是 (角度 x 半径)，所以我们需要转置并调整
    cv::Mat sc_base_resized;
    cv::resize(sc_base, sc_base_resized, cv::Size(num_sector, num_ring), 0, 0, cv::INTER_LINEAR);

    // 映射到圆形区域
    cv::warpPolar(sc_base_resized, polar_img, polar_img.size(), center, max_radius, 
                  cv::INTER_LINEAR + cv::WARP_INVERSE_MAP + cv::WARP_POLAR_LINEAR);

    // 3. 应用伪彩色
    cv::Mat color_mapped;
    cv::applyColorMap(polar_img, color_mapped, cv::COLORMAP_JET);
    
    // 画一些辅助圆圈以便观察
    for(int i=1; i<=4; ++i) {
        cv::circle(color_mapped, center, max_radius * i / 4.0, cv::Scalar(255, 255, 255), 1);
    }

    return color_mapped;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cout << "Usage: ./scancontext_demo /path/to/cloud1.pcd /path/to/cloud2.pcd" << endl;
        return -1;
    }

    string pcd_path1 = argv[1];
    string pcd_path2 = argv[2];
    
    // 1. 读取两个 PCD 文件
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud2(new pcl::PointCloud<pcl::PointXYZI>);
    
    if (pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_path1, *cloud1) == -1 || 
        pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_path2, *cloud2) == -1) {
        cerr << "Couldn't read PCD files." << endl;
        return -1;
    }
    cout << "Loaded cloud1: " << cloud1->size() << " points." << endl;
    cout << "Loaded cloud2: " << cloud2->size() << " points." << endl;

    auto start_total = chrono::steady_clock::now();

    // 2. 初始化 Scancontext 管理器
    SCManager sc_manager(20, 30, 30.0, 0.25);
    
    // 3. 分别生成 Scancontext 描述子
    auto start_sc1 = chrono::steady_clock::now();
    Eigen::MatrixXd sc1 = sc_manager.makeScancontext(*cloud1);
    auto end_sc1 = chrono::steady_clock::now();

    auto start_sc2 = chrono::steady_clock::now();
    Eigen::MatrixXd sc2 = sc_manager.makeScancontext(*cloud2);
    auto end_sc2 = chrono::steady_clock::now();
    
    // 4. 计算匹配结果
    auto start_match = chrono::steady_clock::now();
    auto result = sc_manager.distanceBtnScanContext(sc1, sc2);
    auto end_match = chrono::steady_clock::now();

    double distance = result.first;
    int argmin_yaw = result.second;
    
    // 转换 yaw 偏置为角度
    double yaw_diff_deg = argmin_yaw * (360.0 / sc_manager.PC_NUM_SECTOR);
    if (yaw_diff_deg > 180.0) yaw_diff_deg -= 360.0;

    cout << "\n--- Matching Results ---" << endl;
    cout << "Scancontext Distance: " << distance << endl;
    cout << "Optimal Yaw Shift (index): " << argmin_yaw << endl;
    cout << "Estimated Yaw Difference: " << yaw_diff_deg << " degrees" << endl;
    cout << "Matching Status: " << (distance < sc_manager.SC_DIST_THRES ? "SUCCESS (Loop Found)" : "FAILED (Too different)") << endl;

    auto end_total = chrono::steady_clock::now();

    // 耗时统计输出
    auto diff_sc1 = chrono::duration_cast<chrono::microseconds>(end_sc1 - start_sc1).count() / 1000.0;
    auto diff_sc2 = chrono::duration_cast<chrono::microseconds>(end_sc2 - start_sc2).count() / 1000.0;
    auto diff_match = chrono::duration_cast<chrono::microseconds>(end_match - start_match).count() / 1000.0;
    auto diff_total = chrono::duration_cast<chrono::microseconds>(end_total - start_total).count() / 1000.0;

    cout << "\n--- Time Cost Profiling ---" << endl;
    cout << "Make Scancontext 1: " << diff_sc1 << " ms" << endl;
    cout << "Make Scancontext 2: " << diff_sc2 << " ms" << endl;
    cout << "Matching Process:   " << diff_match << " ms" << endl;
    cout << "Total Pure Process: " << diff_total << " ms" << endl;

    // 5. 可视化
    cv::Mat sc_img1 = visualizeScancontext(sc1);
    cv::Mat sc_img2 = visualizeScancontext(sc2);
    cv::Mat polar_img1 = visualizeScancontextPolar(sc1);
    cv::Mat polar_img2 = visualizeScancontextPolar(sc2);
    
    // 将两张图并排显示以便对比
    cv::Mat combined_sc, combined_polar;
    cv::hconcat(sc_img1, sc_img2, combined_sc);
    cv::hconcat(polar_img1, polar_img2, combined_polar);

    cv::imshow("Scancontext Comparison (Left: Cloud1, Right: Cloud2)", combined_sc);
    cv::imshow("Polar Comparison (Left: Cloud1, Right: Cloud2)", combined_polar);
    
    cout << "\nPress any key to exit." << endl;
    cv::waitKey(0);

    return 0;
}
