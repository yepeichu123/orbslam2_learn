#ifndef FRAME_H
#define FRAME_H

#include <iostream>
#include <opencv2/core/core.hpp>

namespace XIAOC {
    class Frame {
    public:
        // 默认构造函数
        Frame(int id, cv::Mat& camera_intrinsic);

        // 拷贝构造函数
        Frame(const Frame& f);

        // **构造函数
        Frame(int id, std::vector<cv::KeyPoint>& kpt, cv::Mat& desp, 
            std::vector<double>& depth, cv::Mat& camera_intrinsic);

        // 默认析构函数
       ~Frame();

        // 计算地图点
        void ComputeMapPoints(std::vector<cv::DMatch>& matches, bool queryIdx);

        // 设置三维点
        void Set3DPointsAndIndex();

        // 获取当前帧id
        int GetFrameId() {
            return mnId_;
        }

        // 获取当前帧特征点
        std::vector<cv::KeyPoint>& GetKeyPoints() {
            return mvKpt_;
        }

        // 获取当前帧特征点对应的描述子
        cv::Mat GetDescriptors() {
            return mmDesp_;
        }

        // 获取当前帧特征点对应的深度值
        std::vector<double>& GetDepth() {
            return mvDepth_;
        }

        // 获取相机内参数
        cv::Mat GetCameraIntrinsic() {
            return mmK_;
        }

        // 获取地图点
        std::vector<cv::Point3d>& GetMapPoints() {
            return mvMapPoints_;
        }

        // 获取地图点对应索引
        std::vector<int>& GetMapPointsId() {
            return mvMapPointsId_;
        }

    private:
        // 当前帧id
        int mnId_;

        // 当前帧关键点
        std::vector<cv::KeyPoint> mvKpt_;

        // 对应特征点的描述子
        cv::Mat mmDesp_;

        // 对应特征点的深度值
        std::vector<double> mvDepth_;

        // 相机内参数
        cv::Mat mmK_;

        // 有效三维点,姑且视其为地图点(免得需要额外去构建个地图类)
        std::vector<cv::Point3d> mvMapPoints_;

        // 对应有效地图点的索引
        std::vector<int> mvMapPointsId_;
    };
}

#endif // FRAME_H