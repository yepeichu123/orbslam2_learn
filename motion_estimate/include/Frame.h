#ifndef FRAME_H
#define FRAME_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

namespace XIAOC {
    /*
    struct MyPoint3d {
        int id_;
        cv::KeyPoint kpt_;
        cv::Mat desp_;
        cv::Point3f p3d_;
        cv::Point2f p2d_;
    };
    */
    class Frame {
    public:
        // 默认构造函数
        Frame();

        // 赋值构造函数
        Frame(int id, cv::Mat& rgbImg, cv::Mat& depthImg, 
                std::vector<cv::KeyPoint>& kpt, cv::Mat& desp );

        // 默认析构函数
        ~Frame() {}

        // 设置相机内参数
        void SetCameraIntrinsic(cv::Mat& K);

        void SetCameraIntrinsic(const double& fx, const double& fy, 
            const double& cx, const double& cy);

        // 设置图像
        void SetImg(cv::Mat& rgbImg, cv::Mat& depthImg);

        // 设置特征点
        void SetKeyPoints(std::vector<cv::KeyPoint>& kpt);

        // 设置描述子
        void SetDescriptors(cv::Mat& desp);

        // 计算匹配点的三维信息
        void ComputePoint3d(std::vector<cv::DMatch>& matches, bool queryIdx);

        // 获取所有匹配点的三维信息
        // std::vector<XIAOC::MyPoint3d>& GetAllPoint3d() {
            // return mvMP3d_;
        // }

        // 获取点的深度值
        double GetPointDepth(double u, double v);

        // 获取所有描述子
        cv::Mat GetAllDescriptors() {
            return mDesp_.clone();
        }

        // 获取指定id的描述子
        cv::Mat GetDescriptor(int id) {
            return mDesp_.row(id).clone();
        }

        // 获取所有特征点
        std::vector<cv::KeyPoint>& GetAllKeyPoints() {
            return mvKpt_;
        }
        
        // 获取指定id的特征点
        cv::KeyPoint& GetKeyPoint(int id) {
            return mvKpt_[id];
        }

        cv::Mat GetImg() {
            return mImgRgb_.clone();
        }

    private:
        // 图像ID
        int mId_;

        // 图像
        cv::Mat mImgRgb_;
        cv::Mat mImgDepth_;

        // 图像特征点
        std::vector<cv::KeyPoint> mvKpt_;

        // 图像描述子
        cv::Mat mDesp_;

        // 三维点
        // std::vector<cv::Point3f> mvP3d_;
        // std::vector<XIAOC::MyPoint3d> mvMP3d_;

        // 相机内参数
        double mFx_, mFy_, mCx_, mCy_;
    };
}

#endif // FRAME_H