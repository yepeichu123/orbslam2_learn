#ifndef G2O_TYPE_H
#define G2O_TYPE_H

// eigen
#include <Eigen/Core>
#include <Eigen/StdVector>

// sophus
#include "sophus/se3.h"
#include "sophus/so3.h"

// g2o
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>

// opencv
#include <opencv2/core/core.hpp>

typedef Sophus::SE3 SE3;
typedef Eigen::Matrix<double, 6, 1> Vec6;
typedef Eigen::Matrix<double, 3, 1> Vec3;
typedef Eigen::Matrix<double, 2, 1> Vec2;

using namespace g2o;

namespace XIAOC {
    
    // 定义6自由度的姿态
    class VertexPose: public BaseVertex<6, SE3> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        VertexPose(): BaseVertex<6, SE3>() {}

        bool read(std::istream& is) override { return true; }

        bool write(std::ostream& os) const override { return true; }

        // 姿态重置
        virtual void setToOriginImpl() override {
            _estimate = SE3();
        }

        // 更新姿态
        virtual void oplusImpl(const double *update_) override {
            Vec6 update;
            update << update_[0], update_[1], update_[2], update_[3], update_[4], update_[5];
            _estimate = SE3::exp(update) * _estimate;
        }

        // 获取姿态旋转
        inline Matrix3d R() const {
            return _estimate.so3().matrix();
        }

        // 获取姿态位移
        inline Vector3d t() const {
            return _estimate.translation();
        }
    };

    // 定义重投影单边
    class EdgeProjectPoseOnly : public BaseUnaryEdge<2, Vec2, VertexPose> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        // 相机内参和三维点初始化
        EdgeProjectPoseOnly(cv::Mat& K, Vec3& p3d) 
            : BaseUnaryEdge<2, Vec2, VertexPose>(), pr_(p3d) {
            fx_ = K.at<double>(0, 0);
            fy_ = K.at<double>(1, 1);
            cx_ = K.at<double>(0, 2);
            cy_ = K.at<double>(1, 2);
        }

        bool read(std::istream& is) override { return true; }

        bool write(std::ostream& os) const override { return true; }

        // 计算重投影误差
        virtual void computeError() override;

    public:
        bool depthValid = true;

    private:
        // 参考帧的三维点
        Vec3 pr_;

        // 相机内参
        double fx_ = 0, fy_ = 0, cx_ = 0, cy_ = 0;
    };

}

#endif // G2O_TYPE_H