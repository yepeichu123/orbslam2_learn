#ifndef MOTION_ESTIMATE_H
#define MOTION_ESTIMATE_H

#include "Frame.h"
#include "G2OType.h"

// eigen
#include <Eigen/Core>
#include <Eigen/StdVector>

// sophus
#include "sophus/se3.h"
#include "sophus/so3.h"

// opencv 
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

// g2o
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/linear_solver_eigen.h>

// c++
#include <iostream>

namespace XIAOC {

    class Frame;

    struct MyPoint3d;

    class MotionEstimate {

    public:
        // 默认构造函数
        MotionEstimate(cv::Mat& K);

        // 默认析构函数
        ~MotionEstimate() {}

        // 基于参考帧的运动估计
        bool MotionEstimateByRefFrame(std::vector<cv::Point3d>& refP3d, std::vector<cv::Point2d>& currP2d, 
            cv::Mat& R, cv::Mat& t, cv::Mat& inliers);

        // 基于运动模型的运动估计
        bool MotionEstimateByModel(std::vector<cv::Point3d>& refP3d, std::vector<cv::Point2d>& currP2d, 
            cv::Mat& R_in, cv::Mat& t_in, cv::Mat& R_out, cv::Mat& t_out);

        void ComputeError(std::vector<cv::Point3d>& refP3d, 
            std::vector<cv::Point2d>& currP2d, cv::Mat& R_in, cv::Mat& t_in);

    private:
        // 调用opencv的pnp求解初值
        bool ComputePoseByPnp(std::vector<cv::Point3d>& vp3d, std::vector<cv::Point2d>& vp2d, 
            cv::Mat& R_out, cv::Mat& t_out, cv::Mat& inliers);

        // 利用g2o优化相机姿态
        bool OptimizePoseByG2O(std::vector<cv::Point3d>& refP3d, std::vector<cv::Point2d>& currP2d,
            cv::Mat& R_in, cv::Mat& t_in, cv::Mat& R_out, cv::Mat& t_out);

        // 相机内参
        double mFx_, mFy_, mCx_, mCy_;

    };
}


#endif // MOTION_ESTIMATE_H