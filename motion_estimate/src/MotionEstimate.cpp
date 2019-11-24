#include "MotionEstimate.h"

// 默认构造函数
XIAOC::MotionEstimate::MotionEstimate(cv::Mat& K) {
    mFx_ = K.at<float>(0, 0);
    mFy_ = K.at<float>(1, 1);
    mCx_ = K.at<float>(0, 2);
    mCy_ = K.at<float>(1, 2);
}

// 基于参考帧的运动估计
bool XIAOC::MotionEstimate::MotionEstimateByRefFrame(std::vector<cv::Point3f>& refP3d, 
            std::vector<cv::Point2f>& currP2d, cv::Mat& R_out, cv::Mat& t_out) {
    std::cout << "Enter MotionEstimate by Reference frame!" << std::endl;
    // 利用opencv提供的pnp函数计算姿态初值
    cv::Mat R_in, t_in;
    if (ComputePoseByPnp(refP3d, currP2d, R_in, t_in)) {
        // 利用g2o优化姿态
        if (OptimizePoseByG2O(refP3d, currP2d, R_in, t_in, R_out, t_out)) {
            std::cout << "Found the best pose by reference frame!" << std::endl;
            return true;
        }
    }
    std::cout << "Cannot find the best pose by reference frame!" << std::endl;
    return false;
}

// 基于运动模型的运动估计
bool XIAOC::MotionEstimate::MotionEstimateByModel(std::vector<cv::Point3f>& refP3d, std::vector<cv::Point2f>& currP2d, 
            cv::Mat& R_in, cv::Mat& t_in, cv::Mat& R_out, cv::Mat& t_out) {
    std::cout << "Enter MotionEstimate by motion model!" << std::endl;
    // 进行非线性优化
    if (OptimizePoseByG2O(refP3d, currP2d, R_in, t_in, R_out, t_out)) {
        std::cout << "Found the best pose by motion model!" << std::endl;
        return true;
    }
    std::cout << "Cannot find the best pose by motion model!" << std::endl;
    return false;
}

void XIAOC::MotionEstimate::ComputeError(std::vector<cv::Point3f>& refP3d, 
            std::vector<cv::Point2f>& currP2d, cv::Mat& R_in, cv::Mat& t_in) {
    double errTotal = 0;
    double errAvg = 0;
    int count = 0;
    for (int i = 0; i < refP3d.size(); ++i) {
        cv::Mat p3d = (cv::Mat_<double>(3,1) << refP3d[i].x, refP3d[i].y, refP3d[i].z);
        cv::Mat p3d_new = R_in * p3d + t_in;
        // project
        cv::Vec3f p3d_trans(p3d_new.at<double>(0,0), p3d_new.at<double>(1,0), p3d_new.at<double>(2,0));
        cv::Vec2f p2d_proj;
        if (p3d_trans[2] > 0) {
            p3d_trans /= p3d_trans[2];
            p2d_proj[0] = (mFx_ * p3d_new.at<double>(0,0) + mCx_);
            p2d_proj[1] = (mFy_ * p3d_new.at<double>(1,0) + mCy_);

            cv::Vec2f p2d(currP2d[i].x, currP2d[i].y);
            errTotal += sqrt( (p2d[0] - p2d_proj[0]) * (p2d[0] - p2d_proj[0]) + 
                            (p2d[1] - p2d_proj[1]) * (p2d[1] - p2d_proj[1]) );
            ++count;
        }
    }
    errAvg = errTotal / count;
    std::cout << "Average error is " << errAvg << std::endl;
}


// 调用opencv的pnp求解初值
bool XIAOC::MotionEstimate::ComputePoseByPnp(std::vector<cv::Point3f>& vp3d, std::vector<cv::Point2f>& vp2d, 
    cv::Mat& R_out, cv::Mat& t_out) {

    cv::Mat K = (cv::Mat_<double>(3,3) << mFx_, 0, mCx_,
                                          0, mFy_, mCy_,
                                          0, 0, 1);

    cv::Mat rvec, tvec;
    cv::Mat inliers;
    bool flag = cv::solvePnPRansac(vp3d, vp2d, K, cv::Mat(), rvec, tvec, false, 100, 2.0, 0.99, inliers);
    // bool flag = cv::solvePnP(vp3d, vp2d, K, cv::Mat(), rvec, tvec, false, 0);
    cv::Rodrigues(rvec, R_out);
    t_out = tvec;

    ComputeError(vp3d, vp2d, R_out, t_out);
    return flag;
}

// 利用g2o优化相机姿态
bool XIAOC::MotionEstimate::OptimizePoseByG2O(std::vector<cv::Point3f>& refP3d, std::vector<cv::Point2f>& currP2d, 
    cv::Mat& R_in, cv::Mat& t_in, cv::Mat& R_out, cv::Mat& t_out) {
    
    Eigen::Matrix3d esti_R;
    esti_R << ( R_in.at<double>(0,0), R_in.at<double>(0,1), R_in.at<double>(0,2), 
                R_in.at<double>(1,0), R_in.at<double>(1,1), R_in.at<double>(1,2),
                R_in.at<double>(2,0), R_in.at<double>(2,1), R_in.at<double>(2,2) );
    Eigen::Vector3d esti_t( t_in.at<double>(0,0), t_in.at<double>(1,0), t_in.at<double>(2,0) );
    Sophus::SE3 esti( esti_R, esti_t ); 

    // 初始化求解器
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType *linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    // g2o::OptimizationAlgorithmGaussNewton *solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
    
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    // 将pose顶点添加进求解器
    VertexPose *vPose = new VertexPose();
    vPose->setId(0);
    vPose->setEstimate(esti);
    vPose->setFixed(false);
    optimizer.addVertex(vPose);

    // 相机内参数
    cv::Mat K = (cv::Mat_<double>(3,3) << mFx_, 0, mCx_,
                                          0, mFy_, mCy_,
                                          0, 0, 1);

    // 构建边
    const float delta = sqrt(7.815);   // 5.991 7.815
    std::vector<EdgeProjectPoseOnly *> edgeProj;
    for (int i = 0; i < refP3d.size(); ++i) {
        // 三维点和相机内参
        Eigen::Vector3d p3d(refP3d[i].x, refP3d[i].y, refP3d[i].z);
        EdgeProjectPoseOnly *edgeP = new EdgeProjectPoseOnly(K, p3d);
        edgeP->setId(i+1);
        edgeP->setVertex(0, vPose);
        
        // 信息矩阵和观测值
        Eigen::Vector2d p2d(currP2d[i].x, currP2d[i].y);
        edgeP->setMeasurement(p2d);
        edgeP->setInformation(Eigen::Matrix2d::Identity());
        edgeProj.push_back(edgeP);

        // 核函数
        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
        rk->setDelta(delta);
        edgeP->setRobustKernel(rk);
        optimizer.addEdge(edgeP);
    }

    optimizer.initializeOptimization(0);
    optimizer.optimize(10);

    // 判断边条件
    int inliers = 0, outliers = 0;
    for (auto &e : edgeProj) {
        if (e->chi2() > delta) {
            e->setLevel(1);
            ++outliers;
        }
        else {
            e->setRobustKernel(nullptr);
            ++inliers;
        }
    }
    
    // 内点数量太少
    if (inliers < 10) {
        return false;
    }

    // 再次优化
    optimizer.initializeOptimization(0);
    optimizer.optimize(10);

    Sophus::SE3 outPose = vPose->estimate();
    Eigen::Matrix3d R = outPose.rotation_matrix();
    Eigen::Vector3d t = outPose.translation();
    R_out = (cv::Mat_<double>(3,3) << R(0,0), R(0,1), R(0,2), 
                                      R(1,0), R(1,1), R(1,2),
                                      R(2,0), R(2,1), R(2,2));
    t_out = (cv::Mat_<double>(3,1) << t(0,0), t(1,0), t(2,0));

    std::cout << "After g2o optimization!" << std::endl;
    ComputeError(refP3d, currP2d, R_out, t_out);

    return true;
}