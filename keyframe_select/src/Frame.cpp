#include "Frame.h"

XIAOC::Frame::Frame(int id, cv::Mat& camera_intrinsic) {
    mnId_ = id;
    mmK_ = camera_intrinsic.clone();
}

XIAOC::Frame::Frame(const Frame& f) {
    mnId_ = f.mnId_;
    mmK_ = f.mmK_;
    mmDesp_ = f.mmDesp_;
    mvKpt_ = f.mvKpt_;
    mvDepth_ = f.mvDepth_;
    mvMapPoints_ = f.mvMapPoints_;
    mvMapPointsId_ = f.mvMapPointsId_;
}

XIAOC::Frame::Frame(int id, std::vector<cv::KeyPoint>& kpt, cv::Mat& desp, 
        std::vector<double>& depth, cv::Mat& camera_intrinsic) {
    mnId_ = id;
    mmK_ = camera_intrinsic.clone();
    mmDesp_ = desp.clone();
    mvKpt_ = kpt;
    mvDepth_ = depth;
}

XIAOC::Frame::~Frame() {

}

 // 计算地图点
void XIAOC::Frame::ComputeMapPoints(std::vector<cv::DMatch>& matches, bool queryIdx=true) {
    
    mvMapPointsId_.clear();
    mvMapPoints_.clear();

    for (int i = 0; i < matches.size(); ++i) {
        int idx = -1;
        if (queryIdx) {
            idx = matches[i].queryIdx;
        }
        else {
            idx = matches[i].trainIdx;
        }

        double d = mvDepth_[idx];
        cv::KeyPoint pt = mvKpt_[idx];
        double x = (pt.pt.x - mmK_.at<double>(0,2)) * d / mmK_.at<double>(0,0);
        double y = (pt.pt.y - mmK_.at<double>(1,2)) * d / mmK_.at<double>(1,1);
        cv::Point3d p3d(x, y, d);
        mvMapPoints_.push_back(p3d);
        mvMapPointsId_.push_back(idx);
    }
}

// 设置三维点
void XIAOC::Frame::Set3DPointsAndIndex() {
    mvMapPoints_.clear();
    mvMapPointsId_.clear();
    
    for (int i = 0; i < mvKpt_.size(); ++i) {
        cv::KeyPoint kpt = mvKpt_[i];
        double d = mvDepth_[i];
        if (d > 0) {
            double u = kpt.pt.x;
            double v = kpt.pt.y;
            double x = (u - mmK_.at<double>(0,2)) * d / mmK_.at<double>(0,0);
            double y = (v - mmK_.at<double>(1,2)) * d / mmK_.at<double>(1,1);
            cv::Point3d p3d(x, y, d);
            mvMapPoints_.push_back(p3d);
            mvMapPointsId_.push_back(i);
        }
    }
    std::cout << "Set 3d map points are " << mvMapPoints_.size() << std::endl;
}