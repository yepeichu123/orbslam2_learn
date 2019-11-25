#include "Frame.h"

// 默认构造函数
XIAOC::Frame::Frame() {

}

// 赋值构造函数
XIAOC::Frame::Frame(int id, cv::Mat& rgbImg, cv::Mat& depthImg, 
    std::vector<cv::KeyPoint>& kpt, cv::Mat& desp ) {
    mId_ = id;
    mImgRgb_ = rgbImg.clone();
    mImgDepth_ = depthImg.clone();
    mDesp_ = desp.clone();

    mvKpt_.clear();
    mvKpt_.insert(mvKpt_.end(), kpt.begin(), kpt.end());
}

// 设置相机内参数
void XIAOC::Frame::SetCameraIntrinsic(cv::Mat& K) {
    mFx_ = K.at<double>(0,0);
    mFy_ = K.at<double>(1,1);
    mCx_ = K.at<double>(0,2);
    mCy_ = K.at<double>(1,2);
}

void XIAOC::Frame::SetCameraIntrinsic(const double& fx, const double& fy, 
    const double& cx, const double& cy) {
    mFx_ = fx;
    mFy_ = fy;
    mCx_ = cx;
    mCy_ = cy;
}

// 设置图像
void XIAOC::Frame::SetImg(cv::Mat& rgbImg, cv::Mat& depthImg) {
    mImgRgb_ = rgbImg.clone();
    mImgDepth_ = depthImg.clone();
}

// 设置特征点
void XIAOC::Frame::SetKeyPoints(std::vector<cv::KeyPoint>& kpt) {
    mvKpt_.clear();
    mvKpt_.insert(mvKpt_.end(), kpt.begin(), kpt.end());
}

// 设置描述子
void XIAOC::Frame::SetDescriptors(cv::Mat& desp) {
    mDesp_ = desp.clone();
}

/*
// 获取匹配点的三维信息
void XIAOC::Frame::ComputePoint3d(std::vector<cv::DMatch>& matches, bool queryIdx) {
    mvP3d_.clear();
    mvMP3d_.clear();
    mvMatches_.clear();
    mvMatches_.insert(mvMatches_.end(), matches.begin(), matches.end());

    for (int i = 0; i < matches.size(); ++i) {
        int idx = 0;
        if (queryIdx) {
            idx = matches[i].queryIdx;
        }
        else {
            idx = matches[i].trainIdx;
        } 
    
        cv::KeyPoint kpt = mvKpt_[idx];
        float d = GetPointDepth(kpt.pt.x, kpt.pt.y);
        if (d > 0)
        {
            float u = kpt.pt.x, v = kpt.pt.y;
            float x = (u - mCx_) / mFx_ * d;
            float y = (v - mCy_) / mFy_ * d;
            cv::Point3f p(x, y, d);
            mvP3d_.push_back(p);

            XIAOC::MyPoint3d mpt3d;
            mpt3d.id_ = idx;
            mpt3d.kpt_ = kpt;
            mpt3d.p3d_ = p;
            mpt3d.p2d_ = cv::Point2f(u, v);
            mpt3d.desp_ = mDesp_.row(idx).clone();
            mvMP3d_.push_back(mpt3d);
        }
    }
}
*/

// 获取点的深度值
double XIAOC::Frame::GetPointDepth(double u, double v) {
    double factor = 5000.0;

    int x = std::floor(u);
    int y = std::floor(v);
    ushort d = mImgDepth_.ptr<ushort>(y)[x];
    if (d != 0) {
        return double(d)/factor;
    }
    else {
        // check the nearby points 
        int dx[4] = {-1,0,1,0};
        int dy[4] = {0,-1,0,1};
        for ( int i=0; i<4; i++ )
        {
            d = mImgDepth_.ptr<ushort>( y+dy[i] )[x+dx[i]];
            if ( d!=0 )
            {
                return double(d)/factor;
            }
        }
    }
    return -1.0;
}
