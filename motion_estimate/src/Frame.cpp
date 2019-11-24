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
    mFx_ = K.at<float>(0,0);
    mFy_ = K.at<float>(1,1);
    mCx_ = K.at<float>(0,2);
    mCy_ = K.at<float>(1,2);
}

void XIAOC::Frame::SetCameraIntrinsic(const float& fx, const float fy, 
    const float& cx, const float& cy) {
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
float XIAOC::Frame::GetPointDepth(float u, float v) {
    float factor = 5000.0;

    ushort d = mImgDepth_.ptr<ushort>(int(v))[int(u)];
    if (d == 0) 
        return 0;
    
    return (float)d / factor;
    /*    
    int count = 0;
    float d_total = 0;
    
    int u1 = std::ceil(u), v1 = std::ceil(v);
    float d = mImgDepth_.at<float>(v1, u1);
    if (d > 0) { d_total += d; ++count; }

    u1 = std::ceil(u); v1 = std::floor(v);
    d = mImgDepth_.at<float>(v1, u1);
    if (d > 0) { d_total += d; ++count; }

    u1 = std::floor(u); v1 = std::ceil(v);
    d = mImgDepth_.at<float>(v1, u1);
    if (d > 0) { d_total += d; ++count; }

    u1 = std::floor(u); v1 = std::floor(v);
    d = mImgDepth_.at<float>(v1, u1);
    if (d > 0) { d_total += d; ++count; }

    if (count > 0)
        return d_total/(count*factor);
    else 
        return 0.;*/
}
