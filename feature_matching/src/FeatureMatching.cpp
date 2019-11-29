#include "FeatureMatching.h"

#include <cmath>

// 默认构造函数,读取bow词典模型并设置阈值
XIAOC::FeatureMatching::FeatureMatching( std::shared_ptr<DBoW3::Vocabulary> voc ):mVoc_(voc) {
    // 设置bow阈值条件
    mTh_ = 50;
    mNNRatio_ = 0.6;
}

// 利用构造函数直接计算特征匹配,并选择匹配方式
XIAOC::FeatureMatching::FeatureMatching( std::shared_ptr<DBoW3::Vocabulary> voc,
        cv::Mat& desp1, cv::Mat& desp2,
        std::vector<cv::DMatch>& matches, bool use_bow = true  ):mVoc_(voc) {
    // 设置bow阈值条件
    mTh_ = 50;
    mNNRatio_ = 0.6;

    bool match_result = true;

    // 选择特征匹配方式
    if (use_bow) {
        match_result = MatchByDBoW(desp1, desp2, matches);
        std::cout << "Match By DBoW result is " << match_result << ", since we have " 
            << matches.size() << " matching pairs!" << std::endl;
    }
    else {
        match_result = MatchByBruteForce(desp1, desp2, matches);
        std::cout << "Match By BruteForce result is " << match_result << ", since we have " 
            << matches.size() << " matching pairs!" << std::endl;
    }

}

// 默认析构函数
XIAOC::FeatureMatching::~FeatureMatching() {
    mVoc_ = nullptr;
}

// 利用暴力匹配法进行特征匹配
bool XIAOC::FeatureMatching::MatchByBruteForce( cv::Mat& desp1, cv::Mat& desp2, std::vector<cv::DMatch>& matches ) {
    
    matches.clear();

    // 保证描述子非空
    if (!(desp1.rows > 0 && desp2.rows > 0)) {
        return false;
    }

    std::vector<cv::DMatch> matches_new;
    cv::BFMatcher bf(cv::NORM_HAMMING);
    // 特征匹配
    bf.match(desp1, desp2, matches_new);

    // 统计所有匹配的最大最小距离
    double min_dist = 10000.0, max_dist = 0.;
    for (int i = 0; i < matches_new.size(); ++i) {
        double dist = matches_new[i].distance;
        if (dist < min_dist) {
            min_dist = dist;
        }
        if (dist > max_dist) {
            max_dist = dist;
        }
    }

    // 设置阈值并根据阈值条件找到合适的匹配
    for (int i = 0; i < matches_new.size(); ++i) {
        double dist = matches_new[i].distance;
        if (dist < std::max(min_dist*2, 30.0)) {
            matches.push_back(matches_new[i]);
        }
    }

    return matches.size() > 0 ? true : false;
}

// 利用基于词袋模型的方法进行特征匹配
bool XIAOC::FeatureMatching::MatchByDBoW( cv::Mat& desp1, cv::Mat& desp2, std::vector<cv::DMatch>& matches ) {
    
    // 保证描述子非空
    if (!(desp1.rows > 0 && desp2.rows > 0)) {
        return false;
    }

    matches.clear();

    // std::cout << "image 1 has " << desp1.rows << " features!" << std::endl;
    // std::cout << "image 2 has " << desp2.rows << " features!" << std::endl; 

    // 计算bow向量
    ComputeBoWVector(desp1, mBowVec1_, mFeatVec1_);
    ComputeBoWVector(desp2, mBowVec2_, mFeatVec2_);

    // 优先进行bow向量匹配
    DBoW3::FeatureVector::iterator f1it = mFeatVec1_.begin();
    DBoW3::FeatureVector::iterator f2it = mFeatVec2_.begin();
    DBoW3::FeatureVector::iterator f1end = mFeatVec1_.end();
    DBoW3::FeatureVector::iterator f2end = mFeatVec2_.end();
    while (f1it != f1end && f2it != f2end) {
        // 特征向量第一项为bow向量id
        if (f1it->first == f2it->first) {
            const std::vector<unsigned int> vIdx1 = f1it->second;
            const std::vector<unsigned int> vIdx2 = f2it->second;
            // 在限定的bow向量包含的特征点中进行逐一匹配
            for (int i = 0; i < vIdx1.size(); ++i) {
                const size_t idx1 = vIdx1[i];
                cv::Mat feat1 = desp1.row(idx1);
                int bestDist1 = 256;
                int bestDist2 = 256;
                int bestIdx = -1;
                for (int j = 0; j < vIdx2.size(); ++j) {
                    const size_t idx2 = vIdx2[j];
                    cv::Mat feat2 = desp2.row(idx2);
                    // 计算描述子间的距离
                    int dist = ComputeMatchingScore(feat1, feat2);

                    if (dist < bestDist1) {
                        bestDist2 = bestDist1;
                        bestDist1 = dist;
                        bestIdx = idx2;
                    }
                    else if (dist < bestDist2) {
                        bestDist2 = dist;
                    }
                }
                // 最近邻比例法
                if (bestDist1 <= mTh_) {
                    if (static_cast<float>(bestDist1) < mNNRatio_ * static_cast<float>(bestDist2)) {
                        cv::DMatch m;
                        m.queryIdx = idx1;
                        m.trainIdx = bestIdx;
                        m.distance = bestDist1;
                        matches.push_back(m);
                    }
                }
            }
            ++f1it;
            ++f2it;
        }
        else if(f1it->first < f2it->first) {
            f1it = mFeatVec1_.lower_bound(f2it->first);
        }
        else {
            f2it = mFeatVec2_.lower_bound(f1it->first);
        }
    }
    return matches.size() > 0 ? true : false;
}

// 利用投影法进行特征匹配
bool XIAOC::FeatureMatching::MatchByProject( std::vector<cv::Point3d>& vp3d1, cv::Mat& desp1, 
    std::vector<cv::Point2d>& vp2d2, cv::Mat& desp2, double& radian, 
    cv::Mat& K, cv::Mat& R, cv::Mat& t, std::vector<cv::DMatch>& matches ) {

    matches.clear();

    // 将参考帧中的三维点投影到当前帧中查找合适的匹配点
    for (int i = 0; i < vp3d1.size(); ++i) {
        cv::Mat p3d = (cv::Mat_<double>(3,1) << vp3d1[i].x, vp3d1[i].y, vp3d1[i].z);
        cv::Mat p3d_trans = R*p3d + t;
        p3d_trans /= p3d_trans.at<double>(2,0);
        double u = K.at<double>(0,0)*p3d_trans.at<double>(0,0) + K.at<double>(0,2);
        double v = K.at<double>(1,1)*p3d_trans.at<double>(1,0) + K.at<double>(1,2);
        if (u < 0 || u > K.at<double>(0,2) || v < 0 || v > K.at<double>(1,2)) {
            continue;
        }
        // 在匹配半径中查找合适的匹配候选点
        std::vector<cv::Mat> desp_temp;
        std::vector<int> desp_index;
        for (int j = 0; j < vp2d2.size(); ++j) {
            cv::Point2d p2d = vp2d2[j];
            // u-radian < x < u+radian
            // v-radian < y < v+radian
            if ( (u-radian) < p2d.x && (u+radian) > p2d.x &&
                    (v-radian) < p2d.y && (v+radian) > p2d.y) {
                desp_temp.push_back(desp2.row(j));
                desp_index.push_back(j);
            }
        }

        // 在候选描述子中找到最合适的匹配点
        cv::Mat d1 = desp1.row(i);
        int min_dist = 256;
        int sec_min_dist = 256;
        int best_id = -1;
        for (int k = 0; k < desp_temp.size(); ++k) {
            cv::Mat d2 = desp2.row(desp_index[k]);
            int dist = ComputeMatchingScore(d1, d2);
            if (dist < min_dist) {
                sec_min_dist = min_dist;
                min_dist = dist;
                best_id = desp_index[k];
            }
            else if (dist < sec_min_dist) {
                sec_min_dist = dist;
            }
        }

        // 利用阈值条件筛选
        if (min_dist < mTh_) {
            if (min_dist < mNNRatio_*sec_min_dist) {
                cv::DMatch m1;
                m1.queryIdx = i;
                m1.trainIdx = best_id;
                m1.distance = min_dist;
                matches.push_back(m1);
            }
        }
    }

    return matches.size() > 0;
}

// 设置bow匹配的阈值条件
void XIAOC::FeatureMatching::SetThreshold( int threshold, float nn_ratio ) {
    mTh_ = threshold;
    mNNRatio_ = nn_ratio;
}

// 将描述子转换为bow向量
void XIAOC::FeatureMatching::ComputeBoWVector( cv::Mat& desp, DBoW3::BowVector& bowVec, DBoW3::FeatureVector& featVec ) {

    // 清除原有数据
    bowVec.clear();
    featVec.clear();

    // 将所有描述子划分成一个一个描述子存入向量中
    std::vector<cv::Mat> allDesp;
    allDesp.reserve(desp.rows);
    for (int i = 0; i < desp.rows; ++i) {
        allDesp.push_back(desp.row(i));
    }

    // 将每个描述子都转换成对应的bow向量
    mVoc_->transform(allDesp, bowVec, featVec, 4);

}

// 计算两个描述子间的匹配分数
int XIAOC::FeatureMatching::ComputeMatchingScore( cv::Mat& desp1, cv::Mat& desp2 ) {
    
    const int *p1 = desp1.ptr<int32_t>();
    const int *p2 = desp2.ptr<int32_t>();

    // 计算描述子匹配分数
    int dist = 0;

    // 位运算,目的是算出两个描述子之间有多少个不同的点
    for (int i = 0; i < 8; ++i, ++p1, ++p2) {
        unsigned int v = (*p1) ^ (*p2);
        v = v - ( (v >> 1) & 0x55555555 );
        v = ( v & 0x33333333 ) + ( (v >> 2) & 0x33333333 );
        dist += ( ( (v + (v >> 4)) & 0xF0F0F0F ) * 0x1010101 ) >> 24;
    }

    return dist;
}