
#ifndef FEATURE_MATCHING_H
#define FEATURE_MATCHING_H

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

// dbow3
#include "DBoW3/src/DBoW3.h"

// c++ 
#include <memory>

namespace XIAOC {

    class FeatureMatching {

    public:
        // 默认构造函数,读取bow词典模型并设置阈值
        FeatureMatching( std::shared_ptr<DBoW3::Vocabulary> voc );

        // 利用构造函数直接计算特征匹配,并选择匹配方式
        FeatureMatching( std::shared_ptr<DBoW3::Vocabulary> voc, 
                            cv::Mat& desp1, cv::Mat& desp2,
                             std::vector<cv::DMatch>& matches, bool use_bow );

        // 默认析构函数
        ~FeatureMatching();

        // 利用暴力匹配法进行特征匹配
        bool MatchByBruteForce( cv::Mat& desp1, cv::Mat& desp2, std::vector<cv::DMatch>& matches );

        // 利用基于词袋模型的方法进行特征匹配
        bool MatchByDBoW( cv::Mat& desp1, cv::Mat& desp2, std::vector<cv::DMatch>& matches );

        // 设置bow匹配的阈值条件
        void SetThreshold( int threshold, float nnratio );

    private:

        // 将描述子转换为bow向量
        void ComputeBoWVector( cv::Mat& desp, DBoW3::BowVector& bowVec, DBoW3::FeatureVector& featVec);

        // 计算两个描述子间的匹配分数
        int ComputeMatchingScore( cv::Mat& desp1, cv::Mat& desp2 ); 

        // 词袋模型
        std::shared_ptr<DBoW3::Vocabulary> mVoc_ = nullptr;

        // bow向量,包含id和权重
        DBoW3::BowVector mBowVec1_, mBowVec2_;

        // feature向量,包含在bow树中的id和特征的id
        DBoW3::FeatureVector mFeatVec1_, mFeatVec2_;

        // 最近邻比例法阈值条件
        int mTh_;
        float mNNRatio_;
    };
}

#endif // FEATURE_MATCHING_H