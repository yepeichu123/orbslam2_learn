#ifndef KEYFRAME_SELECT_H
#define KEYFRAME_SELECT_H

#include "Frame.h"

namespace XIAOC {

    class Frmae;

    class KeyframeSelect {
    public: 
        // 默认构造函数
        KeyframeSelect();

        // 默认析构函数
        ~KeyframeSelect();

        // 检验关键帧
        bool CheckKeyframe(Frame& last_keyframe, Frame& new_frame, std::vector<cv::DMatch>& matches);

        // 检查内点率
        bool CheckInliersRatio(Frame& last_keyframe, std::vector<cv::DMatch>& matches);

    private:

        // 是否正在全局闭环
        bool CheckGlobalLoop() {
            return mbGlobalLoop_;
        }

        // 是否是纯定位模式
        bool CheckLocationMode() {
            return mbLocationMode_;
        }

        // 上一次重定位后的关键帧
        int CheckLastRelocId() {
            return mnLastRelocId_;
        }


        // 上一次重定位的关键帧id
        int mnLastRelocId_;

        // 是否全局闭环
        bool mbGlobalLoop_;

        // 是否纯定位模式
        bool mbLocationMode_;

        // 局部建图是否空闲
        bool mbLocalMapFree_;

        // 局部建图中的关键帧队列中的关键帧数目
        int mnKeyFrameNumInQueue_;

        // -----------------------
        // 插入新关键帧前至少经过多少帧
        int mnMinKeyframePass_;

        // 两个关键帧间能够容许的最大帧数
        int mnMaxKeyframePass_;

        // 最大内点率,超过则表示两帧间重叠率太高
        double mdMaxInlierRatio_;

        // 最小内点率,低于则表示两帧点共视部分太少
        double mdMinInlierRatio_;

        // 最小内点数量,主要考虑做运动估计
        int mnMinInliers_;
    };
}



#endif // KEYFRAME_SELECT_H