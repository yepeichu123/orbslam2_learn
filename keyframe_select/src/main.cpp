#include "KeyframeSelect.h"
#include "Frame.h"

// c++
#include <iostream>
#include <string>
#include <sstream>

// opencv
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#define IMG_NUM 12

using namespace std;
using namespace cv;
using namespace XIAOC;

// 从指定路径中读取深度图和彩色图
void ReadImages(string& path_to_img, vector<Mat>& rgb_imgs, vector<Mat>& depth_imgs);

// 特征提取
void FeatureExtraction(Mat& rgb_img, vector<KeyPoint>& kpt, Mat& desp);

// 计算对应三维点
void Compute3DPoints(Mat& depth_img, vector<KeyPoint>& kpt, Mat& K, 
    vector<Point3d>& vp3d, vector<int>& index);

// 计算深度值
void ComputeDepth(Mat& depth_img, vector<KeyPoint>& kpt, vector<double>& vdepth);

// 特征匹配
void FeatureMatching(Mat& ref_desp, Mat& cur_desp, vector<DMatch>& matches);

int main(int argc, char** argv) {

    if (argc < 2) {
        cout << "Please input ./keyframe_select ./path_to_image" << endl;
        return -1;
    }

    // 读取图像
    string path_to_img = argv[1];
    vector<Mat> rgb_img, depth_img;
    ReadImages(path_to_img, rgb_img, depth_img);
    if (rgb_img.size() == 0) {
        cout << "No valid images! please check the path!" << endl;
        return -1;
    }

    Mat K = ( Mat_<double>(3,3) << 517.3, 0, 318.6, 0, 516.5, 255.3, 0, 0, 1 );

    // 关键帧
    vector<Frame> vKeyframes;
    KeyframeSelect kf_select;

    // 第一帧
    Mat I_rgb = rgb_img[0];
    Mat I_depth = depth_img[0];
    vector<KeyPoint> vkpts;
    Mat desps;
    vector<Point3d> vp3d;
    vector<int> vindex;
    vector<double> vdepth;
    FeatureExtraction(I_rgb, vkpts, desps);
    ComputeDepth(I_depth, vkpts, vdepth);
    Frame kf(0, vkpts, desps, vdepth, K);
    // 设置三维地图点(粗略版本!!!)
    kf.Set3DPointsAndIndex();
    vKeyframes.push_back(kf);
    cout << "Add the first keyframe!" << endl;
    cout << "-----------------------" << endl;

    // 逐帧处理
    for (int i = 1; i < rgb_img.size(); ++i) {
        I_rgb = rgb_img[i];
        I_depth = depth_img[i];
        FeatureExtraction(I_rgb, vkpts, desps);
        ComputeDepth(I_depth, vkpts, vdepth);
        Frame new_frame(i, vkpts, desps, vdepth, K);

        // 特征匹配
        vector<DMatch> matches;
        // 最后一个关键帧
        Frame last_keyframe = vKeyframes[vKeyframes.size()-1];
        Mat last_kf_desp = last_keyframe.GetDescriptors();
        FeatureMatching(last_kf_desp, desps, matches);

        // 查看是否需要插入关键帧
        bool add_new_kf = kf_select.CheckKeyframe(last_keyframe, new_frame, matches);
        if (add_new_kf) {
            // 插入新的地图点(没有去单独实现)
            new_frame.Set3DPointsAndIndex();
            // 插入新的关键帧
            vKeyframes.push_back(new_frame);
            cout << "Add the " << i << " image as keyframe!" << endl;
            cout << "--------------------------------------" << endl;
        }
    }
    return 0;
}

// 从指定路径中读取深度图和彩色图
void ReadImages(string& path_to_img, vector<Mat>& rgb_imgs, vector<Mat>& depth_imgs) {
    rgb_imgs.clear();
    depth_imgs.clear();

    for (int i = 0; i < IMG_NUM; ++i) {
        stringstream ss;
        string rgb_path, depth_path;
        ss << path_to_img << "rgb/" << i << ".png";
        ss >> rgb_path;
        Mat rgb = imread(rgb_path.c_str(), 0);
        rgb_imgs.push_back(rgb);

        ss.clear();
        ss << path_to_img << "depth/" << i << ".png";
        ss >> depth_path;
        Mat depth = imread(depth_path.c_str(), 0);
        depth_imgs.push_back(depth);
    }
}

// 特征提取
void FeatureExtraction(Mat& rgb_img, vector<KeyPoint>& kpt, Mat& desp) {
    kpt.clear();
    desp.release();
    Ptr<ORB> orb = ORB::create(500);
    orb->detectAndCompute(rgb_img, Mat(), kpt, desp);
}

// 计算对应三维点
void Compute3DPoints(Mat& depth_img, vector<KeyPoint>& kpt, Mat& K, 
    vector<Point3d>& vp3d, vector<int>& index) {
    vp3d.clear();

    for (int i = 0; i < kpt.size(); ++i) {
        KeyPoint kp = kpt[i];
        double u = kp.pt.x;
        double v = kp.pt.y;

        ushort d = depth_img.at<ushort>(int(v), int(u));
        if (d > 0) {
            double z = double(d);
            double x = (u - K.at<double>(0,2)) * z / K.at<double>(0,0);
            double y = (v - K.at<double>(1,2)) * z / K.at<double>(1,1);
            Point3d p3d(x, y, z);
            vp3d.push_back(p3d);
            index.push_back(i);
        }
    }
}

// 计算深度值
void ComputeDepth(Mat& depth_img, vector<KeyPoint>& kpt, vector<double>& vdepth) {

    vdepth.clear();

    for (int i = 0; i < kpt.size(); ++i) {
        KeyPoint pt = kpt[i];
        double u = pt.pt.x;
        double v = pt.pt.y;
        ushort d = depth_img.at<ushort>(int(v), int(u));
        vdepth.push_back(double(d));
    }
}

// 特征匹配
void FeatureMatching(Mat& ref_desp, Mat& cur_desp, vector<DMatch>& matches) {
    Ptr<BFMatcher> bf_match = new BFMatcher(NORM_HAMMING);
    vector<DMatch> match_temp;
    bf_match->match(ref_desp, cur_desp, match_temp);

    matches.clear();
    double d_min = min_element(match_temp.begin(), match_temp.end(), 
        [](DMatch& d1, DMatch& d2) { return d1.distance < d2.distance; } )->distance;
    for (int i = 0; i < match_temp.size(); ++i) {
        double d = match_temp[i].distance;
        if (d < max(30.0, d_min * 2)) {
            matches.push_back(match_temp[i]);
        }
    }
}