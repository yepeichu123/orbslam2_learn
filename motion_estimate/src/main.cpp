
#include "MotionEstimate.h"
#include "Frame.h"

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

// c++
#include <iostream>
#include <vector>
#include <memory>

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace XIAOC;

typedef Matrix<double, 3, 1> Vec3;
typedef Matrix<double, 2, 1> Vec2;

// 图像数量
#define N 3

// 读取图像
void ReadImgs(string& path, vector<Mat>& rgb_img, vector<Mat>& depth_img);

// 特征匹配
int FeatureMatching(Mat desp1, Mat desp2, vector<DMatch>& good_matches);

int main(int argc, char** argv) {

    if (argc < 2) {
        cout << "Please input ./motion_estimate path_to_dataset" << endl;
        return -1;
    }

    // 相机内参数
    Mat K = ( Mat_<double>(3,3) << 517.3, 0, 318.6, 0, 516.5, 255.3, 0, 0, 1 );

    // 读取图像
    vector<Mat> rgb_img;
    vector<Mat> depth_img;
    string path = argv[1];
    ReadImgs(path, rgb_img, depth_img);

    // 提取特征
    vector<Frame> vframe;
    Ptr<ORB> orb = ORB::create();
    for (int i = 0; i < rgb_img.size(); ++i) {
        vector<KeyPoint> kpt;
        Mat desp;
        orb->detectAndCompute(rgb_img[i], Mat(), kpt, desp);
        Frame fr(i, rgb_img[i], depth_img[i], kpt, desp);
        fr.SetCameraIntrinsic(K);
        vframe.push_back(fr);
    }
    
    /*************************基于参考帧的运动估计***********************************/
    cout << "********************motion estimation by reference frame***************" << endl;
    // 特征匹配
    vector<DMatch> matches;
    FeatureMatching(vframe[0].GetAllDescriptors(), vframe[1].GetAllDescriptors(), matches);
    // 画出匹配结果
    Mat draw_match_01;
    drawMatches(vframe[0].GetImg(), vframe[0].GetAllKeyPoints(),
        vframe[1].GetImg(), vframe[1].GetAllKeyPoints(), matches, draw_match_01);
    imshow("draw_match_01", draw_match_01);
    waitKey(1);

    // 计算参考帧(图像1)的三维点以及当前帧的图像像素点
    vector<Point3d> vp3d;
    vector<Point2d> vp2d;
    for (int i = 0; i < matches.size(); ++i) {
        KeyPoint pr = vframe[0].GetKeyPoint(matches[i].queryIdx);
        KeyPoint pc = vframe[1].GetKeyPoint(matches[i].trainIdx);
        // 获取参考帧点的深度值
        double u = pr.pt.x, v = pr.pt.y;
        double d = vframe[0].GetPointDepth(u, v);
        if (d > 0) {
            double x = (u - K.at<double>(0,2)) * d / K.at<double>(0,0);
            double y = (v - K.at<double>(1,2)) * d / K.at<double>(1,1);
            Point3d p3d(x, y, d);
            Point2d p2d(pc.pt.x, pc.pt.y);
            vp3d.push_back(p3d);
            vp2d.push_back(p2d);
        } 
    }
    cout << "matching pairs are " << vp3d.size() << endl;

    shared_ptr<MotionEstimate> motionEsti = make_shared<MotionEstimate>(K);
    // 基于参考帧的运动估计
    // Mat R_0W = Mat::eye(3,3,CV_64F);
    // Mat t_0W = (Mat_<double>(3,1) << 0, 0, 0);
    Mat R_10, t_10;
    Mat inliers;
    motionEsti->MotionEstimateByRefFrame(vp3d, vp2d, R_10, t_10, inliers);


    /*************************基于运动模型的运动估计***********************************/
    cout << "********************motion estimation by motion model***************" << endl;
    // 特征匹配
    matches.clear();
    FeatureMatching(vframe[1].GetAllDescriptors(), vframe[2].GetAllDescriptors(), matches);
    Mat draw_match_12;
    drawMatches(vframe[1].GetImg(), vframe[1].GetAllKeyPoints(),
        vframe[2].GetImg(), vframe[2].GetAllKeyPoints(), matches, draw_match_12);
    imshow("draw_match_12", draw_match_12);
    waitKey(0);

    // 计算参考帧(图像1)的三维点以及当前帧的图像像素点
    vp3d.clear();
    vp2d.clear();
    for (int i = 0; i < matches.size(); ++i) {
        KeyPoint pr = vframe[1].GetKeyPoint(matches[i].queryIdx);
        KeyPoint pc = vframe[2].GetKeyPoint(matches[i].trainIdx);
        // 获取参考帧点的深度值
        double u = pr.pt.x, v = pr.pt.y;
        double d = vframe[1].GetPointDepth(u, v);
        if (d > 0) {
            double x = (u - K.at<double>(0,2)) * d / K.at<double>(0,0);
            double y = (v - K.at<double>(1,2)) * d / K.at<double>(1,1);
            Point3d p3d(x, y, d);
            Point2d p2d(pc.pt.x, pc.pt.y);
            vp3d.push_back(p3d);
            vp2d.push_back(p2d);
        } 
    }
    // 基于运动模型的运动估计
    // 由于假设R_0 = I, t_0 = [0 0 0]
    // 所以R_1 = R_10 * R_0, t_1 = t_10;
    // 则由0-->1的运动模型自然便是R_10, t_10; 
    Mat R_21, t_21;
    motionEsti->MotionEstimateByModel(vp3d, vp2d, R_10, t_10, R_21, t_21);

    return 0;
}

// 读取图像
void ReadImgs(string& path, vector<Mat>& rgb_img, vector<Mat>& depth_img) {
    cout << "*****************enter Read images*************" << endl;
    // 读取图像
    for (int i = 0; i < N; ++i) {
        string rgb_path, depth_path;
        stringstream ss;
        ss << path << "rgb/" << i << ".png";
        ss >> rgb_path;
        ss.clear();
        ss << path << "depth/" << i << ".png";
        ss >> depth_path;

        Mat rgb_temp = imread(rgb_path.c_str(), 0);
        Mat depth_temp = imread(depth_path.c_str(), 0);
        rgb_img.push_back(rgb_temp);
        depth_img.push_back(depth_temp);
    }
}

// 特征匹配
int FeatureMatching(Mat desp1, Mat desp2, vector<DMatch>& good_matches) {

    cout << "********************enetr featureMathcing***************" << endl;
    // 特征匹配
    Ptr<BFMatcher> bfmatch = new BFMatcher(NORM_HAMMING);
    vector<DMatch> matches;
    // 计算图像1和图像2间的特征匹配
    bfmatch->match(desp1, desp2, matches);

    good_matches.clear();
    // 找最好的匹配对
    double d_min = min_element(matches.begin(), matches.end(), 
        [](DMatch& m1, DMatch& m2) { return m1.distance < m2.distance; })->distance;
    double d_max = max_element(matches.begin(), matches.end(),
        [](DMatch& m1, DMatch& m2) { return m1.distance < m2.distance; })->distance;
    for (int i = 0; i < matches.size(); ++i) {
        double d = matches[i].distance;
        if (d < std::max(30.0, d_min*2)) {
            good_matches.push_back(matches[i]);
        }
    }
    cout << "matching pairs are " << good_matches.size() << endl;
}
