// feature matching
#include "FeatureMatching.h"

// opencv 
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

// cpp
#include <iostream>
#include <memory>

using namespace std;
using namespace cv;
using namespace XIAOC;

int main(int argc, char** argv) {

    if (argc < 5) {
        cout << "Please input ./bin/feature_matching ./vocab/orbvoc.dbow3 1.png 1depth.png 2.png" << endl;
        return -1;
    }

    // 读取图像
    Mat img1 = imread(argv[2], 0);
    Mat img1_depth = imread(argv[3], 0);
    Mat img2 = imread( argv[4], 0);

    // 特征提取和描述子计算
    Ptr<ORB> orb = ORB::create();
    vector<KeyPoint> kpt1, kpt2;
    cv::Mat desp1, desp2;
    orb->detectAndCompute(img1, Mat(), kpt1, desp1);
    orb->detectAndCompute(img2, Mat(), kpt2, desp2);

    // 画出提取的特征点
    Mat showImg;
    drawKeypoints(img1, kpt1, showImg, cv::Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("kptImg", showImg);
    waitKey(1);

    // 读取词袋模型
    shared_ptr<DBoW3::Vocabulary> voc(new DBoW3::Vocabulary());
    voc->load(argv[1]);

    shared_ptr<FeatureMatching> featMatch = make_shared<FeatureMatching>(voc);
    
    // 暴力匹配法
    vector<DMatch> bfMatch;
    featMatch->MatchByBruteForce(desp1, desp2, bfMatch);
    cout << "Brute force matching pairs are " << bfMatch.size() << endl;
    Mat bfOut;
    drawMatches(img1, kpt1, img2, kpt2, bfMatch, bfOut);
    imshow("bfMatching", bfOut);
    waitKey(1);

    // BoW匹配法
    vector<DMatch> bowMatch;
    featMatch->SetThreshold(50, 0.6);
    featMatch->MatchByDBoW(desp1, desp2, bowMatch);
    cout << "BOW matching pairs are " << bowMatch.size() << endl;
    Mat bowOut;
    drawMatches(img1, kpt1, img2, kpt2, bowMatch, bowOut);
    imshow("bowMatching", bowOut);
    waitKey(1);

    // 投影匹配法
    Mat K = (Mat_<double>(3,3) << 517.3, 0, 318.6, 0, 516.5, 255.3, 0, 0, 1);
    // 获取匹配对相应的3D-2D关系
    vector<Point3d> vp3d1;
    vector<Point2d> vp2d2;
    for (int i = 0; i < bowMatch.size(); ++i) {
        DMatch m1 = bowMatch[i];

        // 参考帧三维点
        double u = kpt1[m1.queryIdx].pt.x;
        double v = kpt1[m1.queryIdx].pt.y;
        ushort d = img1_depth.at<ushort>(int(v), int(u));
        if (d < 0) {
            continue;
        }
        double x = (u - K.at<double>(0,2)) * double(d) / K.at<double>(0,0);
        double y = (v - K.at<double>(1,2)) * double(d) / K.at<double>(1,1);
        Point3d p3d(x, y, double(d));
        vp3d1.push_back(p3d);

        // 当前帧二维点
        Point2d p2d(kpt2[m1.trainIdx].pt.x, kpt2[m1.trainIdx].pt.y);
        vp2d2.push_back(p2d);
    }
    // 运动估计
    cv::Mat rvec, tvec;
    cv::Mat inliers;
    if (vp3d1.size() > 10) {
        cout << "enter solvePnPRansac!" << endl;
        solvePnP(vp3d1,vp2d2, K, Mat(), rvec, tvec);
        cout << "finished solvePnPRansac!" << endl;
    }
    cv::Mat R, t;
    Rodrigues(rvec, R);
    t = tvec.clone();
    

    // 获取当前帧和参考帧中所有点对应的三维点和二维点
    vp3d1.clear();
    vp2d2.clear();
    for (int i = 0; i < kpt1.size(); ++i) {
        // 参考帧的三维点
        KeyPoint kp1 = kpt1[i];
        double u = kp1.pt.x;
        double v = kp1.pt.y;
        ushort d = img1_depth.at<ushort>(int(v), int(u));
        if (d < 0) {
            continue;
        }
        double x = (u - K.at<double>(0,2)) * double(d) / K.at<double>(0,0);
        double y = (v - K.at<double>(1,2)) * double(d) / K.at<double>(1,1);
        Point3d p3d(x, y, double(d));
        vp3d1.push_back(p3d);

        // 当前帧的二维点
        KeyPoint kp2 = kpt2[i];
        Point2d p2d(kp2.pt.x, kp2.pt.y);
        vp2d2.push_back(p2d);
    }
    cout << "start matching by projection!" << endl;
    vector<DMatch> projMatch;
    if (vp3d1.size() > 10) {
        double radian = 10.0;
        featMatch->MatchByProject(vp3d1, desp1, vp2d2, desp2, radian, K, R, t, projMatch);
    }
    if (projMatch.size() > 10) {
        cout << "Size of matching by projection is : " << projMatch.size() << endl;
        Mat projOut;
        drawMatches(img1, kpt1, img2, kpt2, projMatch, projOut);
        imshow("projMatching", projOut);
        waitKey(0);
    }

    return 0;
} 