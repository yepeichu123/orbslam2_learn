// feature matching
#include "FeatureMatching.h"

// opencv 
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

// cpp
#include <iostream>
#include <memory>

using namespace std;
using namespace cv;
using namespace XIAOC;

int main(int argc, char** argv) {

    if (argc < 4) {
        cout << "Please input ./bin/feature_matching ./vocab/orbvoc.dbow3 ./dataset/1.png ./dataset/2.png" << endl;
        return -1;
    }

    // 读取图像
    Mat img1 = imread(argv[2], 0);
    Mat img2 = imread( argv[3], 0);

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
    waitKey(0);

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
    waitKey(0);

    // BoW匹配法
    vector<DMatch> bowMatch;
    featMatch->SetThreshold(50, 0.6);
    featMatch->MatchByDBoW(desp1, desp2, bowMatch);
    cout << "BOW matching pairs are " << bowMatch.size() << endl;
    Mat bowOut;
    drawMatches(img1, kpt1, img2, kpt2, bowMatch, bowOut);
    imshow("bowMatching", bowOut);
    waitKey(0);

    return 0;
} 