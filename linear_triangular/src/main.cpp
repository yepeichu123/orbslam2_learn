#include "linear_triangular.h"
// c++
#include <iostream>
#include <vector>
// opencv
#include <opencv2/calib3d/calib3d.hpp>
// pcl
#include <pcl-1.7/pcl/io/pcd_io.h>
#include <pcl-1.7/pcl/point_types.h>
#include <pcl-1.7/pcl/point_cloud.h>
#include <pcl-1.7/pcl/common/impl/io.hpp>
#include <pcl-1.7/pcl/visualization/cloud_viewer.h>

using namespace cv;
using namespace std;
using namespace pcl;

// extract ORB features from image and compute descriptors
void featureExtraction( const Mat& img, vector<KeyPoint>& kpt, Mat& desp );

// feature matching by brute-force hamming method
// select some good matching pairs according to matching scores' threshold 
void featureMatching( const Mat& rDesp, const Mat& cDesp, vector<DMatch>& matches,
                        const vector<KeyPoint>& rKpt, const vector<KeyPoint>& cKpt,
                        vector<Point2d>& goodRKpt, vector<Point2d>& goodCKpt );

// estimate the motion by Epipolar geometry
bool motionEstimation( const vector<Point2d>& goodRKpt, const vector<Point2d>& goodCKpt,
                       const Mat& K, Mat& Rcr, Mat& tcr );

// to compare the triangular error between our method and opencv
void triangularByOpencv( const vector<Point2d>& RKpt, const vector<Point2d>& CKpt, 
                         const Mat& Trw, const Mat& Tcw, const Mat& K,
                         vector<Point2d>& nRKpt, vector<Point2d>& nCKpt,
                         vector<Point3d>& Pw3dCV );



int main( int argc, char** argv )
{
    // make sure we have right input data
    if( argc < 3 )
    {
        cout << "Please enter ./linear_triangular currImg refImg" << endl;
        return -1;
    }    

    // read images from input path
    Mat currImg = imread( argv[1], 1 );
    Mat refImg = imread( argv[2], 1 );
    
    // set up the camera parameters according to the dataset
    Mat K = ( Mat_<double>(3,3) << 517.3, 0, 318.6, 0, 516.5, 255.3, 0, 0, 1 );
    
    // detect orb features and compute orb descriptors
    vector<KeyPoint> ckpt, rkpt;
    Mat cdesp, rdesp;
    featureExtraction( currImg, ckpt, cdesp );
    featureExtraction( refImg, rkpt, rdesp );
    cout << "finish feature extration!" << endl;
    
    // feature matching
    vector<DMatch> matches;
    vector<Point2d> goodCKpt, goodRKpt;
    featureMatching( rdesp, cdesp, matches, rkpt, ckpt, goodRKpt, goodCKpt );
    cout << "finish feature matching!" << endl;

    // motion estimation
    Mat Rcr, tcr;
    if( !motionEstimation(goodRKpt, goodCKpt, K, Rcr, tcr) )
    {
        return -1;
    }
    cout << "finish motion estimation!" << endl;

    // construct transform matrix
    Mat Trw = Mat::eye(4,4,CV_64F);
    Mat Tcr = (Mat_<double>(3,4) << Rcr.at<double>(0,0), Rcr.at<double>(0,1), Rcr.at<double>(0,2), tcr.at<double>(0),
							     Rcr.at<double>(1,0), Rcr.at<double>(1,1), Rcr.at<double>(1,2), tcr.at<double>(1),
							     Rcr.at<double>(2,0), Rcr.at<double>(2,1), Rcr.at<double>(2,2), tcr.at<double>(2),
                                 0, 0, 0, 1);
    // propagate the transformation
    Mat Tcw = Tcr * Trw;
    Trw = Trw.rowRange(0,3).clone();
    Tcw = Tcw.rowRange(0,3).clone();                             
    cout << "Tcw = " << Tcw << endl;
    cout << "Trw = " << Trw << endl;
    cout << "So now we triangular the matching pairs by linear triangular." << endl;
    

    // begin triangular according to our blog
    vector<Point3d> Pw3d;
    vector<Point2d> Pr2d;
    XIAOC::LinearTriangular myTriangular( K, Trw, Tcw );
    for( int i = 0; i < matches.size(); ++i )
    {
        Point3d pw;
        bool result = myTriangular.TriangularPoint( goodRKpt[i], goodCKpt[i], pw );
        // check if triangular is success
        if( !result )
        {
            continue;
        }
        //cout << "pw in world coordinate is " << pw << endl;
        // save points in 3D world coordinate 
        Pw3d.push_back( pw );
        Pr2d.push_back( goodRKpt[i] );
    }

    // triangular according to opencv 
    vector<Point3d> Pw3dCV;
    vector<Point2d> nRKpt, nCKpt;
    triangularByOpencv( goodRKpt, goodCKpt, Trw, Tcw, K, nRKpt, nCKpt, Pw3dCV);
        

    // reproject error 
    double errorOurs = 0, errorOpencv = 0;
    for( int i = 0; i < Pw3dCV.size(); ++i )
    {
        // our triangular result
        Point3d Pw = Pw3d[i];
        Point2d ppc = Point2d( Pw.x/Pw.z,Pw.y/Pw.z );

        // opencv triangular result
        Point3d Pcv = Pw3dCV[i];
        Point2d ppcv = Point2d( Pcv.x/Pcv.z, Pcv.y/Pcv.z );

        // feature coordinate in reference normalized plane 
        Point2d ppr = nRKpt[i];

        errorOurs += norm( ppc - ppr );
        errorOpencv += norm( ppcv - ppr );
    }
    cout << "total error_ours is " << errorOurs << endl;
    cout << "total error_opencv is " << errorOpencv << endl;
    cout << "average error_ours is " << errorOurs / Pw3d.size() << endl;
    cout << "average error_opencv is " << errorOpencv / Pw3dCV.size() << endl;

    // show the keypoints
    Mat currOut;
    drawKeypoints( currImg, ckpt, currOut, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    imshow( "CurrOut", currOut );
    waitKey(0);
    // show good matching result
    Mat goodMatchOut;
    drawMatches( refImg, rkpt, currImg, ckpt, matches, goodMatchOut );
    imshow( "goodMatchOut", goodMatchOut );
    waitKey(0);
    
    // transform 3d point to point cloud to imshow
    PointCloud<PointXYZRGB>::Ptr cloud( new PointCloud<PointXYZRGB> );
    for( int i = 0; i < Pw3d.size(); ++i )
    {   
        PointXYZRGB point;
        Point3d p = Pw3d[i];
        Point2d pixel = Pr2d[i];
        point.x = p.x;
        point.y = p.y;
        point.z = p.z;
        point.r = refImg.at<Vec3b>(pixel.y,pixel.x)[0];
        point.g = refImg.at<Vec3b>(pixel.y,pixel.x)[1];
        point.b = refImg.at<Vec3b>(pixel.y,pixel.x)[2];
        cloud->push_back( point );
    }
    cout << "the number of point cloud is " << cloud->size() << endl;
    visualization::CloudViewer viewer( "Viewer" );
    viewer.showCloud( cloud );
    while( !viewer.wasStopped() )
    {
        // loop loop loop~~~
    }
    
    cout << "success!" << endl;
    return 0;
}



// feature extraction and compute descriptors
void featureExtraction( const Mat& img, vector<KeyPoint>& kpt, Mat& desp )
{
    // set number of features to extract
    Ptr<FeatureDetector> detector = ORB::create( 10000 );
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    detector->detect( img, kpt );
    descriptor->compute( img, kpt, desp );
}

// feature matching by brute force method and find the good matching
void featureMatching( const Mat& rDesp, const Mat& cDesp, vector<DMatch>& matches, 
                        const vector<KeyPoint>& rKpt, const vector<KeyPoint>& cKpt,
                        vector<Point2d>& goodRKpt, vector<Point2d>& goodCKpt )
{
    // coarse matching by brute force method measured by hamming distance
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create( "BruteForce-Hamming" );
    vector<DMatch> initMatches;
    matcher->match( rDesp, cDesp, initMatches );
    cout << "finish initMatches!" << endl;

    // compute the maximum distance and the minimum distance
    double min_dist = min_element( initMatches.begin(), initMatches.end(), 
				       [](const DMatch& m1, const DMatch& m2) {return m1.distance<m2.distance;} )->distance;
    double max_dist = max_element( initMatches.begin(), initMatches.end(), 
				       [](const DMatch& m1, const DMatch& m2) {return m1.distance<m2.distance;} )->distance;

    // find the good matching and record the according pixel coordinate
    for( int i = 0; i < initMatches.size(); ++i )
    {
        if( initMatches[i].distance <= max(min_dist*2, 30.0) )
        {
            matches.push_back( initMatches[i] );
            goodRKpt.push_back( rKpt[initMatches[i].queryIdx].pt );
            goodCKpt.push_back( cKpt[initMatches[i].trainIdx].pt );
        }
    }
}

// estimate the motion between the current frame and the reference frame
bool motionEstimation( const vector<Point2d>& goodRKpt, const vector<Point2d>& goodCKpt,
                       const Mat& K, Mat& Rcr, Mat& tcr )
{
    // compute the essential matrix according to Epipolar geometry
    Mat E = findEssentialMat( goodRKpt, goodCKpt, K, RANSAC );
    // recover the pose from the essential matrix
    int inliers = recoverPose( E, goodRKpt, goodCKpt, K, Rcr, tcr );
    // make sure that we have enough inliers to triangular
    return inliers > 100;
}

// to compare the triangular error between our method and opencv
void triangularByOpencv( const vector<Point2d>& RKpt, const vector<Point2d>& CKpt, 
                         const Mat& Trw, const Mat& Tcw, const Mat& K, 
                         vector<Point2d>& nRKpt, vector<Point2d>& nCKpt,
                         vector<Point3d>& Pw3dCV )
{
    Mat pts_4d;

    // unproject the features to normalized plane
    for( int i = 0; i < RKpt.size(); ++i )
    {
        Point2d pr( (RKpt[i].x-K.at<double>(0,2))/K.at<double>(0,0),
                    (RKpt[i].y-K.at<double>(1,2))/K.at<double>(1,1) );
        Point2d pc( (CKpt[i].x-K.at<double>(0,2))/K.at<double>(0,0),
                    (CKpt[i].y-K.at<double>(1,2))/K.at<double>(1,1) );
        nRKpt.push_back( pr );
        nCKpt.push_back( pc );                
    }

    // triangular points
    triangulatePoints( Trw, Tcw, nRKpt, nCKpt, pts_4d );

    // recover the position from the homogenerous coordinates 
    for( int i = 0; i < pts_4d.cols; ++i )
    {
        Mat x = pts_4d.col(i);
        x /= x.at<double>(3,0);
        Point3d pcv( x.at<double>(0,0), x.at<double>(1,0), x.at<double>(2,0));
        Pw3dCV.push_back( pcv );
    }  

    cout << "size of pw3d = " << Pw3dCV.size() << endl;
}