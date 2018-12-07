#include "orb_extractor.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;
using namespace XIAOC;

int main( int argc, char** argv )
{
    cout << "Enter the ORBextractor module..." << endl;
    
    // help infomation
    if ( argc != 2 )
    {
        cout << "Please input ./orb_extractor image" << endl;
        return -1;
    }

    // -----grid based orb extractor
    int nfeatures = 1000;
    int nlevels = 8;
    float fscaleFactor = 1.2;
    float fIniThFAST = 20;
    float fMinThFAST = 7;
    // default parameters printf
    cout << "Default parameters are : " << endl;
    cout << "nfeature : " << nfeatures << ", nlevels : " << nlevels << ", fscaleFactor : " << fscaleFactor << endl;
    cout << "fIniThFAST : " << fIniThFAST << ", fMinThFAST : " << fMinThFAST << endl;

    // read image
    cout << "Read image..." << endl;
    Mat image = imread( argv[1], CV_LOAD_IMAGE_UNCHANGED );
    Mat grayImg, mask;
    cvtColor( image, grayImg, CV_RGB2GRAY );
    imshow( "grayImg", grayImg );
    waitKey( 0 );
    cout << "Read image finish!" << endl;

    // orb extractor initialize
    cout << "ORBextractor initialize..." << endl;
    ORBextractor* pORBextractor;
    pORBextractor = new ORBextractor( nfeatures, fscaleFactor, nlevels, fIniThFAST, fMinThFAST );
    cout << "ORBextractor initialize finished!" << endl;

    // orb extractor
    cout << "Extract orb descriptors..." << endl;
    Mat desc;
    vector<KeyPoint> kps;
    (*pORBextractor)( grayImg, mask, kps, desc );
    cout << "Extract orb descriptors finished!" << endl;
    cout << "The number of keypoints are = " << kps.size() << endl;

    // draw keypoints in output image
    Mat outImg;
    drawKeypoints( grayImg, kps, outImg, cv::Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    imshow( "GridOrbKpsImg", outImg );
    waitKey( 0 );
    cout << "Finished! Congratulations!" << endl;


    // ----original orb extractor for comparation
    // orb initialization 
    cout << "Using original orb extractor to extract orb descriptors for comparation." << endl;
    Ptr<ORB> orb_ = ORB::create( 1000, 1.2f, 8, 19 );

    // orb extract
    vector<KeyPoint> orb_kps;
    Mat orb_desc;
    orb_->detectAndCompute( grayImg, mask, orb_kps, orb_desc );

    // draw keypoints in output image
    Mat orbOutImg;
    drawKeypoints( grayImg, orb_kps, orbOutImg, cv::Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    imshow( "OrbKpsImg", orbOutImg );
    waitKey(0);

    // destroy all windows when you press any key
    //destroyAllWindows();

    return 0;
}