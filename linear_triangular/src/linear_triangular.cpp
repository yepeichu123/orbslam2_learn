
#include "linear_triangular.h"
#include <iostream>

// constructor: to setup the camera parameters and the transformation
XIAOC::LinearTriangular::LinearTriangular(const cv::Mat& K, const cv::Mat& Trw, const cv::Mat& Tcw):
    mK_(K), mTrw_(Trw), mTcw_(Tcw)
{
    
}

// triangular the 3D point from the features in two views
// input: 2D pr and 2D pc which is the pixel coordinate in reference frame and in current frame
// output: 3D Pw which is 3D coordinate in world coordinate
bool XIAOC::LinearTriangular::TriangularPoint(const cv::Point2d& pr, const cv::Point2d& pc, cv::Point3d& Pw )
{
    // unproject to get the normal coordinate
    UnprojectPixel( pr, pc );
    // construct the A matrix
    ConstructMatrixA( mPrn_, mPcn_, mTrw_, mTcw_ );
    // get the 3D position
    if( CompBySVD( mA_, mPw_ ) )
    {
        Pw = mPw_;
        return true;
    }
    return false;
}

// unproject the pixel point to the normalized plane
void XIAOC::LinearTriangular::UnprojectPixel(const cv::Point2d& pr, const cv::Point2d& pc)
{
    // X = (u-cx)/fx;
    // Y = (v-cy)/fy;
    mPrn_.x = (pr.x - mK_.at<double>(0,2))/mK_.at<double>(0,0);
    mPrn_.y = (pr.y - mK_.at<double>(1,2))/mK_.at<double>(1,1);
    mPrn_.z  = 1;
    
    mPcn_.x = (pc.x - mK_.at<double>(0,2))/mK_.at<double>(0,0);
    mPcn_.y = (pc.y - mK_.at<double>(1,2))/mK_.at<double>(1,1);
    mPcn_.z = 1;
}

// construct the Matrix A according to our blog
void XIAOC::LinearTriangular::ConstructMatrixA(const cv::Point3d& Prn, const cv::Point3d& Pcn, const cv::Mat& Trw, const cv::Mat& Tcw )
{
    
     // ORBSLAM method
    /* cv::Mat A(4,4,CV_64F);
     A.row(0) = Prn.x*Trw.row(2)-Trw.row(0);
     A.row(1) = Prn.y*Trw.row(2)-Trw.row(1);
     A.row(2) = Pcn.x*Tcw.row(2)-Tcw.row(0);
     A.row(3) = Pcn.y*Tcw.row(2)-Tcw.row(1);
     A.copyTo( mA_ );
     */

     // Original method in my blog    
     cv::Mat PrnX = (cv::Mat_<double>(3,3) << 0, -Prn.z, Prn.y, 
									 Prn.z, 0, -Prn.x,
									 -Prn.y, Prn.x, 0);
     cv::Mat PcnX = (cv::Mat_<double>(3,3) << 0, -Pcn.z, Pcn.y, 
									 Pcn.z, 0, -Pcn.x,
									 -Pcn.y, Pcn.x, 0);
     // A = [prX*Trw; pcX*Tcw ]
     // APw = 0
     cv::Mat B = PrnX * Trw;
     cv::Mat C = PcnX * Tcw;
     cv::vconcat( B, C, mA_ );
     
}

//	solve the problem by computing SVD
bool XIAOC::LinearTriangular::CompBySVD(const cv::Mat& A, cv::Point3d& Pw )
{
    if( A.empty() )
    {
	    return false;
    }
    // compute the SVD of matrix A
    cv::Mat w, u, vt;
    cv::SVD::compute( A, w, u, vt, cv::SVD::MODIFY_A|cv::SVD::FULL_UV );
    
    // normalization
    cv::Mat Pw3d;
    Pw3d = vt.row(3).t();
    if( Pw3d.at<double>(3) == 0 )
    {
	    return false;
    }
    Pw3d = Pw3d.rowRange(0,3)/Pw3d.at<double>(3);
    
    // save the value of the position
    Pw.x = Pw3d.at<double>(0);
    Pw.y = Pw3d.at<double>(1);
    Pw.z = Pw3d.at<double>(2);

    return true;
}

// Check the angle of features between two views 
bool XIAOC::LinearTriangular::CheckCrossAngle( const cv::Point3d& Prn, const cv::Point3d& Pcn, 
								const cv::Mat& Trw, const cv::Mat& Tcw )
{
    // Todo: to check whether is suitable to triangular by angle
}

// check whether the depth of point which we triangulated is right
bool XIAOC::LinearTriangular::CheckDepth( const cv::Point3d& Pw3d )
{
    // Todo: to chech whether is suitable to accept by depth
}
