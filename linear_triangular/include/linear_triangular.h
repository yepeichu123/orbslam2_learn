
#ifndef LINEAR_TRIANGULAR_H
#define LINEAR_TRIANGULAR_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace XIAOC
{
	//	the LinearTriangular class include the linear triangular method,
	//	and some helper function
	class LinearTriangular
	{
	public:
		// constructor: to setup the camera parameters and the transformation
	    LinearTriangular( const cv::Mat& K, const cv::Mat& Trw, const cv::Mat& Tcw );

		// triangular the 3D point from the features in two views
		// input: 2D pr and 2D pc which is the pixel coordinate in reference frame and in current frame
		// output: 3D Pw which is 3D coordinate in world coordinate
	    bool TriangularPoint( const cv::Point2d& pr, const cv::Point2d& pc, cv::Point3d& Pw ); 

		// unproject the pixel point to the normalized plane
	    void UnprojectPixel( const cv::Point2d& pr, const cv::Point2d& pc);

		// construct the Matrix A according to our blog
	    void ConstructMatrixA( const cv::Point3d& Prn, const cv::Point3d& Pcn, 
								const cv::Mat& Trw, const cv::Mat& Tcw );

		//	solve the problem by computing SVD
	    bool CompBySVD( const cv::Mat& A, cv::Point3d& Pw );

		// check whether the angle between two views is suitable 
	    bool CheckCrossAngle( const cv::Point3d& Prn, const cv::Point3d& Pcn, 
								const cv::Mat& Trw, const cv::Mat& Tcw );

		// check whether the depth of point which we triangulated is right
		bool CheckDepth( const cv::Point3d& Pw3d );

	private:
		// camera parameters
	    cv::Mat mK_;
		// transformation of current frame and reference frame
	    cv::Mat mTrw_, mTcw_;
		// the coefficient matrix
	    cv::Mat mA_;
		// 3D coordinate in world coordinate
	    cv::Point3d mPw_;
		// 3D normalized coordinate in camera normalized plane
	    cv::Point3d mPrn_, mPcn_;
		// the pixel coordinate
	    cv::Point2d mpr_, mpc_;
	};
}


#endif //LINEAR_TRIANGULAR_H