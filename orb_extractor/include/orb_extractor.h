
#ifndef ORB_EXTRACTOR_H
#define ORB_EXTRACTOR_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>
#include <list>

namespace XIAOC
{
  // to divide the images
  class ExtractorNode
  {
  public:
    ExtractorNode( ) : bNoMore( false ){ }
    
    void DivideNode( ExtractorNode& n1, ExtractorNode& n2, ExtractorNode& n3, ExtractorNode& n4 ); 
    
    // to save keypoints in the current node
    std::vector<cv::KeyPoint> vKeys;
    // to set the boundary, divide the image into four parts
    cv::Point2i UL, UR, BL, BR;
    // to save the subnodes
    std::list< ExtractorNode >::iterator lit;
    // check if the only keypoint in the current node
    bool bNoMore;
  };

  // to extract orb descriptors
  class ORBextractor
  {
  public:
    
    enum { HARRIS_SCORE = 0, FAST_SCORE = 1 };
    
    ORBextractor( int nfeatures, float scaleFactor, int nlevels, int iniThFAST, int minThFAST );
    
    ~ORBextractor( ){ }
    
     void operator( )( cv::InputArray image, cv::InputArray mask, std::vector< cv::KeyPoint>& keypoints,
       cv::OutputArray descriptors );
     
     std::vector< cv::Mat > mvImagePyramid;
     
     int inline GetLevels( )
     {
       return nlevels;
    }
    
    float inline GetScaleFactor( )
    {
      return scaleFactor;
    }
    
    std::vector< float > inline GetScaleFactors( )
    {
      return mvScaleFactor;
    }
     
     std::vector< float > inline GetInvScaleFactor( )
     {
       return mvInvScaleFactor;
    }
    
    std::vector< float > inline GetLevelSigmaSquares( )
    {
      return mvLevelSigma2;
    }
    
    std::vector< float > inline GetInvLevelSigmaSquares( )
    {
      return mvInvLevelSigma2;
    }
    
  protected:
    
    void ComputePyramid( cv::Mat image );
    
    void ComputeKeyPointsOctTree( std::vector< std::vector< cv::KeyPoint> >& allkeypoints );
    
    std::vector< cv::KeyPoint > DistributeOctTree( const std::vector< cv::KeyPoint >& vToDistributeKeys, 
      const int& minX, const int& maxX, const int& minY, const int& maxY, const int& nFeatures, const int& level );
    
    // to save BRIFE descriptors' pattern  
    std::vector< cv::Point2i > pattern;
    int nfeatures;
    double scaleFactor;
    int nlevels;
    int iniThFAST;
    int minThFAST;
    
    // number of features of each level 
    std::vector< int >  mnFeaturesPerLevel;
    // to set the local block border
    std::vector< int > umax;
    // the scale factor of pyramid
    std::vector< float > mvScaleFactor;
    std::vector< float > mvInvScaleFactor;
    std::vector< float > mvLevelSigma2;
    std::vector< float > mvInvLevelSigma2;
  };
  
}


#endif