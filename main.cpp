#include <QCoreApplication>
#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv/cv.h"
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/objdetect/objdetect.hpp>
/*#if CV_MAJOR_VERSION >= 2 && CV_MINOR_VERSION >= 4
  #include <opencv2/nonfree/features2d.hpp>
#endif*/
#include <opencv2/nonfree/features2d.hpp>
#include "opencv2/nonfree/nonfree.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include <QDir>
#include <QFile>
#include <QTextStream>
#include "FalsePositives.h"
#include "CumulativeData.h"
//#define DEBUG 1
#define NO_IMAGES 5
#define NO_AFFINE 5
#define NO_PERSPECTIVE 5
#define NO_SIM_L1 7
#define NO_SIM_L2 4
#define NO_SCALED 6
#define NO_ROTATED 7

QString pathReference("J:\\iit\\3rd sem\\eel806_vision\\project\\dataset\\reference\\%1.jpg");
QString pathAffine("J:\\iit\\3rd sem\\eel806_vision\\project\\dataset\\affine\\%1_%2.jpg");
QString pathSimilarity("J:\\iit\\3rd sem\\eel806_vision\\project\\dataset\\similarity\\%1_%2_%3.jpg");
QString pathPerspective("J:\\iit\\3rd sem\\eel806_vision\\project\\dataset\\perspective\\%1_%2.jpg");
QString pathRotated("J:\\iit\\3rd sem\\eel806_vision\\project\\dataset\\rotated\\%1_%2.jpg");
QString pathScaled("J:\\iit\\3rd sem\\eel806_vision\\project\\dataset\\scaled\\%1_%2.jpg");

QString pathReference1("J:\\iit\\3rd sem\\eel806_vision\\project\\dataset\\reference\\");
QString pathAffine1("J:\\iit\\3rd sem\\eel806_vision\\project\\dataset\\affine\\");
QString pathSimilarity1("J:\\iit\\3rd sem\\eel806_vision\\project\\dataset\\similarity\\");
QString pathPerspective1("J:\\iit\\3rd sem\\eel806_vision\\project\\dataset\\perspective\\");
QString pathRotated1("J:\\iit\\3rd sem\\eel806_vision\\project\\dataset\\rotated\\");
QString pathScaled1("J:\\iit\\3rd sem\\eel806_vision\\project\\dataset\\scaled\\");

QString resultsAffine("resultsAffine.txt");
QString resultsPerspective("resultsPerspective.txt");
QString resultsScaled("resultsScaled.txt");
QString resultsRotated("resultsRotated.txt");
QString resultsSimilarity("resultsSimilarity.txt");

QFile fileAffine(pathAffine1+resultsAffine);
QFile filePerspective(pathPerspective1+resultsPerspective);
QFile fileScaled(pathScaled1+resultsScaled);
QFile fileRotated(pathRotated1+resultsRotated);
QFile fileSimilarity(pathSimilarity1+resultsSimilarity);

QTextStream outAffine(&fileAffine);
QTextStream outPerspective(&filePerspective);
QTextStream outScaled(&fileScaled);
QTextStream outRotated(&fileRotated);
QTextStream outSimilarity(&fileSimilarity);

using namespace std;
using namespace cv;

void countKeypoints(char *method, Mat image, FileStorage fs, QString filename)
{
    cv::Ptr<FeatureDetector> featureDetector = cv::FeatureDetector::create(method);
    std::vector<KeyPoint> keypoints;
    featureDetector->detect(image, keypoints);
    //FileStorage fs("CountKeypoints.yml", FileStorage::WRITE);
    fs << "image_" + filename.toStdString() << (int)keypoints.size();
}

void saveKeypoints(QDir dir, FileStorage fs, char *method)
{
    dir.setFilter(QDir::Files);
    QFileInfoList file = dir.entryInfoList();

    for(int i = 0; i < file.size(); i++)
    {
        QFileInfo f = file.at(i);
        Mat m = imread(f.absoluteFilePath().toStdString());
       // cout << f.absoluteFilePath().toStdString() << endl;
        countKeypoints(method, m,fs,f.baseName());
    }
}

void matchFeatures(char *method, Mat image, Mat target, QString targetTransform, FileStorage matching, QString targetBaseName)
{

    cv::Ptr<FeatureDetector> featureDetector = cv::FeatureDetector::create(method);
    std::vector<KeyPoint> keypoints;
    cv::Mat descriptors;
    featureDetector->detect(image, keypoints);
    cv::Ptr<DescriptorExtractor> featureExtractor = cv::DescriptorExtractor::create(method);
    featureExtractor->compute(image, keypoints, descriptors);

    cv::Mat descriptors1;
    std::vector<KeyPoint> keypoints1;
    featureDetector->detect(target, keypoints1); // NOTE: featureDetector is a pointer hence the '->'.
    featureExtractor->compute(target, keypoints1, descriptors1);

    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( descriptors, descriptors1, matches );

    double max_dist = 0; double min_dist = 100;

    for( int i = 0; i < descriptors.rows; i++ )
    { double dist = matches[i].distance;
      if( dist < min_dist ) min_dist = dist;
      if( dist > max_dist ) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist );
    printf("-- Min dist : %f \n", min_dist );

    std::vector< DMatch > good_matches;

    for( int i = 0; i < descriptors.rows; i++ )
    { if( matches[i].distance <= 2*min_dist )
      { good_matches.push_back( matches[i]); }
    }



#ifdef DEBUG
    Mat img_matches;
    drawMatches( image, keypoints, target, keypoints1,
                 good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                 vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
#endif
    //imshow( "Good Matches", img_matches );

    float percentMatching = ((float)good_matches.size()/(float)keypoints.size());
    //cout << targetTransform.toStdString() <<  " " << targetBaseName.toStdString()<<endl;
    matching << (targetTransform + QString::fromStdString("_") + targetBaseName).toStdString() << percentMatching;
    //Mat H = findHomography( obj, scene, CV_RANSAC );
}

QString getTargetFileName(QString targetTransform)
{
    if( targetTransform == "affine" )
        return pathAffine;
    else if( targetTransform == "perspective" )
        return pathPerspective;
    else if( targetTransform == "rotated" )
        return pathRotated;
    else if( targetTransform == "scaled" )
        return pathScaled;
    else if(targetTransform == "similarity" )
        return pathSimilarity;
    else if(targetTransform == "reference" )
        return pathReference;
}

int getImageCount(QString targetTransform)
{
    if( targetTransform == "affine" )
        return NO_AFFINE;
    else if( targetTransform == "perspective" )
        return NO_PERSPECTIVE;
    else if( targetTransform == "rotated" )
        return NO_ROTATED;
    else if( targetTransform == "scaled" )
        return NO_SCALED;
    else if(targetTransform == "similarity" )
        return 0;
    else if(targetTransform == "reference" )
        return NO_IMAGES;
}

/**
 * @brief initMatching
 * @param image
 * @param dir
 * @param method
 * @param targetTransform
 * @param fs
 * @param numI
 * For non similarity
 */
void initMatching(Mat image, QDir dir, char *method, QString targetTransform, FileStorage fs, int numI)
{
    dir.setFilter(QDir::Files);
    QFileInfoList file = dir.entryInfoList();

    int n = getImageCount(targetTransform);
    QString targetFileName = getTargetFileName(targetTransform);
    for(int i = 1; i <= n; i++)
    {
        cout << i << " " << n << endl;
       // QFileInfo f = file.at(i);


        cout << "Reading: " << targetFileName.arg(numI).arg(i).toStdString()<<endl;

        Mat m = imread(targetFileName.arg(numI).arg(i).toStdString());
        //Mat m = imread(f.absoluteFilePath().toStdString());
       // cout << f.absoluteFilePath().toStdString() << endl;

        QString f = QString::number( numI ) + QString::fromStdString("_") + QString::number(i);
       // cout << f.toStdString()<<endl;
        matchFeatures(method,image, m,targetTransform,fs,f);
    }
}

Mat getRotationTransform(Mat source, int type)
{
    double angle = 0.0;
    if(type == 6)
    {
        angle = 90.0;
    }
    else if(type == 7)
    {
        angle = 180.0;
    }
    else
        angle = 10.0*type;

    Point2f src_center(source.cols/2.0F, source.rows/2.0F);
    Mat rot_mat = getRotationMatrix2D(src_center, angle, 1.0);
    return rot_mat;
}

Mat getPerspectiveTransform(Mat src, int type)
{
    Point2f srcTri[4];
    Point2f dstTri[4];
    float xDest[4]; float yDest[4];
    Mat  warp_dst;
    warp_dst = Mat::zeros( src.rows, src.cols, src.type() );

    if(type == 1)
    {
        xDest[0] = 0; xDest[1] = 0.9; xDest[2] = 0; xDest[3] = 0.9;
        yDest[0] = 0.1; yDest[1] = 0.1; yDest[2] = 0.9; yDest[3] = 0.9;
    }
    else if(type == 2)
    {
        xDest[0] = 0.3; xDest[1] = 0.7; xDest[2] = 0.3; xDest[3] = 0.7;
        yDest[0] = 0.3; yDest[1] = 0.3; yDest[2] = 0.7; yDest[3] = 0.7;
    }
    else if(type == 3)
    {
        xDest[0] = 0.0; xDest[1] = 1.0; xDest[2] = 0.0; xDest[3] = 1.0;
        yDest[0] = 0.2; yDest[1] = 0.2; yDest[2] = 0.9; xDest[3] = 0.9;
    }

    else if(type == 4)
    {
        xDest[0] = 0.5; xDest[1] = 1.0; xDest[2] = 0.0; xDest[3] = 0.5;
        yDest[0] = 0.0; yDest[1] = 0.5; yDest[2] = 0.5; yDest[3] = 1.0;
    }
    else if(type == 5)
    {
        xDest[0] = 0.0; xDest[1] = 0.8; xDest[2] = 0.2; xDest[3] = 0.7;
        yDest[0] = 0.4; yDest[1] = 0.2; yDest[2] = 0.7; yDest[3] = 0.6;
    }
    srcTri[0] = Point2f( 0,0 );
    srcTri[1] = Point2f( src.cols - 1, 0 );
    srcTri[2] = Point2f( 0, src.rows - 1 );
    srcTri[3] = Point2f( src.cols -1 , src.rows - 1 );

    dstTri[0] = Point2f( src.cols*xDest[0], src.rows*yDest[0] );
    dstTri[1] = Point2f( src.cols*xDest[1], src.rows*yDest[1] );
    dstTri[2] = Point2f( src.cols*xDest[2], src.rows*yDest[2] );
    dstTri[3] = Point2f( src.cols*xDest[3], src.rows*yDest[3] );
    return getPerspectiveTransform( srcTri, dstTri );

}

Mat getAffineTransform(Mat src, int type)
{
    int xSrc[3], ySrc[3]; float xDest[3], yDest[3];
    xSrc[0] = 0; xSrc[1] = src.cols-1; xSrc[2] = 0;
    ySrc[0] = 0; ySrc[1] = 0; ySrc[2] = src.rows-1;
    if(type == 1)
    {
        xDest[0] = 0.0; xDest[1] = 0.9; xDest[2] = 0.0;
        yDest[0] = 0.1; yDest[1] = 0.1; yDest[2] = 0.9;


    }
    else if(type == 2)
    {
        xDest[0] = 0.3; xDest[1] = 0.7; xDest[2] = 0.3;
        yDest[0] = 0.3; yDest[1] = 0.3; yDest[2] = 0.7;
    }
    else if(type == 3)
    {
        xDest[0] = 0.0; xDest[1] = 1.0; xDest[2] = 0.0;
        yDest[0] = 0.2; yDest[1] = 0.2; yDest[2] = 0.9;
    }
    else if(type == 4)
    {
        xDest[0] = 0.5; xDest[1] = 1.0; xDest[2] = 0.0;
        yDest[0] = 0.0; yDest[1] = 0.5; yDest[2] = 0.5;
    }
    else if(type == 5)
    {
        xDest[0] = 0.0; xDest[1] = 0.8; xDest[2] = 0.2;
        yDest[0] = 0.4; yDest[1] = 0.2; yDest[2] = 0.7;
    }
    Point2f srcTri[3];
    Point2f dstTri[3];
    srcTri[0] = Point2f( xSrc[0],ySrc[0] );
    srcTri[1] = Point2f( xSrc[1], ySrc[1] );
    srcTri[2] = Point2f( xSrc[2], ySrc[2] );

    dstTri[0] = Point2f( src.cols*xDest[0], src.rows*yDest[0] );
    dstTri[1] = Point2f( src.cols*xDest[1], src.rows*yDest[1] );
    dstTri[2] = Point2f( src.cols*xDest[2], src.rows*yDest[2] );

    Mat warp_mat = getAffineTransform( srcTri, dstTri );
    return warp_mat;
}

void fpMatchingAccuracyAffine(Mat image, Mat target, char *method, int type, int min_t, int &fm, int &f1, int &f2)
{
    //type = 4;
    int t = min_t;
    outAffine << "Threshold: " << min_t<<endl;
    float threshold_r = 2.12; // 3x3
    float threshold_r1 = 6.36; // 9x9
//    Mat image = imread(pathReference.arg(1).toStdString());
//    Mat target = imread(pathAffine.arg(1).arg(4).toStdString());

    cv::Ptr<FeatureDetector> featureDetector = cv::FeatureDetector::create(method);
    std::vector<KeyPoint> keypoints;
    cv::Mat descriptors;
    featureDetector->detect(image, keypoints);
    cv::Ptr<DescriptorExtractor> featureExtractor = cv::DescriptorExtractor::create(method);
    featureExtractor->compute(image, keypoints, descriptors);

    cv::Mat descriptors1;
    std::vector<KeyPoint> keypoints1;
    featureDetector->detect(target, keypoints1); // NOTE: featureDetector is a pointer hence the '->'.
    featureExtractor->compute(target, keypoints1, descriptors1);

    cout << "Total Keypoints " << keypoints1.size() << endl;
    outAffine << "Total Keypoints: " << keypoints.size() << endl;

    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( descriptors, descriptors1, matches );

    double max_dist = 0; double min_dist = 100;

    for( int i = 0; i < descriptors.rows; i++ )
    { double dist = matches[i].distance;
      if( dist < min_dist ) min_dist = dist;
      if( dist > max_dist ) max_dist = dist;
    }

   // printf("-- Max dist : %f \n", max_dist );
    //printf("-- Min dist : %f \n", min_dist );

    std::vector< DMatch > good_matches;

    float threshold = t*min_dist;
    for( int i = 0; i < descriptors.rows; i++ )
    { if( matches[i].distance <= t*min_dist )
      { good_matches.push_back( matches[i]); }
    }

    Mat warp_mat = getAffineTransform(image, type);
    warp_mat.convertTo(warp_mat,CV_32FC1,1,0);
    std::vector<FalsePositives> fp;
    int countMatched_r = 0;
    int countMatched_r1 = 0;
   // int threshold  = 0.5;

   cout << "Keypoints Matched: " << good_matches.size()<<endl;
   outAffine << "Keypoints Matched: " << good_matches.size()<<endl;
   fm = (int)good_matches.size();

   for( int i = 0; i < good_matches.size(); i++ )
   {
       Point2f f = keypoints[good_matches[i].queryIdx].pt;
       Point2f f1 = keypoints1[good_matches[i].trainIdx].pt;

       vector<Point3f> vec;
       vec.push_back(Point3f(f.x,f.y,1));

       Mat srcMat = Mat(vec).reshape(1).t();
       Mat dstMat = warp_mat*srcMat; //USE MATRIX ALGEBRA

       Point2f dst=Point2f(dstMat.at<float>(0,0),dstMat.at<float>(1,0));

       double distance = norm(dst-f1);
       fp.push_back(FalsePositives(f,dst,f1, norm(dst-f1)));
       if(distance <= threshold_r)
       {
           countMatched_r++;
       }
       if(distance <= threshold_r1)
       {
           countMatched_r1++;
       }
   }
   cout << "Matched after Removing falsepositives(3x3): " << countMatched_r << endl;
   cout << "Matched after Removing falsepositives(9x9): " << countMatched_r1 << endl;

   f1 = countMatched_r;
   f2 = countMatched_r1;
   outAffine << "Matched after Removing falsepositives(3x3): " << countMatched_r << endl;
   outAffine << "Matched after Removing falsepositives(9x9): " << countMatched_r1 << endl;
   outAffine << "\n";
}

void fpMatchingAccuracyPerspective(Mat image, Mat target, char *method, int type, int min_t, int &fm, int &f1, int &f2)
{
     //type = 4;
    int t = min_t;

    float threshold_r = 2.12; // 3x3
    float threshold_r1 = 6.36; // 9x9

    outPerspective << "Threshold: " << min_t;

    cv::Ptr<FeatureDetector> featureDetector = cv::FeatureDetector::create(method);
    std::vector<KeyPoint> keypoints;
    cv::Mat descriptors;
    featureDetector->detect(image, keypoints);
    cv::Ptr<DescriptorExtractor> featureExtractor = cv::DescriptorExtractor::create(method);
    featureExtractor->compute(image, keypoints, descriptors);

    cv::Mat descriptors1;
    std::vector<KeyPoint> keypoints1;
    featureDetector->detect(target, keypoints1); // NOTE: featureDetector is a pointer hence the '->'.
    featureExtractor->compute(target, keypoints1, descriptors1);

    cout << "Total Keypoints " << keypoints1.size() << endl;
    outPerspective << "Total Keypoints " << keypoints1.size() << endl;

    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( descriptors, descriptors1, matches );

    double max_dist = 0; double min_dist = 100;

    for( int i = 0; i < descriptors.rows; i++ )
    { double dist = matches[i].distance;
      if( dist < min_dist ) min_dist = dist;
      if( dist > max_dist ) max_dist = dist;
    }

  //  printf("-- Max dist : %f \n", max_dist );
    //printf("-- Min dist : %f \n", min_dist );

    std::vector< DMatch > good_matches;

    for( int i = 0; i < descriptors.rows; i++ )
    { if( matches[i].distance <= t*min_dist )
      { good_matches.push_back( matches[i]); }
    }

    Mat warp_mat = getAffineTransform(image, type);
    warp_mat.convertTo(warp_mat,CV_32FC1,1,0);
    std::vector<FalsePositives> fp;
    int countMatched_r = 0;
    int countMatched_r1 = 0;

    cout << "Keypoints Matched: " << good_matches.size()<<endl;
    outPerspective << "Keypoints Matched: " << good_matches.size()<<endl;
    fm = (int)good_matches.size();

   for( int i = 0; i < good_matches.size(); i++ )
   {
       Point2f f = keypoints[good_matches[i].queryIdx].pt;
       Point2f f1 = keypoints1[good_matches[i].trainIdx].pt;

       vector<Point3f> vec;
       vec.push_back(Point3f(f.x,f.y,1));

       Mat srcMat = Mat(vec).reshape(1).t();
       Mat dstMat = warp_mat*srcMat; //USE MATRIX ALGEBRA

       Point2f dst=Point2f(dstMat.at<float>(0,0),dstMat.at<float>(1,0));

       double distance = norm(dst-f1);
       fp.push_back(FalsePositives(f,dst,f1, norm(dst-f1)));
       if(distance <= threshold_r)
       {
           countMatched_r++;
       }
       if(distance <= threshold_r1)
       {
           countMatched_r1++;
       }
   }
   cout << "Matched after Removing falsepositives(3x3): " << countMatched_r<<endl;
   cout << "Matched after Removing falsepositives(9x9): " << countMatched_r1<<endl;
   outPerspective << "Matched after Removing falsepositives(3x3): " << countMatched_r<<endl;
   outPerspective << "Matched after Removing falsepositives(9x9): " << countMatched_r1<<endl;
   outPerspective <<"\n";
   f1 = countMatched_r;
   f2 = countMatched_r1;
}

void fpMatchingAccuracyRotation(Mat image, Mat target, int type, char *method, int min_t, int &fm, int &f1, int &f2)
{
    int t = min_t;

    float threshold_r = 2.12; // 3x3
    float threshold_r1 = 6.36; // 9x9
    outRotated << "Threshold: " << min_t;

    cv::Ptr<FeatureDetector> featureDetector = cv::FeatureDetector::create(method);
    std::vector<KeyPoint> keypoints;
    cv::Mat descriptors;
    featureDetector->detect(image, keypoints);
    cv::Ptr<DescriptorExtractor> featureExtractor = cv::DescriptorExtractor::create(method);
    featureExtractor->compute(image, keypoints, descriptors);

    cv::Mat descriptors1;
    std::vector<KeyPoint> keypoints1;
    featureDetector->detect(target, keypoints1); // NOTE: featureDetector is a pointer hence the '->'.
    featureExtractor->compute(target, keypoints1, descriptors1);
    cout << "Total Keypoints " << keypoints1.size() << endl;
    outRotated << "Total Keypoints " << keypoints1.size() << endl;
    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( descriptors, descriptors1, matches );

    double max_dist = 0; double min_dist = 100;

    for( int i = 0; i < descriptors.rows; i++ )
    { double dist = matches[i].distance;
      if( dist < min_dist ) min_dist = dist;
      if( dist > max_dist ) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist );
    printf("-- Min dist : %f \n", min_dist );

    std::vector< DMatch > good_matches;


    for( int i = 0; i < descriptors.rows; i++ )
    { if( matches[i].distance <= t*min_dist )
      { good_matches.push_back( matches[i]); }
    }

    Mat warp_mat = getRotationTransform(image, type);
    warp_mat.convertTo(warp_mat,CV_32FC1,1,0);
    std::vector<FalsePositives> fp;
    int countMatched_r = 0;
    int countMatched_r1 = 0;


    cout << "Keypoints Matched: " << good_matches.size()<<endl;
    outRotated << "Keypoints Matched: " << good_matches.size()<<endl;
    fm = (int)good_matches.size();
   for( int i = 0; i < good_matches.size(); i++ )
   {
       Point2f f = keypoints[good_matches[i].queryIdx].pt;
       Point2f f1 = keypoints1[good_matches[i].trainIdx].pt;

       vector<Point3f> vec;
       vec.push_back(Point3f(f.x,f.y,1));

       Mat srcMat = Mat(vec).reshape(1).t();
       Mat dstMat = warp_mat*srcMat; //USE MATRIX ALGEBRA

       Point2f dst=Point2f(dstMat.at<float>(0,0),dstMat.at<float>(1,0));

       double distance = norm(dst-f1);
       fp.push_back(FalsePositives(f,dst,f1, norm(dst-f1)));
       if(distance <= threshold_r)
       {
           countMatched_r++;
       }
       if(distance <= threshold_r1)
       {
           countMatched_r1++;
       }
   }
   cout << "Matched after Removing falsepositives(3x3): " << countMatched_r<<endl;
   cout << "Matched after Removing falsepositives(9x9): " << countMatched_r1<<endl;

   outRotated << "Matched after Removing falsepositives(3x3): " << countMatched_r<<endl;
   outRotated << "Matched after Removing falsepositives(9x9): " << countMatched_r1<<endl;
   outRotated << "\n";
   f1 = countMatched_r;
   f2 = countMatched_r1;
}

float getScaledTransform(int type)
{
    float d[1][2];
    if(type == 1) {
        d[0][1] = d[0][0] = 0.125;
        return 0.125;
    }
    else if( type == 2)
    {
       //d[1][0] = d[0][0] = 0.25;
        return 0.25;
    }
    else if(type == 3)
    {
        //d[1][0] = d[0][0] = 0.5;
        return 0.5;
    }
    else if(type == 4)
    {
        //d[1][0] = d[0][0] = 0.75;
        return 0.75;
    }
    else if(type == 5)
    {
        //d[1][0] = d[0][0] = 2.0;
        return 2.0;
    }
    else if(type == 6)
    {
        //d[1][0] = d[0][0] = 4.0;
        return 4.0;
    }
    //return Mat(2, 1, CV_32FC1, &d);
}

void fpMatchingAccuracyScaled(Mat image, Mat target, int type, char *method, int min_t, int &fm, int &f1, int &f2)
{
    int t = min_t;

    float threshold_r = 2.12; // 3x3
    float threshold_r1 = 6.36; // 9x9
    outScaled << "Threshold: " << min_t;

    cv::Ptr<FeatureDetector> featureDetector = cv::FeatureDetector::create(method);
    std::vector<KeyPoint> keypoints;
    cv::Mat descriptors;
    featureDetector->detect(image, keypoints);
    cv::Ptr<DescriptorExtractor> featureExtractor = cv::DescriptorExtractor::create(method);
    featureExtractor->compute(image, keypoints, descriptors);

    cv::Mat descriptors1;
    std::vector<KeyPoint> keypoints1;
    featureDetector->detect(target, keypoints1); // NOTE: featureDetector is a pointer hence the '->'.
    featureExtractor->compute(target, keypoints1, descriptors1);
    cout << "Total Keypoints " << keypoints1.size() << endl;
    outScaled << "Total Keypoints " << keypoints1.size() << endl;

    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( descriptors, descriptors1, matches );

    double max_dist = 0; double min_dist = 100;

    for( int i = 0; i < descriptors.rows; i++ )
    { double dist = matches[i].distance;
      if( dist < min_dist ) min_dist = dist;
      if( dist > max_dist ) max_dist = dist;
    }

    //printf("-- Max dist : %f \n", max_dist );
    //printf("-- Min dist : %f \n", min_dist );

    std::vector< DMatch > good_matches;


    for( int i = 0; i < descriptors.rows; i++ )
    { if( matches[i].distance <= t*min_dist )
      { good_matches.push_back( matches[i]); }
    }

   // Mat warp_mat = getScaledTransform(image, type);
   // warp_mat.convertTo(warp_mat,CV_32FC1,1,0);
    std::vector<FalsePositives> fp;
    int countMatched_r = 0;
    int countMatched_r1 = 0;

    float s = getScaledTransform(type);
    cout << "Keypoints Matched: " << good_matches.size()<<endl;
    outScaled << "Keypoints Matched: " << good_matches.size()<<endl;
    fm = (int)good_matches.size();
   for( int i = 0; i < good_matches.size(); i++ )
   {
       Point2f f = keypoints[good_matches[i].queryIdx].pt;
       Point2f f1 = keypoints1[good_matches[i].trainIdx].pt;

       vector<Point3f> vec;
       vec.push_back(Point3f(f.x,f.y,1));

       Mat srcMat = Mat(vec).reshape(1).t();
       //Mat dstMat = warp_mat*srcMat; //USE MATRIX ALGEBRA
       // s = 1.0;
       Point2f dst=Point2f(f.x * s,f.y *s);

       double distance = norm(dst-f1);
       fp.push_back(FalsePositives(f,dst,f1, norm(dst-f1)));
       if(distance <= threshold_r)
       {
           countMatched_r++;
       }
       if(distance <= threshold_r1)
       {
           countMatched_r1++;
       }
   }
   cout << "Matched after Removing falsepositives(3x3): " << countMatched_r<<endl;
   cout << "Matched after Removing falsepositives(9x9): " << countMatched_r1<<endl;
   outScaled << "Matched after Removing falsepositives(3x3): " << countMatched_r<<endl;
   outScaled << "Matched after Removing falsepositives(9x9): " << countMatched_r1<<endl;
   outScaled << "\n";
   f1 = countMatched_r;
   f2 = countMatched_r1;
}


float getSimScaledTransform(int type)
{
    float s = 0.0;
    if(type == 1)
    {
        s = 0.25;
    }
    else if(type == 2)
    {
         s = 0.5;
    }
    else if(type == 3)
    {
        s = 2.0;
    }
    else if(type == 4)
    {
        s = 4.0;
    }
    return s;
}

void fpMatchingAccuracySimilarity(Mat image, Mat target, char *method, int rType, int sType, int min_t, int &fm, int &f1, int &f2)
{
     //type = 4;
    int t = min_t;

    float threshold_r = 2.12; // 3x3
    float threshold_r1 = 6.36; // 9x9

    outSimilarity << "Threshold: " << min_t;

    cv::Ptr<FeatureDetector> featureDetector = cv::FeatureDetector::create(method);
    std::vector<KeyPoint> keypoints;
    cv::Mat descriptors;
    featureDetector->detect(image, keypoints);
    cv::Ptr<DescriptorExtractor> featureExtractor = cv::DescriptorExtractor::create(method);
    featureExtractor->compute(image, keypoints, descriptors);

    cv::Mat descriptors1;
    std::vector<KeyPoint> keypoints1;
    featureDetector->detect(target, keypoints1); // NOTE: featureDetector is a pointer hence the '->'.
    featureExtractor->compute(target, keypoints1, descriptors1);

    cout << "Total Keypoints " << keypoints1.size() << endl;
    outSimilarity << "Total Keypoints " << keypoints1.size() << endl;

    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( descriptors, descriptors1, matches );

    double max_dist = 0; double min_dist = 100;

    for( int i = 0; i < descriptors.rows; i++ )
    { double dist = matches[i].distance;
      if( dist < min_dist ) min_dist = dist;
      if( dist > max_dist ) max_dist = dist;
    }

  //  printf("-- Max dist : %f \n", max_dist );
    //printf("-- Min dist : %f \n", min_dist );

    std::vector< DMatch > good_matches;

    for( int i = 0; i < descriptors.rows; i++ )
    { if( matches[i].distance <= t*min_dist )
      { good_matches.push_back( matches[i]); }
    }

    Mat warp_rot = getRotationTransform(image,rType);
    warp_rot.convertTo(warp_rot,CV_32FC1,1,0);

    float warp_s= getSimScaledTransform(sType);
    std::vector<FalsePositives> fp;
    int countMatched_r = 0;
    int countMatched_r1 = 0;

    cout << "Keypoints Matched: " << good_matches.size()<<endl;
    outSimilarity << "Keypoints Matched: " << good_matches.size()<<endl;
    fm = (int)good_matches.size();

   for( int i = 0; i < good_matches.size(); i++ )
   {
       Point2f f = keypoints[good_matches[i].queryIdx].pt;
       Point2f f1 = keypoints1[good_matches[i].trainIdx].pt;

       vector<Point3f> vec;
       vec.push_back(Point3f(f.x,f.y,1));

       Mat srcMat = Mat(vec).reshape(1).t();
       Mat dstMat = warp_rot*srcMat; //USE MATRIX ALGEBRA

       Point2f dst=Point2f(dstMat.at<float>(0,0)*warp_s,dstMat.at<float>(1,0)*warp_s);

       double distance = norm(dst-f1);
       fp.push_back(FalsePositives(f,dst,f1, norm(dst-f1)));
       if(distance <= threshold_r)
       {
           countMatched_r++;
       }
       if(distance <= threshold_r1)
       {
           countMatched_r1++;
       }
   }
   cout << "Matched after Removing falsepositives(3x3): " << countMatched_r<<endl;
   cout << "Matched after Removing falsepositives(9x9): " << countMatched_r1<<endl;
   outSimilarity << "Matched after Removing falsepositives(3x3): " << countMatched_r<<endl;
   outSimilarity << "Matched after Removing falsepositives(9x9): " << countMatched_r1<<endl;
   outSimilarity <<"\n";
   f1 = countMatched_r;
   f2 = countMatched_r1;
}

void runSimALL(char *method)
{

    outSimilarity << "Method: " << method << endl;

    CumulativeData d2;
    CumulativeData d5;
    CumulativeData d10;

    for(int i = 1; i <= NO_IMAGES; i++)
    {
        cout << "Image No: " << i;
        Mat image = imread(pathReference.arg(i).toStdString());

        for(int j = 1; j <= NO_SIM_L1; j++)
        {
            for(int k = 1; k <= NO_SIM_L2; k++)
            {
                outSimilarity << i << "_" << j << "_" << k << endl;
                Mat target = imread(pathSimilarity.arg(i).arg(j).arg(k).toStdString());
                int fm,f1,f2;
                fm = f1 = f2 = 0;
                fpMatchingAccuracySimilarity(image, target, method, j,k,2,fm,f1,f2);
                d2.matchedFeatures += fm;
                d2.fp_1 += f1;
                d2.fp_2 += f2;
                fpMatchingAccuracySimilarity(image, target, method, j,k,5,fm,f1,f2);
                d5.matchedFeatures += fm;
                d5.fp_1 += f1;
                d5.fp_2 += f2;
                fpMatchingAccuracySimilarity(image, target, method, j,k,10,fm,f1,f2);
                d10.matchedFeatures += fm;
                d10.fp_1 += f1;
                d10.fp_2 += f2;
            }
        }
    }

    outSimilarity <<"\n.......................\n";
    outSimilarity << "T2: Matched Features: " << d2.matchedFeatures << " FP 3x3 " << d2.fp_1 << " FP 9x9 " << d2.fp_2<<endl;
    outSimilarity << "T5: Matched Features: " << d5.matchedFeatures << " FP 3x3 " << d5.fp_1 << " FP 9x9 " << d5.fp_2<<endl;
    outSimilarity << "T10: Matched Features: " << d10.matchedFeatures << " FP 3x3 " << d10.fp_1 << " FP 9x9 " << d10.fp_2 <<endl;
    outSimilarity <<"\n.......................\n";
}



void runAll(char *method)
{

    outAffine << "Method: " << method<<endl;
    outPerspective << "Method: " << method<<endl;
    outRotated << "Method: " << method<<endl;
    outScaled << "Method: " << method<<endl;

    CumulativeData d2;
    CumulativeData d5;
    CumulativeData d10;

    CumulativeData p2;
    CumulativeData p5;
    CumulativeData p10;

    CumulativeData s2;
    CumulativeData s5;
    CumulativeData s10;

    CumulativeData r2;
    CumulativeData r5;
    CumulativeData r10;

    for(int j = 1; j <= NO_IMAGES; j++)
    {
        cout << "Image No: " << j;
        Mat image = imread(pathReference.arg(j).toStdString());

        outAffine <<"Image No: " << j<<endl;
        for(int i = 1; i <= NO_AFFINE; i++ )
        {
            outAffine << j << "_" <<i<<endl;
            Mat target = imread(pathAffine.arg(j).arg(i).toStdString());

            int fm, f1,f2; fm = f1 = f2 = 0;
            fpMatchingAccuracyAffine(image, target,method,i,2,fm,f1,f2);
            d2.matchedFeatures += fm;
            d2.fp_1 += f1;
            d2.fp_2 += f2;
            fpMatchingAccuracyAffine(image, target, method,i,5,fm,f1,f2);
            d5.matchedFeatures += fm;
            d5.fp_1 += f1;
            d5.fp_2 += f2;
            fpMatchingAccuracyAffine(image, target, method,i,10,fm,f1,f2);
            d10.matchedFeatures += fm;
            d10.fp_1 += f1;
            d10.fp_2 += f2;
        }
        cout << "FINISHED with AFFINE"<<endl;

        //Rotation
        outRotated << "Image No: " << j<<endl;
        for(int i = 1; i <= NO_ROTATED; i++)
        {
            cout << j << "_" <<i<<endl;
            outRotated << j << "_" <<i<<endl;
            //outPerspective << QString::number(j).toStdString() << QString::fromStdString("_") << QString::number(i).toStdString()<<endl;
            Mat target = imread(pathRotated.arg(j).arg(i).toStdString());
             int fm, f1,f2; fm = f1 = f2 = 0;
            fpMatchingAccuracyRotation(image, target, i,method,2, fm,f1,f2);
            r2.matchedFeatures += fm;
            r2.fp_1 += f1;
            r2.fp_2 += f2;
            fpMatchingAccuracyRotation(image, target, i,method,5,fm,f1,f2);
            r5.matchedFeatures += fm;
            r5.fp_1 += f1;
            r5.fp_2 += f2;
            fpMatchingAccuracyRotation(image, target, i,method,10,fm,f1,f2);
            r10.matchedFeatures += fm;
            r10.fp_1 += f1;
            r10.fp_2 += f2;
        }
        cout << "FINISHED with ROTATION"<<endl;


        outPerspective << "Image No: " << j<<endl;
        for(int i = 1; i <= NO_PERSPECTIVE; i++)
        {
            cout << j << "_" <<i<<endl;
            outPerspective << j << "_" <<i<<endl;
            Mat target = imread(pathPerspective.arg(j).arg(i).toStdString());
            int fm, f1,f2; fm = f1 = f2 = 0;
            fpMatchingAccuracyPerspective(image, target,method,i,2, fm,f1,f2);
            p2.matchedFeatures += fm;
            p2.fp_1 += f1;
            p2.fp_2 += f2;
            fpMatchingAccuracyPerspective(image, target, method,i,5,fm,f1,f2);
            p5.matchedFeatures += fm;
            p5.fp_1 += f1;
            p5.fp_2 += f2;
            fpMatchingAccuracyPerspective(image, target, method,i,10,fm,f1,f2);
            p10.matchedFeatures += fm;
            p10.fp_1 += f1;
            p10.fp_2 += f2;
        }
        cout << "FINISHED with PERSPECTIVE"<<endl;
        outScaled << "Image No: " << j<<endl;
        for(int i = 1; i <= NO_SCALED; i++)
        {
            cout << j << "_" <<i<<endl;
            outScaled << j << "_" <<i<<endl;
            Mat target = imread(pathScaled.arg(j).arg(i).toStdString());
             int fm, f1,f2; fm = f1 = f2 = 0;
            fpMatchingAccuracyScaled(image, target, i,method,2,fm,f1,f2);
            s2.matchedFeatures += fm;
            s2.fp_1 += f1;
            s2.fp_2 += f2;
            fpMatchingAccuracyScaled(image, target, i,method,5,fm,f1,f2);
            s5.matchedFeatures += fm;
            s5.fp_1 += f1;
            s5.fp_2 += f2;
            fpMatchingAccuracyScaled(image, target, i,method,10,fm,f1,f2);
            s10.matchedFeatures += fm;
            s10.fp_1 += f1;
            s10.fp_2 += f2;
        }
        cout << "FINISHED with SCALED"<<endl;
    }

    outAffine <<"\n.......................\n";
    outAffine << "T2: Matched Features: " << d2.matchedFeatures<< " FP 3x3 " << d2.fp_1 << " FP 9x9 " << d2.fp_2<<endl;
    outAffine << "T5: Matched Features: " << d5.matchedFeatures<< " FP 3x3 " << d5.fp_1 << " FP 9x9 " << d5.fp_2<<endl;
    outAffine << "T10: Matched Features: " << d10.matchedFeatures<< " FP 3x3 " << d10.fp_1 << " FP 9x9 " << d10.fp_2 <<endl;
    outAffine <<"\n.......................\n";

    outRotated <<"\n.......................\n";
    outRotated << "T2: Matched Features: " << r2.matchedFeatures<< " FP 3x3 " << r2.fp_1 << " FP 9x9 " << r2.fp_2<<endl;
    outRotated << "T5: Matched Features: " << r5.matchedFeatures<< " FP 3x3 " << r5.fp_1 << " FP 9x9 " << r5.fp_2<<endl;
    outRotated << "T10: Matched Features: " << r10.matchedFeatures<< " FP 3x3 " << r10.fp_1 << " FP 9x9 " << r10.fp_2 <<endl;
    outRotated <<"\n.......................\n";

    outPerspective <<"\n.......................\n";
    outPerspective << "T2: Matched Features: " << p2.matchedFeatures<< " FP 3x3 " << p2.fp_1 << " FP 9x9 " << p2.fp_2<<endl;
    outPerspective << "T5: Matched Features: " << p5.matchedFeatures<< " FP 3x3 " << p5.fp_1 << " FP 9x9 " << p5.fp_2<<endl;
    outPerspective << "T10: Matched Features: " << p10.matchedFeatures<< " FP 3x3 " << p10.fp_1 << " FP 9x9 " << p10.fp_2 <<endl;
    outPerspective <<"\n.......................\n";

    outScaled <<"\n.......................\n";
    outScaled << "T2: Matched Features: " << s2.matchedFeatures<< " FP 3x3 " << s2.fp_1 << " FP 9x9 " << s2.fp_2<<endl;
    outScaled << "T5: Matched Features: " << s5.matchedFeatures<< " FP 3x3 " << s5.fp_1 << " FP 9x9 " << s5.fp_2<<endl;
    outScaled << "T10: Matched Features: " << s10.matchedFeatures<< " FP 3x3 " << s10.fp_1 << " FP 9x9 " << s10.fp_2 <<endl;
    outScaled <<"\n.......................\n";

    runSimALL(method);

}


int main(int argc, char *argv[])
{
    //QCoreApplication a(argc, argv);
    cv::initModule_nonfree();
    //fileAffine.open(QIODevice::ReadWrite);
//    filePerspective.open(QIODevice::ReadWrite);
//    fileRotated.open(QIODevice::ReadWrite);
//    fileScaled.open(QIODevice::ReadWrite);
    fileSimilarity.open((QFile::ReadWrite));

    char method[] = "SIFT";
//    runAll(method);
    char method1[] = "SURF";
  //  runAll(method1);

    //runSimALL(method);
    runSimALL(method1);

//    fileAffine.close();
//    filePerspective.close();
//    fileRotated.close();
//    fileScaled.close();
    fileSimilarity.close();

//    Mat m = imread(pathReference.arg(1).toStdString());
//    QString targetTransform;
//    // Reference images
//    FileStorage fs("CountKeypoints_ori.yml", FileStorage::WRITE);
//    QDir dir(pathReference1);
//   // saveKeypoints(dir,fs, method);

//    // count affine keypoints
//    FileStorage fs1("CountAffineKeypoints.yml", FileStorage::WRITE);
//    QDir dirAffine(pathAffine1);
// //   saveKeypoints(dirAffine,fs1, method);

//    // Match with affine
//    targetTransform = "affine";
//    FileStorage fsAffineMatching("AffineMatching.yml", FileStorage::WRITE);

//    dir.setFilter(QDir::Files);
//    QFileInfoList file = dir.entryInfoList();

//    for(int i = 1; i <= NO_IMAGES; i++)
//    {
//        //QFileInfo f = file.at(i);
//      //  cout << pathReference.arg(i).toStdString() <<endl;
//        //Mat m = imread(f.absoluteFilePath().toStdString());
//        //Mat m = imread(pathReference.arg(i).toStdString());

//       // initMatching(m,dirAffine,method,targetTransform,fsAffineMatching,i);
//    }
////    Mat image = imread(pathReference.arg(1).toStdString());
////    QString pathReference2("J:\\iit\\3rd sem\\eel806_vision\\project\\dataset\\reference1\\%1.jpg");
////    Mat image1 = imread(pathReference2.arg(1).toStdString());
////    Mat target = imread(pathScaled.arg(1).arg(5).toStdString());
////    cout << "SURF";
////             cout << "Threshold " << 2 << endl;
////            fpMatchingAccuracyScaled(image1, target, 5,method,2);
////             cout << "Threshold " << 5 << endl;
////            fpMatchingAccuracyScaled(image1, target, 5,method,5);
////            cout << endl;

//    for(int i = 1; i <= NO_SCALED; i++)
//    {

//        //Mat target = imread(pathAffine.arg(1).arg(i).toStdString());
//       // cout << "Threshold " << 2<< endl;
//        //fpMatchingAccuracyAffine(image, target,method,i,2);
//        //cout << "Threshold " << 5 << endl;
//        //fpMatchingAccuracyAffine(image, target, method,i,5);
//        //Mat target = imread(pathPerspective.arg(1).arg(i).toStdString());
//        //fpMatchingAccuracyPerspective(image, target,method,i,2);
//        //fpMatchingAccuracyPerspective(image, target, method,i,5);

////        Mat target = imread(pathRotated.arg(1).arg(i).toStdString());
////        cout << "Image No: " << i << endl;
////         cout << "Threshold " << 2 << endl;
////        fpMatchingAccuracyRotation(image, target, i,method,2);
////         cout << "Threshold " << 5 << endl;
////        fpMatchingAccuracyRotation(image, target, i,method,5);
////        cout << endl;

////        Mat target = imread(pathScaled.arg(1).arg(i).toStdString());
////        cout << "Image No: " << i << endl;
////         cout << "Threshold " << 2 << endl;
////        fpMatchingAccuracyScaled(image, target, i,method,2);
////         cout << "Threshold " << 5 << endl;
////        fpMatchingAccuracyScaled(image, target, i,method,5);
////        cout << endl;
//    }

    // For all images


//    Mat src = imread(pathReference.arg(1).toStdString());
//    std::vector<Point2f> obj(4);
//    std::vector<Point2f> scene(4);
//    int xSrc[3], ySrc[3]; float xDest[3], yDest[3];
//    xSrc[0] = 0; xSrc[1] = src.cols-1; xSrc[2] = 0;
//    ySrc[0] = 0; ySrc[1] = 0; ySrc[2] = src.rows-1;

//    xDest[0] = 0.0; xDest[1] = 0.9; xDest[2] = 0.0;
//    yDest[0] = 0.1; yDest[1] = 0.1; yDest[2] = 0.9;

//    Point2f srcTri[4];
//    Point2f dstTri[4];
//    srcTri[0] = Point2f( xSrc[0],ySrc[0] );
//    srcTri[1] = Point2f( xSrc[1], ySrc[1] );
//    srcTri[2] = Point2f( xSrc[2], ySrc[2] );
//    cout << "\nadfdf"<< src.cols*xDest[0] << endl;
//    dstTri[0] = Point2f( src.cols*xDest[0], src.rows*yDest[0] );
//    dstTri[1] = Point2f( src.cols*xDest[1], src.rows*yDest[1] );
//    dstTri[2] = Point2f( src.cols*xDest[2], src.rows*yDest[2] );

//    Mat warp_mat = getAffineTransform( srcTri, dstTri );

//    warp_mat.convertTo(warp_mat,CV_32FC1,1,0);
//    Point2f x;
//    x = Point2f(src.cols-1, src.rows-1);
//    std::vector<cv::Point3f> img1;
//    img1.push_back(Point3f(src.cols-1, src.rows-1,1));
//    Mat srcMat = Mat(img1).reshape(1).t();
//    Mat dstMat = warp_mat*srcMat; //USE MATRIX A
//    cout << dstMat;
//    //Mat warp_dst = Mat::zeros(1, 1,src.type() );
//    //warpAffine(img1, warp_dst, warp_mat, warp_dst.size() );
//    //warpAffine( img1, warp_dst, warp_mat, warp_dst.size() );
//    //imshow("aa", warp_dst);
//    cout << dstMat.at<float>(0,0)<<endl;
//    cout << dstMat.at<float>(0,1);
//    //warp_dst = Mat_<float>(warp_dst);
//    //cout  << " " << warp_dst;
//    obj[0] = srcTri[0];
//    obj[1] = srcTri[1];
//    obj[2] = srcTri[2];
//    obj[3] = Point2f(src.cols -1, src.rows-1);
//    scene[0] = dstTri[0];
//    scene[1] = dstTri[1];
//    scene[2] = dstTri[2];
//    scene[3] = Point2f(dstMat.at<float>(0,0), dstMat.at<float>(0,1));

//    std::vector<Point2f> scene_corners(4);

//    // Mat H = findHomography( obj, scene, CV_RANSAC );

//    Mat warp_mst = getPerspectiveTransform( obj, scene );
//    perspectiveTransform(obj,scene_corners,warp_mst);

//    cout << scene_corners;
//    Mat image = imread("J:\\iit\\3rd sem\\eel806_vision\\project\\dataset\\singletest\\1.jpg");
//    Mat target = imread("J:\\iit\\3rd sem\\eel806_vision\\project\\dataset\\singletest\\affine_1_1.jpg");
//    resize(image,image,Size(400,400));
//    resize(target,target,Size(400,400));
//    cv::Ptr<FeatureDetector> featureDetector = cv::FeatureDetector::create(method);
//    std::vector<KeyPoint> keypoints;
//    cv::Mat descriptors;
//    featureDetector->detect(image, keypoints); // NOTE: featureDetector is a pointer hence the '->'.

//    cv::Ptr<DescriptorExtractor> featureExtractor = cv::DescriptorExtractor::create(method);
//    featureExtractor->compute(image, keypoints, descriptors);

//    cv::Mat descriptors1;
//    std::vector<KeyPoint> keypoints1;
//    featureDetector->detect(target, keypoints1); // NOTE: featureDetector is a pointer hence the '->'.
//    featureExtractor->compute(target, keypoints1, descriptors1);

//    FlannBasedMatcher matcher;
//    std::vector< DMatch > matches;
//    matcher.match( descriptors, descriptors1, matches );

//    double max_dist = 0; double min_dist = 100;

//    for( int i = 0; i < descriptors.rows; i++ )
//    { double dist = matches[i].distance;
//      if( dist < min_dist ) min_dist = dist;
//      if( dist > max_dist ) max_dist = dist;
//    }

//    printf("-- Max dist : %f \n", max_dist );
//    printf("-- Min dist : %f \n", min_dist );

//    std::vector< DMatch > good_matches;

//    for( int i = 0; i < descriptors.rows; i++ )
//    { if( matches[i].distance <= 2*min_dist )
//      { good_matches.push_back( matches[i]); }
//    }

//    Mat img_matches;
//    drawMatches( image, keypoints, target, keypoints1,
//                 good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
//                 vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
//    //imshow( "Good Matches", img_matches );

//    std::vector<Point2f> obj;
//      std::vector<Point2f> scene;
//      cout << "Total Matches: " << good_matches.size();
//    for( int i = 0; i < good_matches.size(); i++ )
//      {

//        //-- Get the keypoints from the good matches
//        obj.push_back( keypoints[ good_matches[i].queryIdx ].pt );
//        scene.push_back( keypoints1[ good_matches[i].trainIdx ].pt );
//      }

//      Mat H = findHomography( obj, scene, CV_RANSAC );
//      std::vector<Point2f> obj_corners(4);
//        obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( image.cols, 0 );
//        obj_corners[2] = cvPoint( image.cols, image.rows ); obj_corners[3] = cvPoint( 0, image.rows );
//        std::vector<Point2f> scene_corners(4);

//        perspectiveTransform( obj_corners, scene_corners, H);

//        //-- Draw lines between the corners (the mapped object in the scene - image_2 )
//        line( img_matches, scene_corners[0] + Point2f( image.cols, 0), scene_corners[1] + Point2f( image.cols, 0), Scalar(0, 255, 0), 4 );
//        line( img_matches, scene_corners[1] + Point2f( image.cols, 0), scene_corners[2] + Point2f( image.cols, 0), Scalar( 0, 255, 0), 4 );
//        line( img_matches, scene_corners[2] + Point2f( image.cols, 0), scene_corners[3] + Point2f( image.cols, 0), Scalar( 0, 255, 0), 4 );
//        line( img_matches, scene_corners[3] + Point2f( image.cols, 0), scene_corners[0] + Point2f( image.cols, 0), Scalar( 0, 255, 0), 4 );

        //-- Show detected matches
        //imshow( "Good Matches & Object detection", img_matches );

cvWaitKey(0);
    return 0;
}
