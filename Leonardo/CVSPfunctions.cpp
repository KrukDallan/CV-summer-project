#include "CVSPfunctions.h"

cv::Mat meanshift(cv::Mat img)
{
    // Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    //cv::TermCriteria term_crit(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 1);

    cv::Mat dst;
    //cv::Mat src = img(rect);
    
    cv::pyrMeanShiftFiltering(img, dst,5, 5, 1);
    return dst;
    cv::Mat empty;
   

    /*dst.copyTo(img(rect));

    cv::imshow("result", img);
    cv::waitKey(0);*/
}

cv::Mat floodfill(cv::Mat img, cv::Rect rect)
{
    cv::Mat empty;
    cv::Mat dst = img(rect);
    cv::Point seed(floor(dst.rows / 2), floor(dst.cols / 2));
    cv::Scalar newval(250, 50, 190);
    cv::Scalar lodiff(5, 5, 5);
    cv::Scalar updiff(25,25, 25);
    cv::floodFill(dst, empty, seed, newval,0,lodiff, updiff );
   /* cv::imshow("flooded", dst);
    cv::waitKey(0);*/
    return dst;
}

cv::Mat removeBG(cv::Mat img)
{
    // Params
    // Good params: th1:10, th2:50
    double th1 = 10;
    double th2 = 50;

    // detect edges
    cv::Mat edges;
    cv::Canny(img, edges, th1, th2);
    cv::dilate(edges, edges, cv::Mat());
    cv::erode(edges, edges, cv::Mat());
    /*cv::imshow("edges", edges);
    cv::waitKey(0);*/

    // find contours
    std::vector<std::vector<cv::Point>> contours;
    //cv::Mat contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(edges, contours, hierarchy, cv::RetrievalModes::RETR_LIST, cv::ContourApproximationModes::CHAIN_APPROX_NONE);
    int contoursIdx = 0;
    double maxArea = 0.0;
    for (int i = 0; i < contours.size(); i++)
    {
        double tmp = cv::contourArea(contours[i]);
        if (tmp > maxArea)
        {
            contoursIdx = i;
            maxArea = tmp;
        }
    }
    cv::Mat mask = cv::Mat::zeros(edges.rows, edges.cols, CV_8UC3);
    //cv::fillConvexPoly(mask, contours[contoursIdx], (255, 255, 255));
    cv::drawContours(mask, contours, contoursIdx, cv::Scalar(255, 255, 255), -1);
    /*cv::imshow("mask", mask);
    cv::waitKey(0);*/
    return mask;
}