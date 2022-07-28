
#include "segmentation.h"

// @Leonardo Sforzin
cv::Mat meanshift(cv::Mat src)
{
    cv::Mat dst;

    cv::pyrMeanShiftFiltering(src, dst, 5, 5, 1);
    return dst;
}
// @Leonardo Sforzin
cv::Mat floodfill(cv::Mat src, cv::Rect rect, cv::Scalar colour)
{
    cv::Mat empty;

    cv::Mat dst = meanshift(src(rect));
    cv::Point seed(floor(dst.rows / 2), floor(dst.cols / 2));
    cv::Scalar lodiff(5, 5, 5);
    cv::Scalar updiff(20, 20, 20);

    cv::floodFill(dst, empty, seed, colour, 0, lodiff, updiff);

    return dst;
}

// @Alberto Makosa
float accuracy(cv::Mat mask, std::string path)
{
    cv::Mat gt = cv::imread(path);
    int true_pos = 0, true_neg=0, false_pos=0, false_neg=0;

    for (int row = 0; row < gt.rows; row++) 
    {
        for (int col = 0; col < gt.cols; col++) 
        {
            if (gt.at<uchar>(row, col) != 0 && mask.at<uchar>(row, col) != 0) 
            {
                true_pos++;
            }
            if (gt.at<uchar>(row, col) == 0 && mask.at<uchar>(row, col) == 0) 
            {
                true_neg++;
            }
            if (gt.at<uchar>(row, col) == 0 && mask.at<uchar>(row, col) != 0)
            {
                false_pos++;
            }
            if (gt.at<uchar>(row, col) != 0 && mask.at<uchar>(row, col) == 0)
            {
                false_neg++;
            }
        }
    }


    return (true_pos + true_neg) / (float)(true_pos + true_neg + false_pos + false_neg);
}