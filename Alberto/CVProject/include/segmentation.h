#ifndef SEGMENTATION_H
#define SEGMENTATION_H
// @name Leonardo Sforzin
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>


#include <iostream>
#include <vector>
#include <string>
#include <math.h>
#include <fstream>

cv::Mat meanshift(cv::Mat src);

cv::Mat floodfill(cv::Mat src, cv::Rect rect, cv::Scalar colour);

float accuracy(cv::Mat mask, std::string path);

#endif