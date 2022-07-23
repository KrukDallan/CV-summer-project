#pragma once

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>


#include <iostream>
#include <vector>
#include <string>

float bowTest(cv::Mat image);
void kNearest(std::string path);
cv::Mat computeTestHistogram(cv::Mat descriptor);
void bow(std::string path);

void blobDetector(std::string path);

cv::Mat siftDescriptor(cv::Mat img);
cv::Mat blobDescriptor(cv::Mat img);
