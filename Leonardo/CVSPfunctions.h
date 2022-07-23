#pragma once

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


std::vector<cv::Rect> cascade(cv::Mat img);

std::vector<float> correctBB(cv::Mat img, std::vector<cv::Rect> hands, cv::Mat hand_img);


void bow(std::string path);

float bowTest(cv::Mat image);

void kNearest(std::string path);

cv::Mat computeTestHistogram(cv::Mat descriptor);

cv::Mat siftDescriptor(cv::Mat img);

void trash();
