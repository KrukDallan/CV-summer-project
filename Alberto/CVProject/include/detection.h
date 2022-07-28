#ifndef DETECTION_H
#define DETECTION_H
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



std::vector<cv::Rect> detectHands(cv::Mat src, cv::Mat hogft);

std::vector<cv::Rect> cascade(cv::Mat img);

std::vector<float> hog(cv::Mat img);

double comparehog(cv::Mat img, cv::Rect bb, cv::Mat hogft);

std::vector<float> templateMatching(cv::Mat img, std::vector<cv::Rect> hands, cv::Mat hand_img);

std::vector<float> IoUDetection(std::vector<cv::Rect> hands, std::string path);

cv::Mat removeBG(cv::Mat src);

cv::Mat sharpenImage(cv::Mat img);

#endif
