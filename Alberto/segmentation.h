#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include "utilities.h"

cv::Mat my_grabcut(std::vector<cv::Rect> boudingBox, cv::Mat input);



#endif