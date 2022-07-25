#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include "utilities.h"

typedef struct {
	int ch1;
	int ch2;
	int ch3;

}color;

cv::Mat my_grabcut(std::vector<cv::Rect> boudingBox, cv::Mat input);
std::vector<cv::Mat> mask_segm(std::vector<cv::Rect> boudingBox, cv::Mat input);
cv::Mat gen_output(cv::Mat input, std::vector<cv::Mat> hands);
void select_color(int i, color* color);
std::vector<cv::Mat> color_hands(std::vector<cv::Mat> hands);



#endif