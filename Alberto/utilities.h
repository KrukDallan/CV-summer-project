#ifndef UTILITIES_H
#define UTILITIES_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/gapi/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>
#include <iostream>

typedef struct
{
	char* image_path;
	char* set;
	
	std::vector<cv::Mat> image;
	std::vector<cv::Mat> grey_image;
	std::vector<cv::Mat> img_edges;
	std::vector<cv::Mat> output_SIFT;
	cv::Mat dictionary;

	int verbose;
	
}
instance;

void read_input(instance* inst, int argc, char** argv);
void upload_img(instance* inst);
//void binarization(instance* inst);
void img_canny(instance* inst);
void first_BoF_step(instance* inst);
void second_BoF_step(instance* inst);

#endif