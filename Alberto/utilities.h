#ifndef UTILITIES_H
#define UTILITIES_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/calib3d.hpp>
#include <vector>
#include <iostream>

typedef struct
{
	char* image_path;
	char* set;
	
	std::vector<cv::Mat> image;
	std::vector<cv::Mat> grey_image;

	int verbose;
	
}
instance;

void read_input(instance* inst, int argc, char** argv);
void upload_img(instance* inst);


#endif