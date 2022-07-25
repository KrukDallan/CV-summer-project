#ifndef MY_CASCADE_H
#define MY_CASCADE_H

#include "utilities.h"
#include "segmentation.h"

#define SUFFIX "*.jpg"

void generate_neg_file(std::string path);

void cascade_algo(cv::Mat input_test, cv::Mat output, instance* inst);

void negative_dataset(std:: string path);


#endif