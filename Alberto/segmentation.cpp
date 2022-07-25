#include "segmentation.h"

using namespace cv;
using namespace std;

Mat my_grabcut(vector<Rect> boudingBox, Mat input)
{
	Mat mask, bgdModel, fgdModel, img=input, out=input, in=input;
	vector<Mat> hands(boudingBox.size());
	vector<Mat> segmented(boudingBox.size());
	int COLOR[4][3] = { {0, 0, 255}, {255, 0, 0}, {0, 255, 0}, {250, 215, 30} };

	for(int i=0; i<boudingBox.size(); i++){

		grabCut(input, mask, boudingBox[i], bgdModel, fgdModel, 1, GC_INIT_WITH_RECT);

		for (int row = 0; row < input.rows; row++) {
			for (int col = 0; col < input.cols; col++) {
				if (mask.at<uchar>(row, col) == 0 || mask.at<uchar>(row,col) == 2) {
					input.at<Vec3b>(row,col)[0] = 0;
					input.at<Vec3b>(row, col)[1] = 0;
					input.at<Vec3b>(row, col)[2] = 0;
				}
			}
		}
		cvtColor(input, hands[i], COLOR_BGR2GRAY);
	}
	for (int j = 0; j < hands.size(); j++) {
		for (int row = 0; row < img.rows; row++) {
			for (int col = 0; col < img.cols; col++) {
				if (hands[j].at<uchar>(row,col) != 0) {
					img.at<Vec3b>(row, col) = Vec3b(COLOR[j][0], COLOR[j][1], COLOR[j][2]);
					//img.at<Vec3b>(row, col)[0] = 0;
					//img.at<Vec3b>(row, col)[1] = 0;
					//img.at<Vec3b>(row, col)[2] = 255;
				}
			}
		}
		segmented[j] = img;
	}
	for (int t = 0; t < segmented.size(); t++) {
		
		bitwise_not(segmented[t], segmented[t]);
		//bitwise_and(in,in,out,segmented[t]);

	}
	return img;
}

vector<Mat> mask_segm(std::vector<cv::Rect> boudingBox, cv::Mat input)
{
	Mat in;
	//Mat img(input.rows, input.cols, CV_8UC1, Scalar(0, 0, 0));
	vector<Mat> hands(boudingBox.size());
	in = preprocessing(input);

	for (int i = 0; i < boudingBox.size(); i++) {
		Mat img(input.rows, input.cols, CV_8UC1, Scalar(0, 0, 0));
		for (int y = boudingBox[i].y; y < boudingBox[i].y + boudingBox[i].height; y++) {
			for (int x = boudingBox[i].x; x < boudingBox[i].x + boudingBox[i].width; x++) {
				img.at<uchar>(y, x) = in.at<uchar>(y, x);
			}
		}
		hands[i] = img;
	}
	return hands;
}

void select_color(int i, color* color) {
	switch (i)
	{
	case 0:
		color->ch1 = 0;
		color->ch2 = 0;
		color->ch3 = 255;
		break;
	case 1:
		color->ch1 = 0;
		color->ch2 = 255;
		color->ch3 = 0;
		break;
	case 2:
		color->ch1 = 255;
		color->ch2 = 0;
		color->ch3 = 0;
		break;
	case 3:
		color->ch1 = 250;
		color->ch2 = 215;
		color->ch3 = 30;
		break;
	default:
		color->ch1 = 255;
		color->ch2 = 255;
		color->ch3 = 255;
		break;
	}
}

vector<Mat> color_hands(vector<Mat> hands) {
	
	color clr;
	for (int i = 0; i < hands.size(); i++) {
		Mat img(hands[i].rows, hands[i].cols, CV_8UC3, Scalar(0, 0, 0));
		select_color(i, &clr);
		cout << "ch1 " << clr.ch1 << "ch2 " << clr.ch2 << "ch3" << clr.ch3 << '\n';
		for (int row = 0; row < hands[i].rows; row++) {
			for (int col = 0; col < hands[i].cols; col++) {
				if (hands[i].at<uchar>(row, col) != 0) {
					img.at<Vec3b>(row, col)[0] = clr.ch1;
					img.at<Vec3b>(row, col)[1] = clr.ch2;
					img.at<Vec3b>(row, col)[2] = clr.ch3;
				}
			}
		}
		hands[i] = img;
	}
	
	return hands;
}

Mat gen_output(Mat input, vector<Mat> hands)
{
	for (int i = 0; i < hands.size(); i++) {
		for (int row = 0; row < hands[i].rows; row++) {
			for (int col = 0; col < hands[i].cols; col++) {
				if (hands[i].at<Vec3b>(row, col) != Vec3b(0,0,0)) {
					input.at<Vec3b>(row, col)[0] = hands[i].at<Vec3b>(row, col)[0];
					input.at<Vec3b>(row, col)[1] = hands[i].at<Vec3b>(row, col)[1];
					input.at<Vec3b>(row, col)[2] = hands[i].at<Vec3b>(row, col)[2];
				}
			}
		}
	}
	return input;
}
