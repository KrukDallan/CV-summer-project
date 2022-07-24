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
