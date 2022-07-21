///@name Alberto Makosa
#include "utilities.h"
#include "my_cascade.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {

	string path = "negative/*.jpg";
	string path2 = "pos2/";
	instance inst;
	Mat test;

	read_input(&inst, argc, argv);

	upload_img(&inst);

	//img_canny(&inst);

	generate_neg_file(path);

	
	preprocessing(&inst);
	//first_BoF_step(&inst);
	//second_BoF_step(&inst);
	bool flag = false;
	
	for (int i = 0; i < inst.filtered.size(); i++) {
		cascade_algo(inst.image[i], inst.image[i]);


		if (flag) {
			//addWeighted(inst.image[0], 0.7, inst.filtered[0], 0.3, 0.0, test);
			stringstream ss;
			ss << i;
			path2 += ss.str();
			path2 += ".jpg";

			for (int row = 0; row < inst.filtered[i].rows; row++) {
				for (int col = 0; col < inst.filtered[i].cols; col++) {
					if (inst.filtered[i].at<uchar>(row, col) != 0) {
						inst.image[i].at<Vec3b>(row, col)[0] = inst.filtered[i].at<uchar>(row, col);
						inst.image[i].at<Vec3b>(row, col)[1] = inst.filtered[i].at<uchar>(row, col);
						inst.image[i].at<Vec3b>(row, col)[2] = inst.filtered[i].at<uchar>(row, col);

					}
				}
			}
			imwrite(path2, inst.image[i]);
			path2 = "pos2/";

		}
		
	}
	
	
	free_instance(&inst);
	


	return 0;
}