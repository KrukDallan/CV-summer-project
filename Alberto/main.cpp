///@name Alberto Makosa
#include "utilities.h"
#include "my_cascade.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {

	string path = "negative/*.jpg";
	string path2 = "pos2/";
	instance inst;

	read_input(&inst, argc, argv);

	upload_img(&inst);

	//img_canny(&inst);

	generate_neg_file(path);

	
	preprocessing(&inst);
	//first_BoF_step(&inst);
	//second_BoF_step(&inst);

	
	for (int i = 0; i < inst.image.size(); i++) {
		//cascade_algo(inst.filtered[i], inst.image[i]);
		for (int row = 0; row < inst.image[i].rows; row++) {
			for (int col = 0; col < inst.image[i].cols; col++) {
				if (inst.filtered[i].at<Vec3b>(row, col)[0] != 0 && inst.filtered[i].at<Vec3b>(row, col)[1] != 0 && inst.filtered[i].at<Vec3b>(row, col)[2] != 0) {
					inst.image[i].at<Vec3b>(row, col)[0] = inst.filtered[i].at<Vec3b>(row, col)[0];
					inst.image[i].at<Vec3b>(row, col)[1] = inst.filtered[i].at<Vec3b>(row, col)[1];
					inst.image[i].at<Vec3b>(row, col)[2] = inst.filtered[i].at<Vec3b>(row, col)[2];

				}
			}
		}

		
	}

	imshow("test", inst.image[1]);
	waitKey(0);

	/*
	path2 += i;
		path2 += ".jpg";
		imwrite(path2, inst.filtered[i]);
		path2 = "pos2/";
	*/
	
	free_instance(&inst);
	


	return 0;
}