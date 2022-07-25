///@name Alberto Makosa
#include "utilities.h"
#include "my_cascade.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {

	string path = "neg2/*.jpg";
	string path2 = "neg2/";
	string path3 = "test/";
	vector<Mat> hands;
	instance inst;
	Mat test;
	int neg = 4699;

	read_input(&inst, argc, argv);

	upload_img(&inst);

	//img_canny(&inst);

	generate_neg_file(path);

	//negative_dataset("_LABELLED_SAMPLES");

	//for (int i = 0; i < inst.image.size(); i++) {
	//	inst.filtered[i]=preprocessing(inst.image[i]);
	//}
	
	//first_BoF_step(&inst);
	//second_BoF_step(&inst);
	bool flag = false;
	cascade_algo(inst.image[9], inst.image[9], &inst);
	hands=mask_segm(inst.hands,inst.image[9]);
	hands = color_hands(hands);
	test=gen_output(inst.image[9], hands);

	imshow("tets", test);
	waitKey(0);

	for (int i = 0; i < inst.image.size(); i++) {
		
		//cascade_algo(inst.image[i], inst.image[i]);
		stringstream ss;
		//ss << i;
		//path3 = path3 + ss.str() + ".jpg";
		//imwrite(path3, inst.image[i]);
		//path3 = "test/";

		if (flag) {
			//addWeighted(inst.image[0], 0.7, inst.filtered[0], 0.3, 0.0, test);
			
			ss << neg;
			path2 += ss.str();
			path2 += ".jpg";
			//cout << path2 << '\n';

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
			neg++;
			path2 = "neg2/";

		}
		
	}
	
	
	free_instance(&inst);
	


	return 0;
}