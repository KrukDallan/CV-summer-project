///@name Alberto Makosa
#include "utilities.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {


	instance inst;

	read_input(&inst, argc, argv);

	upload_img(&inst);

	img_canny(&inst);

	first_BoF_step(&inst);
	second_BoF_step(&inst);
	//imshow("Edge Map", inst.output_SIFT[2]);

	//waitKey(0);
	
	


	return 0;
}