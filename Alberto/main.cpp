///@name Alberto Makosa
#include "utilities.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {

	instance inst;

	read_input(&inst, argc, argv);

	upload_img(&inst);

	imshow("Edge Map", inst.grey_image[1]);
	
	waitKey(0);


	return 0;
}