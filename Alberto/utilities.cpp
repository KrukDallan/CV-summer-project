#include"utilities.h"

using namespace std;

using namespace cv;

void read_input(instance* inst, int argc, char** argv) {
	if (argc <= 1) {
		printf("Type \"%s -help\" to see available comands\n", argv[0]);
		exit(1);
	}

	
	inst->set = 0;
	int help = 0;

	for (int i = 0; i < argc; i++) {
		if (strcmp(argv[i], "-f") == 0) {
			const char* path = argv[++i];
			inst->image_path = (char*)calloc(strlen(path), sizeof(char));
			strncpy(inst->image_path, path, strlen(path));
			continue;
		}

		if (strcmp(argv[i], "-s") == 0) {
			const char* path = argv[++i];
			inst->set = (char*)calloc(strlen(path), sizeof(char));
			strncpy(inst->set, path, strlen(path));
			continue;
		}

		if (strcmp(argv[i], "-help") == 0) {
			help = 1;
			continue;
		}

		if (help) {
			printf("-f <image's path>         To pass the path of the images \n");
			printf("-s <set of images>        To pass which set of images use in the algorithm, if is 0 we use all\n");
			exit(0);
		}
	}
}

void upload_img(instance* inst) {
	string path = inst->image_path;
	path += "/";
	path += inst->set;
	path += "/*.jpg";
	//cout << path;

	string folder(path);
	vector<string> filenames;

	glob(folder, filenames);
	vector<Mat> img(filenames.size());
	vector<Mat> greyscale(filenames.size());

	for (int i = 0 ; i < filenames.size(); i++) {
		img[i] = imread(filenames[i]);
		cvtColor(img[i], greyscale[i], COLOR_BGR2GRAY, 1);
	}
	inst->image = img;
	inst->grey_image = greyscale;
}