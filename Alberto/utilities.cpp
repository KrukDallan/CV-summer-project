#include"utilities.h"

using namespace std;

using namespace cv;

void read_input(instance* inst, int argc, char** argv) {
	if (argc <= 1) {
		printf("Type \"%s -help\" to see available comands\n", argv[0]);
		exit(1);
	}

	inst->image_path = NULL;
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
	//path += "/";
	//path += inst->set;
	path += "/*.jpg";


	string folder(path);
	vector<string> filenames;

	glob(folder, filenames);
	vector<Mat> img(filenames.size());
	vector<Mat> greyscale(filenames.size());

	for (int i = 0; i < filenames.size(); i++) {
		img[i] = imread(filenames[i]);
		cvtColor(img[i], greyscale[i], COLOR_BGR2GRAY, 1);
	}
	inst->image = img;
	inst->grey_image = greyscale;
}

void img_canny(instance* inst) {
	vector<Mat> img(inst->grey_image.size());
	vector<Mat> detected_edges(inst->grey_image.size());
	img = inst->grey_image;
	for (int i = 0; i < img.size(); i++) {
		GaussianBlur(img[i], detected_edges[i], Size(3, 3), 0);
		Canny(detected_edges[i], detected_edges[i], 50, 250, 3);

	}
	inst->img_edges = detected_edges;
}

void first_BoF_step(instance* inst)
{
	vector<Mat> img(inst->grey_image.size());
	vector<Mat> output(inst->grey_image.size());
	Mat featureUnclustered;
	img = inst->grey_image;

	for (int i = 0; i < img.size() / 2; i = i + 2) {
		Ptr<SiftFeatureDetector> detector = SiftFeatureDetector::create();
		vector<KeyPoint> keypoints;
		detector->detect(img[i], keypoints);

		Ptr<SiftDescriptorExtractor> featureExtractor = SiftDescriptorExtractor::create();
		Mat descriptors;
		featureExtractor->compute(img[i], keypoints, descriptors);
		featureUnclustered.push_back(descriptors);
		Scalar keypointColor = Scalar(255, 0, 0);
		drawKeypoints(img[i], keypoints, output[i], keypointColor, DrawMatchesFlags::DEFAULT);
	}

	int dictionarySize = 200;

	TermCriteria tc(TermCriteria::MAX_ITER, 10, 0.001);

	int retries = 1;

	int flags = KMEANS_PP_CENTERS;

	BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);

	Mat dictionary = bowTrainer.cluster(featureUnclustered);

	inst->dictionary = dictionary;

	FileStorage fs("dictionary.yml", FileStorage::WRITE);
	fs << "vocabulary" << dictionary;
	fs.release();

	inst->output_SIFT = output;
}

void second_BoF_step(instance* inst) {
	Mat dictionary;
	Mat bowDescriptor;

	FileStorage fs("dictionary.yml", FileStorage::READ);
	fs["vocabulary"] >> dictionary;
	fs.release();

	Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);

	Ptr<FeatureDetector> detector(new SiftFeatureDetector());

	Ptr<DescriptorExtractor> extractor(new SiftDescriptorExtractor);

	BOWImgDescriptorExtractor bowDE(extractor, matcher);

	bowDE.setVocabulary(dictionary);

	char* filename = new char[100];

	char* imageTag = new char[10];

	FileStorage fs1("descriptor.yml", FileStorage::WRITE);

	sprintf(filename, "data/0/02.jpg");

	Mat img = imread(filename, IMREAD_GRAYSCALE);

	vector<KeyPoint> keypoints;

	detector->detect(img, keypoints);

	bowDE.compute(img, keypoints, bowDescriptor);

	sprintf(imageTag, "img1");

	fs1 << imageTag << bowDescriptor;

	fs1.release();
}

Mat preprocessing(Mat input) {
	Mat tmp;
	blur(input, tmp, Size(5, 5));

	cvtColor(tmp, tmp, COLOR_BGR2HSV);

	inRange(tmp, Scalar(0, 30, 60), Scalar(20, 150, 255), tmp);

	erode(tmp, tmp, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));

	dilate(tmp, tmp, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));

	return tmp;

}


void free_instance(instance* inst) {
	free(inst->image_path);
	free(inst->set);
}

/*
void binarization(instance* inst) {
	vector<Mat> img(inst->image.size());
	img = inst->image;
	for (int i = 0; i < img.size(); i++) {
		for (int j = 0; j < img[i].rows; j++) {
			for (int k = 0; k < img[i].cols; k++) {
				if (45 < img[i].at<Vec3b>(j, k)[0] && img[i].at<Vec3b>(j, k)[0] < 255 &&
					34 < img[i].at<Vec3b>(j, k)[1] && img[i].at<Vec3b>(j, k)[1] < 210 &&
					30  < img[i].at<Vec3b>(j, k)[2] && img[i].at<Vec3b>(j, k)[2] < 190
					) {

					img[i].at<Vec3b>(j, k)[0] = 255;
					img[i].at<Vec3b>(j, k)[1] = 255;
					img[i].at<Vec3b>(j, k)[2] = 255;



				}
				else {
				img[i].at<Vec3b>(j, k)[0] = 0;
				img[i].at<Vec3b>(j, k)[1] = 0;
				img[i].at<Vec3b>(j, k)[2] = 0;
				}

			}
		}
	}

	inst->bin_image = img;

}
*/