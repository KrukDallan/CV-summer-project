#include "my_cascade.h"
#include <fstream>

using namespace std;
using namespace cv;

void generate_neg_file(std::string path)
{
	string folder(path);
	vector<string> filenames;

	glob(folder, filenames);

	ofstream negative;

	negative.open("neg.txt");


	for (int i = 0; i < filenames.size(); i++) {
		
		negative << filenames[i] << "\n";
	}

	negative.close();
}

void negative_dataset(string path) {

	ifstream file("folders.txt");
	string str, outpath("negative/");
	
	Mat img, tmp;
	while (getline(file, str)) {
		string prefix(path+str);
		string paths = prefix + SUFFIX;
		string folder(paths);
		vector<string> filenames;

		glob(paths, filenames);

		for (int i = 0; i < filenames.size(); i++) {
			Mat out;
			stringstream ss;
			ss << i;
			img = imread(filenames[i]);
			tmp = preprocessing(img);
			bitwise_not(tmp, tmp);
			bitwise_and(img, img, out, tmp);
			imshow("filter test", out);
			waitKey(0);
			imwrite(outpath + ss.str() + SUFFIX, out);


			cout << outpath+ss.str()+SUFFIX << '\n';
			outpath = "negative/";
		}
	}
	

}

void cascade_algo(Mat input_test, Mat output, instance* inst) {
	
	Mat img = input_test;
	GaussianBlur(img, img, Size(5, 5),0);
	
	//load trained model
	CascadeClassifier my_cascade("cascade/cascade.xml");


	vector<Rect> hands;

	my_cascade.detectMultiScale(img, hands);

	for (int i=0; i<hands.size();i++)
	{
		
		rectangle(output, hands[i], Scalar(0, 255,0));
		//hands.push_back(Rect(img.cols - r->x - r->width, r->y, r->width, r->height));
	}

	//imwrite("test.jpg", output);
	imshow("Matches", output);

	waitKey(0);

	inst->hands = hands;
	//my_grabcut(hands, img);
}
