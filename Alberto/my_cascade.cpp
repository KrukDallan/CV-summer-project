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

void cascade_algo(Mat input_test, Mat output) {
	
	Mat img = input_test;
	
	//load trained model
	CascadeClassifier my_cascade("cascade/cascade.xml");


	vector<Rect> hands;

	my_cascade.detectMultiScale(img, hands, 1.1, 2, 0, Size(120,120), Size(160,160));

	for (int i=0; i<hands.size();i++)
	{
		
		rectangle(output, hands[i], Scalar(0, 255,0));
		//hands.push_back(Rect(img.cols - r->x - r->width, r->y, r->width, r->height));
	}

	//imwrite("test.jpg", output);
	imshow("Matches", output);

	

	waitKey(0);

}
