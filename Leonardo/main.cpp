///@name Leonardo Sforzin

#include "CVSPfunctions.h"


//PROBLEMS:
//1) Sift is NOT rotation invariant --> should we add vocabularies of rotated images
//2) 


//Things to add?:


//Notes: histograms are used to train a classifier, like kNearestNeighbour
// After creating the codebook with k centroids, iterate through every image used and
// search for words present both in he dictionary and in the image, then increase the count of that particular word <-- this is how histograms are created
int main(int argc, char** argv)
{
	std::string path = argv[1];
	//kNearest(path);
	//bow(path);
	//blobDetector(path);
	cv::Mat hand_img = cv::imread("D:\\Desktop2\\MAGISTRALE\\Primo_anno-secondo_semestre\\ComputerVision\\0FinalProject\\Hands\\Hand_0006335.jpg");
	resize(hand_img, hand_img, cv::Size(), 1, 1, cv::INTER_LINEAR);
	//cv::rotate(hand_img, hand_img, cv::ROTATE_90_CLOCKWISE);
	cv::Mat test_img = cv::imread("D:\\Desktop2\\MAGISTRALE\\Primo_anno-secondo_semestre\\ComputerVision\\0FinalProject\\_LABELLED_SAMPLES\\CARDS_COURTYARD_B_T\\frame_0316.jpg");
	/*int result_cols = test_img.cols - hand_img.cols + 1;
	int result_rows = test_img.rows - hand_img.rows + 1;
	cv::Mat result;
	result.create(result_rows, result_cols, CV_32FC1);*/
	float rslt = bowTest(test_img);
	std::cout <<"\n" << "result: " << rslt << "\n";
	return 0;
	//cv::Mat tmp, res;

	//cv::blur(test_img, tmp, cv::Size(5, 5));

	//cv::cvtColor(tmp, tmp, cv::COLOR_BGR2HSV);

	//cv::inRange(tmp, cv::Scalar(0, 110, 0), cv::Scalar(55, 255, 169),  tmp);

	////cv::erode(tmp, tmp, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

	////cv::dilate(tmp, tmp, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));	

	//
	//cv::bitwise_and(test_img, test_img,res, tmp);
	//
	///*cv::imshow("original", test_img);
	//cv::waitKey(0); 
	//cv::imshow("prepro", tmp);
	//cv::waitKey(0);
	//cv::imshow("res", res);
	//cv::waitKey(0);*/
	///*cv::cvtColor(hand_img, hand_img, cv::COLOR_BGR2GRAY);
	//std::vector<std::vector<cv::Point>> contours;
	//cv::findContours(hand_img, contours, cv::RetrievalModes::RETR_LIST, cv::ContourApproximationModes::CHAIN_APPROX_TC89_KCOS);
	//cv::drawContours(hand_img, contours, -1, (0, 255, 0), 3);
	//cv::imshow("contours", hand_img);
	//cv::waitKey(0);*/
	//
	//cv::matchTemplate(test_img, hand_img, result, cv::TemplateMatchModes::TM_CCORR_NORMED); //
	//double minVal; double maxVal; cv::Point minLoc; cv::Point maxLoc;
	//cv::Point matchLoc;
	//cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
	//matchLoc = maxLoc;
	//cv::rectangle(test_img, matchLoc, cv::Point(matchLoc.x + hand_img.cols, matchLoc.y + hand_img.rows), cv::Scalar::all(255), -1, 8, 0);//
	////cv::rectangle(result, matchLoc, cv::Point(matchLoc.x + hand_img.cols, matchLoc.y + hand_img.rows), cv::Scalar::all(255), 2, 8, 0);
	///*cv::imshow("Test_image", res);
	//cv::waitKey();*/
	///*cv::Rect roi(matchLoc, cv::Point(matchLoc.x + hand_img.cols, matchLoc.y + hand_img.rows));
	//cv::Mat subimage = test_img(roi);
	//float btresult = bowTest(subimage);
	//std::cout << "\n" << "+++Bow result: " << btresult << "+++" << "\n";
	//return 0;*/

	////Second hand
	//for (int i = 0; i < 4; i++)
	//{
	//	cv::Mat result2;
	//	cv::Mat testImage = test_img.clone();//
	//	result2.create(result_rows, result_cols, CV_32FC1);
	//	//cv::rotate(hand_img, hand_img, cv::ROTATE_90_COUNTERCLOCKWISE);
	//	cv::matchTemplate(testImage, hand_img, result2, cv::TemplateMatchModes::TM_CCOEFF_NORMED);
	//	cv::minMaxLoc(result2, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
	//	matchLoc = maxLoc;
	//	cv::Rect roi(matchLoc,cv::Point(matchLoc.x + hand_img.cols, matchLoc.y + hand_img.rows));
	//	cv::Mat subimage = testImage(roi);
	//	float btresult = bowTest(subimage);
	//	std::cout << "\n" << "+++Bow result: " << btresult << "+++" << "\n";
	//	cv::imshow("subimage",subimage);//
	//	cv::waitKey();
	//	break;
	//	cv::rectangle(testImage, matchLoc, cv::Point(matchLoc.x + hand_img.cols, matchLoc.y + hand_img.rows), cv::Scalar::all(255), -1, 8, 0);
	//	btresult = bowTest(testImage);
	//	std::cout << "\n" << "+++Bow result: " << btresult << "+++" << "\n";
	//	break;
	//	if ( btresult - 1 < 0.01 )
	//	{
	//		//std::cout <<"\n" << "+++Bow result: " << btresult << "+++" << "\n";
	//		cv::rectangle(test_img, matchLoc, cv::Point(matchLoc.x + hand_img.cols, matchLoc.y + hand_img.rows), cv::Scalar::all(255),1, 8, 0);//
	//	}
	//	//cv::rectangle(result, matchLoc, cv::Point(matchLoc.x + hand_img.cols, matchLoc.y + hand_img.rows), cv::Scalar::all(255), 2, 8, 0);
	//	 
	//}
	//return 0;
	//cv::imshow("second hand", test_img);//
	//cv::waitKey();
	
	

	
	

	//cv::Mat img = cv::imread("D:\\Desktop2\\MAGISTRALE\\Primo_anno-secondo_semestre\\ComputerVision\\0FinalProject\\_LABELLED_SAMPLES\\CARDS_COURTYARD_B_T\\frame_0580.jpg");

	//load trained model
	//cv::CascadeClassifier my_cascade("cascade2k48.xml");


	//std::vector<cv::Rect> hands;
	//cv::Mat output = test_img; 
	//my_cascade.detectMultiScale(test_img, hands, 2, 2, 0, cv::Size(120, 120), cv::Size(280, 280));

	//for (int i = 0; i < hands.size(); i++)
	//{
	//	cv::rectangle(output, hands[i], cv::Scalar(0, 255, 0));
	//	//hands.push_back(Rect(img.cols - r->x - r->width, r->y, r->width, r->height));
	//}

	//cv::imshow("Matches", output);
	//cv::waitKey(0);
}

void blobDetector(std::string path)
{
	std::vector<cv::Mat> imgVec;
	int fNSize = 0;
	// load images
	std::cout << "\n" << "Loading images" << "\n";
	cv::Mat img = cv::imread("D:\\Desktop2\\MAGISTRALE\\Primo_anno-secondo_semestre\\ComputerVision\\0FinalProject\\Hands\\Hand_0006335.jpg");
	imgVec.push_back(img); // "model"

	img = cv::imread("D:\\Desktop2\\MAGISTRALE\\Primo_anno-secondo_semestre\\ComputerVision\\0FinalProject\\_LABELLED_SAMPLES\\CARDS_COURTYARD_B_T\\frame_0011.jpg");
	imgVec.push_back(img);

	cv::Mat modelDescriptor = siftDescriptor(imgVec[0]);
	cv::SimpleBlobDetector::Params params;
	//Change thresholds
	params.minThreshold = 10;
	params.maxThreshold = 200;

	// Filter by Area.
	params.filterByArea = true;
	params.minArea = 1500;

	// Filter by Circularity
	params.filterByCircularity = true;
	params.minCircularity = 0.1;

	// Filter by Convexity
	params.filterByConvexity = true;
	params.minConvexity = 0.87;

	// Filter by Inertia
	params.filterByInertia = true;
	params.minInertiaRatio = 0.01;

	cv::Ptr<cv::SimpleBlobDetector> sbd = cv::SimpleBlobDetector::create();
	cv::Mat idescriptor, mdescriptor;
	std::vector<cv::KeyPoint> iKeyPoints, mKeyPoints;
	sbd->detect(imgVec[1], iKeyPoints);
	sbd->compute(imgVec[1], iKeyPoints, idescriptor);

	sbd->detect(imgVec[0], mKeyPoints);
	sbd->compute(imgVec[0], mKeyPoints, mdescriptor);

	cv::BFMatcher matcher;
	std::vector<cv::DMatch> matches;
	matcher.match(idescriptor, mdescriptor, matches);

	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < idescriptor.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);

	//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	std::vector< cv::DMatch > good_matches;

	for (int i = 0; i < idescriptor.rows; i++)
	{
		if (matches[i].distance < 3 * min_dist)
		{
			good_matches.push_back(matches[i]);
		}
	}

	cv::Mat img_matches;
	cv::drawMatches(imgVec[1], iKeyPoints, imgVec[0], mKeyPoints,
		good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
		std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	cv::imshow("Good matches", img_matches);
	cv::waitKey(0);

}

cv::Mat blobDescriptor(cv::Mat img)
{
	cv::SimpleBlobDetector::Params params;
	params.filterByColor = true;

	params.minArea = 80;

	params.filterByConvexity = true;
	params.minConvexity = 0.5;

	cv::Ptr<cv::SimpleBlobDetector> sbd = cv::SimpleBlobDetector::create(params);
	cv::Mat descriptor;
	std::vector<cv::KeyPoint> keyPoints;
	sbd->detect(img, keyPoints);
	sbd->compute(img, keyPoints, descriptor);
	cv::Mat output;
	
	return descriptor;
}

cv::Mat siftDescriptor(cv::Mat img)
{
	std::vector<cv::KeyPoint> keyPoints;
	cv::Mat descriptor, features;
	cv::Ptr<cv::SiftDescriptorExtractor> extractor = cv::SiftDescriptorExtractor::create();
	//detects features
	extractor->detect(img, keyPoints);
	//Computes the descriptors for a set of keypoints detected in an image
	extractor->compute(img, keyPoints, descriptor);

	return descriptor;
}


void bow(std::string path)
{
	int executionCode = 1;

	// Training of the BoW
	if (executionCode == 1)
	{
		
		std::vector<std::string> datasetType{ "hands",  "chess", "jenga", "wb" }; //"cards",

		std::vector<cv::Mat> greyScale;
		int fNSize = 0;
		// load images
		std::cout << "\n" << "Loading images" << "\n";
		for (int i = 0; i < datasetType.size(); i++)
		{
			std::string fullPath = path + "\\" + datasetType[i] + "Dataset" + "/*.jpg";
			cv::String folderString(fullPath);
			std::vector<cv::String> fileNames;
			cv::glob(folderString, fileNames, false);
			fNSize += fileNames.size();
			for (int i = 0; i < fileNames.size(); i++)
			{
				cv::Mat img = cv::imread(fileNames[i], cv::IMREAD_GRAYSCALE);
				greyScale.push_back(img);
			}
			std::cout << "\n" << "Loaded dataset " << datasetType[i] + "Dataset" << "\n";
		}
		//Features extraction
		std::vector<cv::KeyPoint> keyPoints;
		cv::Mat descriptor, features;
		cv::Ptr<cv::SiftDescriptorExtractor> extractor = cv::SiftDescriptorExtractor::create();
		// variables used to display the progress of the for loop
		int increment1 = 0.01 * fNSize;
		int countdown = increment1;
		int percent1 = 0;
		std::cout << "\n";
		//Compute the features for each image of the "training set"
		std::cout << "\n" << "Starting BoW" << "\n";
		for (int i = 0; i < fNSize; i++)
		{
			//detects features
			extractor->detect(greyScale[i], keyPoints);
			//Computes the descriptors for a set of keypoints detected in an image
			extractor->compute(greyScale[i], keyPoints, descriptor);
			//insert them in features
			features.push_back(descriptor);
			if (--countdown == 0)
			{
				percent1++;
				std::cout << "\r" << std::string(percent1, '|') << percent1 * 1 << "%";
				countdown = increment1;
				std::cout.flush();
			}
		}

		std::cout << "\n" << "\n" << "Features computed" << "\n";

		//Build BoW trainer
		// dictSize (dictionary size) is the "k" of kmeans
		int dictSize = 4;
		//Termination criteria
		cv::TermCriteria tc(cv::TermCriteria::Type::COUNT, 30, 0.001);
		//attempts
		int attempts = 1;
		//flags
		int flags = cv::KMEANS_PP_CENTERS;
		//create BoW trainer
		cv::BOWKMeansTrainer bowTrainer(dictSize, tc, attempts, flags);
		//cluster the feature vectors
		std::cout << "\n" << "Starting clustering" << "\n";
		cv::Mat bow = bowTrainer.cluster(features);
		//store the vocabulary
		std::cout << "\n" << "Storing the bag of visual words" << "\n";
		//cv::FileStorage fs("cards-courtyard-bt.yml", cv::FileStorage::WRITE);
		//cv::FileStorage fs("cards-livingroom-hs.yml", cv::FileStorage::WRITE);
		cv::FileStorage fs("bag-of-words.yml", cv::FileStorage::WRITE);
		fs << "vocabulary" << bow;
		fs.release();
	}
	
	//Test section of the BoW algo
	if (executionCode == 3)
	{
		//bowTest();
	}
}

void kNearest(std::string path)
{
	cv::Mat bow;
	cv::FileStorage fs("bag-of-words.yml", cv::FileStorage::READ);
	fs["vocabulary"] >> bow;
	fs.release();

	std::cout << "\n" << "Creating histograms" << "\n";

	//Preprocessing for the kNN
	//Create the histograms
	cv::Mat trainingInput;
	cv::Mat trainingLabels;

	std::vector<cv::Mat> greyScale;
	int fNSize = 0;

	std::vector<std::string> datasetType{ "bighands",  "bigchess", "bigjenga", "wb" };//"cards"
	std::cout << "\n" << "Loading images" << "\n";
	for (int i = 0; i < datasetType.size(); i++)
	{
		std::string fullPath = path + "\\" + datasetType[i] + "Dataset" + "/*.jpg";
		cv::String folderString(fullPath);
		std::vector<cv::String> fileNames;
		cv::glob(folderString, fileNames, false);
		fNSize += fileNames.size();
		for (int i = 0; i < fileNames.size(); i++)
		{
			cv::Mat img = cv::imread(fileNames[i], cv::IMREAD_GRAYSCALE);
			greyScale.push_back(img);
		}
		std::cout << "\n" << "Loaded dataset " << datasetType[i] + "Dataset" << "\n";
	}
	int label = 1;
	for (int i = 0; i < datasetType.size(); i++)
	{
		std::cout << "\n" << "Current dataset: " << datasetType[i] << "Dataset" << "\n";

		std::string fullPath = path + "\\" + datasetType[i] + "Dataset" + "/*.jpg";
		cv::String folderString(fullPath);
		std::vector<cv::String> fileNames;
		cv::glob(folderString, fileNames, false);
		int increment5 = 0.05 * fileNames.size();
		int countdown = increment5;
		int percent5 = 0;

		for (int j = 0; j < fileNames.size(); j++)
		{
			cv::Mat img = cv::imread(fileNames[j], cv::IMREAD_GRAYSCALE);
			cv::Ptr<cv::SiftDescriptorExtractor> extractor = cv::SiftDescriptorExtractor::create();
			std::vector<cv::KeyPoint> keyPoints;
			cv::Mat descriptor;
			//detects features
			extractor->detect(greyScale[i], keyPoints);
			//Computes the descriptors for a set of keypoints detected in an image
			extractor->compute(greyScale[i], keyPoints, descriptor);

			cv::Ptr<cv::DescriptorMatcher> matcher = cv::FlannBasedMatcher::create();
			std::vector<cv::DMatch> matches;
			matcher->match(descriptor, bow, matches);
			cv::Mat histogram = cv::Mat::zeros(1, bow.rows, CV_32F);
			int index = 0;
			for (std::vector<cv::DMatch>::iterator k = matches.begin(); k < matches.end(); k++, index++)
			{
				histogram.at<float>(0, matches.at(index).trainIdx) += 1.0;
			}
			trainingInput.push_back(histogram);
			trainingLabels.push_back(cv::Mat(1, 1, CV_32SC1, label)); //CV_32SC1 is a 1 Channel of signed 32 bit integer

			if (--countdown == 0)
			{
				percent5++;
				std::cout << "\r" << std::string(percent5, '|') << percent5 * 5 << "%";
				countdown = increment5;
				std::cout.flush();
			}
		}
		label += 1;
	}
	/*cv::FileStorage fs1("labels.yml", cv::FileStorage::WRITE);
	fs1 << "klabels" << trainingLabels;
	fs1.release();*/
	//Create and train kNearest
	std::cout << "\n" << "Creating and training kNearest" << "\n";
	cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();
	knn->setDefaultK(4);


	//cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	/*svm->setKernel(cv::ml::SVM::LINEAR);
	svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 1e3, 1e-6));*/
	cv::Ptr<cv::ml::TrainData> trainData = cv::ml::TrainData::create(trainingInput, cv::ml::ROW_SAMPLE, trainingLabels);
	//svm->train(trainData);
	knn->train(trainData);
	std::cout << "\n" << "kNearest trained" << "\n";
	std::cout << "\n" << "Saving kNearest" << "\n";
	knn->save("220722-kNN.xml");
	std::cout << "\n" << "kNearest saved" << "\n";
}

float bowTest(cv::Mat image)
{
	// Note: class 1 == hands

	cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::load("220722-kNN.xml");
	
	//cv::Mat testImg = imread("D:\\Desktop2\\MAGISTRALE\\Primo_anno-secondo_semestre\\ComputerVision\\0FinalProject\\Hands\\Hand_0000002.jpg", cv::IMREAD_GRAYSCALE);

	//Compute testImage histogram
	cv::Ptr<cv::SiftDescriptorExtractor> extractor = cv::SiftDescriptorExtractor::create();
	std::vector<cv::KeyPoint> keyPoints;
	cv::Mat descriptor;
	//detects features
	extractor->detect(image, keyPoints);
	//Computes the descriptors for a set of keypoints detected in an image
	extractor->compute(image, keyPoints, descriptor);
	cv::Mat imgHist = computeTestHistogram(descriptor);
	cv::Mat result;
	float resultfloat= knn->predict(imgHist,result, cv::ml::StatModel::Flags::PREPROCESSED_INPUT);
	/*std::cout << resultfloat << "\n";
	cv::FileStorage fs("knnresult.yml", cv::FileStorage::WRITE);
	fs << "knnvector" << result;
	fs.release();*/
	return resultfloat;
}

cv::Mat computeTestHistogram(cv::Mat descriptor)
{
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::FlannBasedMatcher::create();

	cv::FileStorage fs("bag-of-words.yml", cv::FileStorage::READ);
	cv::Mat bow;
	fs["vocabulary"] >> bow;
	fs.release();

	std::vector<cv::DMatch> matches;
	matcher->match(descriptor, bow, matches);
	cv::Mat histogram = cv::Mat::zeros(1, bow.rows, CV_32F);
	int index = 0;
	for (std::vector<cv::DMatch>::iterator k = matches.begin(); k < matches.end(); k++, index++)
	{
		histogram.at<float>(0, matches.at(index).trainIdx) += 1.0;
	}
	return histogram;
}

