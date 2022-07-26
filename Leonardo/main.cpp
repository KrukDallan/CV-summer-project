///@name Leonardo Sforzin

#include "CVSPfunctions.h"


// HISTOGRAM TO DISCARD NON-HAND OBJECTS?

int main(int argc, char** argv)
{
	std::string path = argv[1];
	cv::Rect r;

	// Image of the hand to be used for Template Matching
	cv::Mat hand_img = cv::imread("D:\\Desktop2\\MAGISTRALE\\Primo_anno-secondo_semestre\\ComputerVision\\0FinalProject\\test_hand.png");
	//cv::Mat hand_img = meanshift(hhand_img, r);
	//resize(hand_img, hand_img, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
	//cv::cvtColor(hand_img, hand_img, cv::COLOR_BGR2GRAY);
	std::vector<cv::Mat> handVector;
	handVector.push_back(hand_img);
	//Obtain rotated versions of the hand
	for (int i = 0; i < 3; i++)
	{
		cv::rotate(hand_img, hand_img, cv::ROTATE_90_CLOCKWISE);
		handVector.push_back(hand_img);
	}
	
	// Image to be tested
	cv::Mat iimg = cv::imread("D:\\Desktop2\\MAGISTRALE\\Primo_anno-secondo_semestre\\ComputerVision\\0FinalProject\\02.jpg");
	std::string fn = "01.jpg";
	//cv::Mat img = meanshift(iimg);
	cv::Mat mask = removeBG(iimg);
	cv::Mat img;
	iimg.copyTo(img, mask);
	//cv::inRange(img, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255), tmp);
	cv::imshow("Mask result", img);
	cv::waitKey(0);
	
	//cv::GaussianBlur(img, img, cv::Size(5, 5),0,0);
	//histogram(img);
	//return 0;
	
	// Vector to store every rect (bb) found by the cascade of classifiers 
	// Note: intersecting rects will be dealt with in the following lines
	//std::vector<cv::Rect> hands = cascade(img); //temp_hands
	//cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
	std::vector<cv::Rect> hands = cascade(img); //temp_hands

	/*for (int i = 0; i < hands.size(); i++)
	{
		cv::rectangle(img, hands[i], cv::Scalar(0, 255, 0));
	}
	cv::imshow("All bb", img);
	cv::waitKey(0);
	return 0; */

	//cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
	//	1, 1, 1,
	//	1, -6, 1,
	//	1, 1, 1); //-6 -> very bright
	//cv::Mat imgLaplacian;
	//cv::filter2D(img, imgLaplacian, CV_32F, kernel);
	//cv::Mat sharp;
	//img.convertTo(sharp, CV_32F);
	//cv::Mat imgResult = sharp - imgLaplacian;
	//// convert back to 8bits gray scale
	//imgResult.convertTo(imgResult, CV_8UC3); //why CV_8UC3?
	//imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
	/*cv::imshow("Laplace Filtered Image", imgLaplacian);
	cv::waitKey(0);
	cv::imshow("New Sharped Image", imgResult);
	cv::waitKey(0);
	return 0;
	cv::imshow("All bb", img);
	cv::waitKey(0);
	return 0;*/
	//img = imgLaplacian;
	
	/*std::vector<int> areas;
	for (int i = 0; i < temp_hands.size(); i++)
	{
		areas.push_back(temp_hands[i].area());
	}*/

	//std::sort(areas.begin(), areas.end(), std::greater<int>());
	//for (int i = 0; i < temp_hands.size(); i++)
	//{
	//	/*std::vector<float>::iterator iter = std::find(areas.begin(), areas.end(), temp_hands[i]);
	//	int feindex = std::distance(areas.begin(), iter);*/
	//	for (int j = i; j < areas.size(); j++)
	//	{
	//		int tmp = temp_hands[j].area();
	//		if (tmp == areas[i] && i != j)
	//		{
	//			std::iter_swap(temp_hands.begin() + i, temp_hands.begin() + j);
	//			continue;
	//		}
	//	}
	//}
	// Now temp_hands is sorted by dimension of rect (bigger first)
	
	// Vector to store only the correct rects, which will be applied to the test image
	std::vector<cv::Rect> rectVector;
	 
	// Vector that will contain no intersecting rectangles
	//std::vector<cv::Rect> hands;

	//bool no_intersection = true;
	// +++++ WORK IN PROGRESS +++++
	//for (int i = 0; i < temp_hands.size() - 1; i++)
	//{
	//	std::vector<bool> intersectionvector;
	//	for (int j = i + 1; j < temp_hands.size(); j++)
	//	{
	//		cv::Rect intersection = temp_hands[i] & temp_hands[j];
	//		
	//		double area = intersection.width * intersection.height;
	//		double areaRecti = temp_hands[i].width * temp_hands[i].height;
	//		double areaRectj = temp_hands[j].width * temp_hands[j].height;
	//		double minAreaBetween = cv::min(areaRecti, areaRectj);
	//		if (area > 0)
	//		{
	//			no_intersection = false;
	//			/*cv::Rect tmp = temp_hands[i] | temp_hands[j];
	//			hands.push_back(tmp);
	//			no_intersection = false;
	//			break;*/
	//			if (area >60 * minAreaBetween / 100)
	//			{
	//				cv::Rect tmp = temp_hands[i] | temp_hands[j];
	//				hands.push_back(tmp);
	//				no_intersection = false;
	//				break;
	//			}
	//			else if ((area < 60 * minAreaBetween / 100) && (minAreaBetween< (cv::max(areaRecti, areaRectj))*0.5))
	//			{
	//				cv::Rect tmp = temp_hands[i] | temp_hands[j];
	//				hands.push_back(tmp);
	//				no_intersection = false;
	//				break;
	//			}
	//			else if (area < 15*minAreaBetween/100)
	//			{
	//				//temp_hands[j] = temp_hands[i];
	//				hands.push_back(temp_hands[i]);
	//				hands.push_back(temp_hands[j]);
	//				i++;
	//				break;
	//			}
	//		}
	//		
	//	} 
	//	if (no_intersection == true)
	//	{
	//		hands.push_back(temp_hands[i]);
	//	}
	//	no_intersection = true;
	//}

	// If you want to check how good the remaining bb are
	/*for (int i = 0; i < hands.size(); i++)
	{
		cv::rectangle(img, hands[i], cv::Scalar(0, 255, 0));
	}
	cv::imshow("Union bb", img);
	cv::waitKey(0);
	return 0;*/

	// Vector that will contain the values returned by the "correctBB" method
	std::vector<float> outputVector;
	// Index of the best bb
	int bbIndex = 0;
	// "Accuracy" of the best bounding box
	double maxvalue = 0.0;
	for (int i = 0; i < handVector.size(); i++)
	{
		outputVector = correctBB(img, hands, handVector[i]);
		if (outputVector[1] > maxvalue)
		{
			bbIndex = (int)outputVector[0];
			maxvalue = outputVector[1];
		}
	}
	//std::cout << "\n" << "Final Maxval = " << maxvalue << "\n";
	// Check if the accuracy is too low
	if (maxvalue < 0.1)
	{
		return 0;
	}
	maxvalue = 0.0;
	// Image that will be modified in order to hide the part of the image where the best bb is
	//This is done because in this way the cascade will not find the same bb again
	cv::Mat modimg = img.clone();

	rectVector.push_back(hands[bbIndex]);
	
	cv::rectangle(modimg, hands[bbIndex], cv::Scalar(0, 0, 0), -1);

	std::vector<cv::Rect>hand2 = cascade(modimg);

	// Used for imshow
	int index = 2;

	//Number of hands
	int counter = 1;
	
	while(hand2.size() > 0 || counter != 4)
	{
		// Used for imshow
		std::string nhands = "mani:";
		maxvalue = 0.0;
		for (int i = 0; i < handVector.size(); i++)
		{
			outputVector = correctBB(modimg, hand2, handVector[i]);
			if (outputVector[1] > maxvalue)
			{
				bbIndex = (int)outputVector[0];
				maxvalue = outputVector[1];
			}
		}
		//std::cout << "\n" << "Final Maxval = " << maxvalue << ", index = "<< bbIndex <<  "\n";
		
		if (maxvalue<0.1)
		{
			break;
		}
		else
		{
			/*for (int i = 0; i < rectVector.size(); i++)
			{
				cv::Rect intersection = rectVector[i] & hand2[bbIndex];
				double area = intersection.width * intersection.height;
				if (area > 0)
				{
					hand2[bbIndex] = rectVector[i] | hand2[bbIndex];
					rectVector.push_back(hand2[bbIndex]);
					break;
				}
			}*/
			//cv::rectangle(img, hand2[bbIndex], cv::Scalar(0, 255, 0));
			cv::rectangle(modimg, hand2[bbIndex], cv::Scalar(255, 255, 255), -1);
			rectVector.push_back(hand2[bbIndex]);
			nhands += std::to_string(index);
			index++;
			/*cv::imshow(nhands, img);
			cv::waitKey(0);*/
		}
		hand2 = cascade(modimg);
		counter++;
		if (counter == 4)
		{
			break;
		}
	}
	cv::Mat floodedRect = floodfill(img, rectVector[1]);
	floodedRect.copyTo(img(rectVector[1]));
	cv::imshow("Flooded", img);
	cv::waitKey(0);
	for (int i = 0; i < rectVector.size(); i++)
	{
		cv::rectangle(img, rectVector[i], cv::Scalar(0, 255, 0));
	}
	//cv::imwrite(fn, img);
	cv::imshow("Final result", img);
	cv::waitKey(0);
}

std::vector<float> correctBB(cv::Mat img, std::vector<cv::Rect> hands, cv::Mat hand_img)
{
	if (hands.size() == 0)
	{
		std::vector<float> output;
		output.push_back(0.0);
		output.push_back(0.0);
		return output;
	}
	//std::cout << "\n" << hands.size() << "\n"; 
	cv::Mat result;
	std::vector<float> output;
	double maxMaxVal = 0.0;
	int bbIndex = 0;
	for (int i = 0; i < hands.size(); i++)
	{
		//cv::rectangle(img, hands[i], cv::Scalar(0, 255, 0));
		cv::Mat subimg = img(hands[i]);
		cv::matchTemplate(subimg, hand_img, result, cv::TemplateMatchModes::TM_CCOEFF_NORMED); //
		double minVal; double maxVal; cv::Point minLoc; cv::Point maxLoc;
		cv::Point matchLoc;
		cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
		if (maxVal > maxMaxVal)
		{
			maxMaxVal = maxVal;
			bbIndex = i +0.0;
		}
	}
	//std::cout << "\n" << "Maxval = " << maxMaxVal << "\n";
	if (maxMaxVal < 0.40)
	{
		std::vector<float> output;
		output.push_back(0.0);
		output.push_back(0.0);
		return output;
	}
	output.push_back(bbIndex);
	output.push_back(maxMaxVal);
	return output;
}

std::vector<cv::Rect> cascade(cv::Mat img)
{
	//load trained model
	cv::CascadeClassifier my_cascade("best_cascade.xml");

	std::vector<cv::Rect> hands;
	cv::Mat output = img; 
	my_cascade.detectMultiScale(img, hands, 1.2, 3, 0, cv::Size(100, 100), cv::Size(1500, 1500));//working pretty good
	//my_cascade.detectMultiScale(img, hands, 1.2, 4, 0, cv::Size(100, 100), cv::Size(300, 300)); //tested
	//my_cascade.detectMultiScale(img, hands, 1.2, 3, 0, cv::Size(100, 100), cv::Size(1500, 1500));//tested
	//my_cascade.detectMultiScale(img, hands, 1.3, 4, 0, cv::Size(100, 100), cv::Size(1500, 1500));
	
	//for (int i = 0; i < hands.size(); i++)
	//{
	//	cv::rectangle(output, hands[i], cv::Scalar(0, 255, 0));
	//////	//hands.push_back(Rect(img.cols - r->x - r->width, r->y, r->width, r->height));
	//}
	//cv::imshow("All bb", output);
	//cv::waitKey(0);
	return hands;
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


void histogram(cv::Mat img)
{
	cv::Mat histogram;
	//cv::calcHist(img, 3, cv::noArray(), histogram);
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

void trash()
{
	/*cv::Mat edges;
	cv::Canny(img, edges, 10, 40);

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(edges, contours, hierarchy, cv::RetrievalModes::RETR_LIST, cv::ContourApproximationModes::CHAIN_APPROX_NONE);
	int contoursIdx = 0;
	double maxArea = 0.0;
	for (int i = 0; i < contours.size(); i++)
	{
		double tmp = cv::contourArea(contours);
		if (tmp > maxArea)
		{
			contoursIdx = i;
			maxArea = tmp;
		}
	}
	cv::Mat mask = cv::Mat::zeros(edges.rows, edges.cols, CV_8UC3 );
	cv::fillConvexPoly(mask, contours[contoursIdx], (255, 255, 255));
	cv::imshow("edges", mask);
	cv::waitKey(0);
	return 0;*/

	/*int result_cols = test_img.cols - hand_img.cols + 1;
	int result_rows = test_img.rows - hand_img.rows + 1;
	cv::Mat result;
	result.create(result_rows, result_cols, CV_32FC1);*/
	//float rslt = bowTest(test_img);
	//std::cout <<"\n" << "result: " << rslt << "\n";
	//return 0;

	//Preprocessing 
	//cv::Mat tmp, res;

	//cv::blur(img, tmp, cv::Size(5, 5));

	//cv::cvtColor(tmp, tmp, cv::COLOR_BGR2HSV);
	///*cv::imshow("prepro", tmp);
	//cv::waitKey(0);*/
	//
	//cv::inRange(img, cv::Scalar(0, 50, 0), cv::Scalar(85, 255, 209),  tmp);

	//cv::erode(tmp, tmp, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));

	//cv::dilate(tmp, tmp, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));	

	////
	//cv::bitwise_and(img, img,res, tmp);
	//
	//img = res;
	/*cv::imshow("original", img);
	cv::waitKey(0); */

	/*cv::imshow("prepro", tmp);
	cv::waitKey(0);*/
	/*cv::imshow("res", res);
	cv::waitKey(0);*/
	// End preprocessing
}

