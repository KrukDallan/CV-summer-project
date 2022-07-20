///@name Leonardo Sforzin

#include "CVSPfunctions.h"


//PROBLEMS:
//1) Sift is NOT rotation invariant --> should we add vocabularies of rotated images
//2) ONLY ONE DICTIONARY OF VISUAL WORDS IS NEEDED


//Things to add?:
//1) histograms
//2) SVM
//3) labels
//4) brute force matcher for histograms


//Notes: histograms are used to train a classifier, like svm
// After creating the codebook with k centroids, iterate through every image used and
// search for words present both in he dictionary and in the image, then increase the count of that particular word <-- this is how histograms are created
int main(int argc, char** argv)
{
	int executionCode = 1;

	// Training of the BoW
	if (executionCode == 1)
	{
		std::string path = argv[1];
		path += "/*.jpg";
		cv::String folderString(path);
		std::vector<cv::String> fileNames;
		cv::glob(folderString, fileNames, false);

		std::vector<cv::Mat> greyScale(fileNames.size());

		for (int i = 0; i < fileNames.size(); i++)
		{
			cv::Mat img = cv::imread(fileNames[i], cv::IMREAD_GRAYSCALE);
			greyScale[i] = img;
		}

		std::vector<cv::KeyPoint> keyPoints;
		cv::Mat descriptor, features;
		cv::Ptr<cv::SiftDescriptorExtractor> extractor = cv::SiftDescriptorExtractor::create();
		// variables used to display the progress of the for loop
		int increment5 = 0.05 * fileNames.size();
		int countdown = increment5;
		int percent5 = 0;
		std::cout << "\n";
		//Compute the features for each image of the "training set"
		for (int i = 0; i < fileNames.size(); i++)
		{
			//detects features
			extractor->detect(greyScale[i], keyPoints);
			//Computes the descriptors for a set of keypoints detected in an image
			extractor->compute(greyScale[i], keyPoints, descriptor);
			//insert them in features
			features.push_back(descriptor);
			if (--countdown == 0)
			{
				percent5++;
				std::cout << "\r" << std::string(percent5, '|') << percent5 * 5 << "%";
				countdown = increment5;
				std::cout.flush();
			}
		}
		std::cout << "\n" << "\n" << "Features computed" << "\n";

		//build BoW trainer
		// dictSize (dictionary size) is the "k" of kmeans
		int dictSize = 50;
		//Termination criteria
		cv::TermCriteria tc(cv::TermCriteria::Type::COUNT, 50, 0.001);
		//attempts
		int attempts = 1;
		//flags
		int flags = cv::KMEANS_PP_CENTERS;
		//create BoW trainer
		cv::BOWKMeansTrainer bowTrainer(dictSize, tc, attempts, flags);
		//cluster the feature vectors
		std::cout << "\n" << "Starting clustering" << "\n";
		cv::Mat dict = bowTrainer.cluster(features);
		//store the vocabulary
		std::cout << "\n" << "Storing the vocabulary" << "\n";
		//cv::FileStorage fs("cards-courtyard-bt.yml", cv::FileStorage::WRITE);
		//cv::FileStorage fs("cards-livingroom-hs.yml", cv::FileStorage::WRITE);
		cv::FileStorage fs("bag-of-words.yml", cv::FileStorage::WRITE);
		fs << "vocabulary" << dict;
		fs.release();
	}
	//Test section of the BoW algo
	else if (executionCode == 2)
	{
		bowLastStep();
	}
}

void bowLastStep()
{
	//test each image against each vocabulary computed in step 1
	std::vector<std::string> vocabs{ "bag-of-words.yml" };
	std::cout << "\n" << "Starting test" << "\n";
	for (int i = 0; i < vocabs.size(); i++)
	{
		cv::Mat dict;
		cv::FileStorage fs(vocabs[i], cv::FileStorage::READ);
		fs["vocabulary"] >> dict;
		fs.release();

		cv::Ptr<cv::DescriptorMatcher> matcher = cv::FlannBasedMatcher::create();
		cv::Ptr<cv::FeatureDetector> detector = cv::SiftFeatureDetector::create();
		cv::Ptr<cv::DescriptorExtractor> extractor = cv::SiftDescriptorExtractor::create();
		cv::BOWImgDescriptorExtractor bowDE(extractor, matcher);
		bowDE.setVocabulary(dict);

		std::string descr("descriptor with");
		descr += vocabs[i];
		cv::FileStorage fs1(descr, cv::FileStorage::WRITE);
		cv::Mat img = cv::imread("D:\\Desktop2\\MAGISTRALE\\Primo_anno-secondo_semestre\\ComputerVision\\0FinalProject\\_LABELLED_SAMPLES\\CHESS_COURTYARD_B_T\\frame_0197.jpg", cv::IMREAD_GRAYSCALE);
		
		std::vector<cv::KeyPoint> keypoints;
		detector->detect(img, keypoints);
		cv::Mat bowDescriptor;
		std::vector<std::vector<int>> pointIdxsOfClusters;
		bowDE.compute(img, keypoints, bowDescriptor, &pointIdxsOfClusters);
		
		//fs1 << "testImage" << bowDescriptor;
		fs1 << "pointIdxsOfClusters" << pointIdxsOfClusters;
		fs1.release();
		std::cout << "\r" << "Vocab " << i << " tested";
		std::cout.flush();
	}
	

	//To find the nearest cluster --> SVM (multiclass)?


}

