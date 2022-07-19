///@name Leonardo Sforzin

#include "CVSPfunctions.h"




int main(int argc, char** argv)
{

	int executionCode = 2;

	if (executionCode == 1)
	{
		std::string path = argv[1];
		path += "/*.jpg";
		cv::String folderString(path);
		std::vector<cv::String> fileNames;
		cv::glob(folderString, fileNames, false);

		std::vector<cv::Mat> greyScale(fileNames.size());

		for (int i = 0; i < fileNames.size() / 2; i++)
		{
			cv::Mat img = cv::imread(fileNames[i], cv::IMREAD_GRAYSCALE);
			greyScale[i] = img;
		}

		//vector clear
		std::vector<cv::KeyPoint> keyPoints;
		cv::Mat descriptor, features;
		cv::Ptr<cv::SiftDescriptorExtractor> extractor = cv::SiftDescriptorExtractor::create();

		for (int i = 0; i < 10; i++)
		{
			//detects features
			extractor->detect(greyScale[i], keyPoints);
			std::cout << "Features detected" << "\n";
			//Computes the descriptors for a set of keypoints detected in an image
			extractor->compute(greyScale[i], keyPoints, descriptor);
			//insert them in features
			features.push_back(descriptor);
		}
		std::cout << "For ended" << "\n";

		//build BoW trainer
		int dictSize = 10;
		//Termination criteria
		cv::TermCriteria tc(cv::TermCriteria::Type::COUNT, 10, 0.001);
		//attempts
		int attempts = 1;
		//flags
		int flags = cv::KMEANS_PP_CENTERS;
		//create BoW trainer
		cv::BOWKMeansTrainer bowTrainer(dictSize, tc, attempts, flags);
		//cluster the feature vectors
		cv::Mat dict = bowTrainer.cluster(features);
		//store the vocabulary
		cv::FileStorage fs("dictionary.yml", cv::FileStorage::WRITE);
		fs << "vocabulary" << dict;
		fs.release();
		/*cv::imshow("0", greyScale[0]);
		cv::waitKey(0);*/
	}
	else if (executionCode == 2)
	{
		bowLastStep();
	}

}

void bowLastStep()
{
	cv::Mat dict;
	cv::FileStorage fs("dictionary.yml", cv::FileStorage::READ);
	fs["vocabulary"] >> dict;
	fs.release();

	cv::Ptr<cv::DescriptorMatcher> matcher = cv::FlannBasedMatcher::create();
	cv::Ptr<cv::FeatureDetector> detector = cv::SiftFeatureDetector::create();
	cv::Ptr<cv::DescriptorExtractor> extractor = cv::SiftDescriptorExtractor::create();
	cv::BOWImgDescriptorExtractor bowDE(extractor, matcher);
	bowDE.setVocabulary(dict);

	cv::FileStorage fs1("descriptor.yml", cv::FileStorage::WRITE);
	cv::Mat img = cv::imread("D:\\Desktop2\\MAGISTRALE\\Primo_anno-secondo_semestre\\ComputerVision\\0FinalProject\\_LABELLED_SAMPLES\\CARDS_COURTYARD_B_T\\frame_0011.jpg", cv::IMREAD_GRAYSCALE);
	std::vector<cv::KeyPoint> keypoints;
	detector->detect(img, keypoints);
	cv::Mat bowDescriptor;
	bowDE.compute(img, keypoints, bowDescriptor);

	fs1 << "img1" << bowDescriptor;
	fs1.release();
}

