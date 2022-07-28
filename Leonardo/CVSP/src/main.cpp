///@name Leonardo Sforzin

#include "utilities.h"
#include "detection.h"
#include "segmentation.h"

#define PURPLE cv::Scalar(255,0,127)
#define GREEN cv::Scalar(0,255,0)
#define BLUE cv::Scalar(255,0,0)
#define RED cv::Scalar(0,0,255)

#define EVALUATION 1 // set to 0 if you don't want to evaluate accuracy
// to evaluate accuracy you must have folders det and mask as in https://drive.google.com/drive/folders/1ORmMRRxfLHGLKgqHG-1PKx1ZUCIAJYoa

int main(int argc, char** argv)
{
	// Check if the command line arguments are valid
	int executionCode = checkInput(argc, argv);
	// If they are not valid, print a message and terminate
	if (executionCode == -1)
	{
		std::cout << "\n" << "Check command line arguments" << "\n";
		return -1;
	}

	// Colours for segmentation
	std::vector<cv::Scalar> colours{ PURPLE, GREEN, BLUE, RED };

	// Image of the hand (test hand) that will be used for HOG comparison
	cv::Mat hand_img = cv::imread("test_hand.png", cv::IMREAD_GRAYSCALE);
	// If hand_img.png is missing, print a message and terminate
	if (hand_img.empty())
	{
		std::cout << "\n" << "test_hand.png not found, please insert test_hand.png in the folder where main.cpp is" << "\n";
		return -1;
	}
	// Resize the image for HOG (Histogram of Oriented Gradients)
	//cv::resize(hand_img, hand_img, cv::Size(1280, 720));
	// Vector that will store the "test hands" descriptors
	std::vector<std::vector<float>> handsDescriptor;
	// Vector used for Template Matching
	std::vector<cv::Mat> handVector;
	handsDescriptor.push_back(hog(hand_img));
	handVector.push_back(hand_img);
	//Obtain rotated versions of hand_image.png
	for (int i = 0; i < 3; i++)
	{
		cv::rotate(hand_img, hand_img, cv::ROTATE_90_CLOCKWISE);
		handsDescriptor.push_back(hog(hand_img));
		handVector.push_back(hand_img);
	}
	// Create the structure to store test hands descriptors ("features")
	cv::Mat hogft(handsDescriptor[0].size(), 4, CV_32FC1);
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < handsDescriptor[0].size(); j++)
		{
			hogft.at<float>(j, i) = handsDescriptor[i].at(j);
		}
	}

	// Get path 
	std::string path = argv[2];
	// Get image/images
	std::vector<cv::String> filenames;
	std::vector<cv::Mat> images;
	std::vector<cv::Mat> det_results;
	std::vector<cv::Mat> seg_results;
	// Case: set of images
	if (executionCode == 20)
	{
		path += "/*.jpg";
		cv::glob(path, filenames);
		for (int i = 0; i < filenames.size(); i++)
		{
			cv::Mat tmp = cv::imread(filenames[i]);
			if (tmp.empty())
			{
				std::cout << "\n" << "Cannot load " << filenames[i] << ", please make sure the provided path is correct" << "\n";
				return -1;
			}
			det_results.push_back(tmp);
			tmp = cv::imread(filenames[i]);
			seg_results.push_back(tmp);
			tmp = cv::imread(filenames[i]);
			images.push_back(tmp);
			// Note: the same image is re-read for each vector because pushing the same "tmp" in each vector creates problem
			// (i.e. changes in an image in "det_results" would be reflected in "seg_results" and "images")
		}
	}
	// Case: single image
	else
	{
		cv::Mat tmp = cv::imread(path);
		if (tmp.empty())
		{
			std::cout << "\n" << "Cannot load " << path << ", please make sure the provided path is correct" << "\n";
			return -1;
		}
		images.push_back(tmp);
	}
	// Used to store binarized images of the segmentation
	std::vector<cv::Mat> seg_mask;
	// Used to store "detectHands()"' outputs
	std::vector<std::vector<cv::Rect>> bb;

	// Detection and segmentation
	for (int k = 0; k < images.size(); k++)
	{
		std::vector<cv::Rect> detectionOutput;
		// Get bb's of the given image
		detectionOutput = detectHands(images[k], hogft, handVector);
		bb.push_back(detectionOutput);

		// Draw bb's 
		if (detectionOutput.size() != 0)
		{
			for (int i = 0; i < detectionOutput.size(); i++)
			{
				cv::rectangle(det_results[k], detectionOutput[i], cv::Scalar(0, 255, 0));
			}
		}
		// Show result of detection
		std::string detection = "Detection image " + std::to_string(k + 1);
		cv::imshow(detection, det_results[k]);
		cv::waitKey(50);

		// Segmentation
		
		// Black image to store the mask
		cv::Mat black = cv::Mat::zeros(seg_results[k].rows, seg_results[k].cols, seg_results[k].type());
		if (detectionOutput.size() != 0)
		{
			// Color hands
			for (int i = 0; i < detectionOutput.size(); i++)
			{
				cv::Mat floodedRect = floodfill(seg_results[k], detectionOutput[i], colours[i]);
				floodedRect.copyTo(seg_results[k](detectionOutput[i]));

				//Get the mask
				floodedRect.copyTo(black(detectionOutput[i]));
			}
			cv::cvtColor(black, black, cv::COLOR_BGR2GRAY);
			// Binarization, needed to assess segmentation accuracy
			cv::threshold(black, black, 0, 255, cv::THRESH_BINARY);
			seg_mask.push_back(black);
		}
		// If no hand was detected, our mask will just be a black image
		else
		{
			cv::cvtColor(black, black, cv::COLOR_BGR2GRAY);
			cv::threshold(black, black, 0, 0, cv::THRESH_BINARY);
			seg_mask.push_back(black);
		}
		std::string segmentation = "Segmentation image " + std::to_string(k + 1);
		cv::imshow(segmentation, seg_results[k]);
		cv::waitKey(50);
		cv::destroyWindow(detection);
		cv::destroyWindow(segmentation);

	}
	// Used to assess accuracy of detection and segmentation
	if (EVALUATION)
	{
		// Accuracy detection

		// Used to store the ground truth of the detection
		std::vector<cv::String> detGroundTruthFiles;
		// Name of the folder containing the ground truth
		std::string detpath = "det";
		detpath += "/*.txt";
		cv::glob(detpath, detGroundTruthFiles);
		// Used to store the IoU value of each bb
		float IoU = 0.0;
		for (int i = 0; i < bb.size(); i++)
		{
			std::vector<float> ithIoU = IoUDetection(bb[i], detGroundTruthFiles[i]);
			if (ithIoU.size() != 0)
			{
				// Write the result in a file (one file per input image)
				std::ofstream file;
				std::string outputFileName = "DetResult" + std::to_string(i + 1) + ".txt";
				file.open(outputFileName);
				for (int j = 0; j < ithIoU.size(); j++)
				{
					file << std::to_string(ithIoU[j]) << "\n";
					IoU += ithIoU[j];
				}
				file.close();
			}
			// TBD: write one 0 for each missed bb
			else
			{
				std::ofstream file;
				std::string outputFileName = "DetResult" + std::to_string(i + 1) + ".txt";
				file.open(outputFileName);
				file << std::to_string(0) << "\n";
				file.close();
			}
		}
		IoU = IoU / 54;
		//Total IoU
		std::ofstream file;
		std::string outputFileName = "Total_IoU.txt";
		file.open(outputFileName);
		file << std::to_string(IoU) << "\n";
		file.close();

		// Accuracy segmentation

		// Used to store the ground truth of the segmentation
		std::vector<cv::String> segGroundTruthFiles;
		// Name of the folder containing the ground truth
		std::string segpath = "mask";
		segpath += "/*.png";
		cv::glob(segpath, segGroundTruthFiles);

		for (int i = 0; i < seg_mask.size(); i++)
		{
			// Write the result in a file (one file per input image)
			float ithpixelAccuracy = accuracy(seg_mask[i], segGroundTruthFiles[i]);
			std::ofstream file;
			std::string outputFileName = "SegResult" + std::to_string(i + 1) + ".txt";
			file.open(outputFileName);
			file << std::to_string(ithpixelAccuracy) << "\n";
			file.close();
		}
	}
	return 0;
}
