// @name Leonardo Sforzin

#include "detection.h"

// Good combination of values:
// scale:1.2 , minNeigh:2 , TM <0.3 , Canny:10,20 , aperture: 1

std::vector<cv::Rect> detectHands(cv::Mat src, cv::Mat hogft, std::vector<cv::Mat> testHands)
{
	// Used to store "correct" bb's
	std::vector<cv::Rect> rectVector;
	cv::Mat img, mask;
	// Used to keep track of the number of hands detected (max 4)
	int counter = 0;
	// Remove the background as preprocessing to help detection
	mask = removeBG(src);
	src.copyTo(img, mask);
	cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
	cv::resize(img, img, cv::Size(1280, 720));
	while (true)
	{
		// Vector to store every rect (bb) found by the cascade of classifiers 
		std::vector<cv::Rect> hands = cascade(img); //temp_hands
		
		if (hands.size() == 0)
		{
			break;
		}
		// Index of the best bb
		int bbIndex = 0;
		// "Accuracy" of the best bounding box
		//double maxvalue = 0.0;
		//// Minimum distance between HoG descriptors
		//double globalmin = DBL_MAX;
		//for (int i = 0; i < hands.size(); i++)
		//{
		//	// Compare Hog descriptors
		//	double tmp = comparehog(img, hands[i], hogft);
		//	if (tmp < globalmin)
		//	{
		//		globalmin = tmp;
		//		bbIndex = i;
		//	}
		//}
		std::vector<float> outputVector;
		float maxvalue = 0.0;
		for (int i = 0; i < testHands.size(); i++)
		{
			outputVector = templateMatching(img, hands, testHands[i]);
			if (outputVector[1] > maxvalue)
			{
				bbIndex = (int)outputVector[0];
				maxvalue = outputVector[1];
			}
		}
		if (maxvalue < 0.1)
		{
			break;
		}
		// Cover the hand so that the cascade of classifiers doesn't detect it when re-applied
		cv::rectangle(img, hands[bbIndex], cv::Scalar(255, 255, 255), -1);
		rectVector.push_back(hands[bbIndex]);
		counter++;
		// Max number of hands per image
		if (counter == 4)
		{
			break;
		}
	}
	return rectVector;
}

std::vector<cv::Rect> cascade(cv::Mat img)
{
	// Load trained model
	cv::CascadeClassifier my_cascade("cascade220726.xml");

	std::vector<cv::Rect> hands;
	cv::Mat output = img;
	my_cascade.detectMultiScale(img, hands, 1.2, 2, 0, cv::Size(100, 100), cv::Size(5500, 5500));//working pretty good

	//for (int i = 0; i < hands.size(); i++)
	//{
	//	cv::rectangle(output, hands[i], cv::Scalar(0, 255, 0));
	//}
	//cv::imshow("All bb", output);
	//cv::waitKey(0);
	return hands;
}

std::vector<float> hog(cv::Mat img)
{
	cv::HOGDescriptor hog = cv::HOGDescriptor::HOGDescriptor();
	std::vector<float> descriptor;
	hog.compute(img, descriptor);

	return descriptor;
}

double comparehog(cv::Mat img, cv::Rect bb, cv::Mat hogft)
{
	double distance = DBL_MAX;
	cv::Mat roi = img(bb);
	cv::resize(roi, roi, cv::Size(1280, 720));

	// Get the descriptor
	std::vector<float> imghog = hog(roi);

	// Find the min distance between descriptors
	for (int i = 0; i < hogft.cols; i++)
	{
		double mindist = 0.0;
		for (int j = 0; j < hogft.rows; j++)
		{
			mindist += std::abs(hogft.at<float>(j, i) - imghog.at(j));
		}
		if (mindist < distance)
		{
			distance = mindist;
		}
	}
	return distance;
}

std::vector<float> IoUDetection(std::vector<cv::Rect> hands, std::string path)
{
	std::vector<float> IoU;
	// If no bb has been found
	if (hands.size() == 0)
	{
		return IoU;
	}

	// get the file
	std::ifstream infile(path);

	std::vector<cv::Rect> correctRects;
	// variables to store info in the file
	int  x = 0, y = 0, width = 0, height = 0;
	// create a rect for each line and store it
	while (infile >> x >> y >> width >> height)
	{
		cv::Rect bb(x, y, width, height);
		correctRects.push_back(bb);
	}
	
	for (int i = 0; i < correctRects.size(); i++)
	{
		float ithIoU = 0.0;
		for (int j = 0; j < hands.size(); j++)
		{
			cv::Rect Intersection = correctRects[i] & hands[j];
			cv::Rect Union = correctRects[i] | hands[j];

			int areaI = Intersection.area();
			if (areaI == 0) continue;

			int areaU = Union.area();

			float tmpIoU = areaI / (float)areaU;
			if (tmpIoU > ithIoU)
			{
				ithIoU = tmpIoU;
			}
		}
		IoU.push_back(ithIoU);
	}
	return IoU;
}

std::vector<float> templateMatching(cv::Mat img, std::vector<cv::Rect> hands, cv::Mat hand_img)
{
	if (hands.size() == 0)
	{
		std::vector<float> output;
		output.push_back(0.0);
		output.push_back(0.0);
		return output;
	}
	cv::Mat result;
	std::vector<float> output;
	double maxMaxVal = 0.0;
	int bbIndex = 0;
	for (int i = 0; i < hands.size(); i++)
	{
		cv::Mat subimg = img(hands[i]);
		cv::matchTemplate(subimg, hand_img, result, cv::TemplateMatchModes::TM_CCOEFF_NORMED); //

		double minVal; double maxVal; cv::Point minLoc; cv::Point maxLoc;
		cv::Point matchLoc;

		cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
		if (maxVal > maxMaxVal)
		{
			maxMaxVal = maxVal;
			bbIndex = i + 0.0;
		}
	}

	if (maxMaxVal < 0.3) // 0.5 works well with test image 01
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

cv::Mat removeBG(cv::Mat src)
{
	// Params
	// Good params: th1:10, th2:50
	double th1 = 10;
	double th2 = 20;

	// detect edges
	cv::Mat edges;
	cv::Mat img;
	img = sharpenImage(src);
	cv::Canny(img, edges, th1, th2);
	cv::dilate(edges, edges, cv::Mat());
	cv::erode(edges, edges, cv::Mat());
	/*cv::imshow("edges", edges);
	cv::waitKey(0);*/

	// find contours
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(edges, contours, hierarchy, cv::RetrievalModes::RETR_LIST, cv::ContourApproximationModes::CHAIN_APPROX_NONE);
	int contoursIdx = 0;
	double maxArea = 0.0;
	for (int i = 0; i < contours.size(); i++)
	{
		double tmp = cv::contourArea(contours[i]);
		if (tmp > maxArea)
		{
			contoursIdx = i;
			maxArea = tmp;
		}
	}
	cv::Mat mask = cv::Mat::zeros(edges.rows, edges.cols, CV_8UC3);

	cv::drawContours(mask, contours, contoursIdx, cv::Scalar(255, 255, 255), -1);
	/*cv::imshow("mask", mask);
	cv::waitKey(0);*/
	return mask;
}

cv::Mat sharpenImage(cv::Mat img)
{
	cv::Mat dst;
	cv::Mat abs_dst;
	cv::Mat img_gray;
	
	//cv::GaussianBlur(img, img_gray, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
	cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
	cv::Laplacian(img_gray, dst, CV_16S, 3, 1, 0, cv::BORDER_DEFAULT);
	cv::convertScaleAbs(dst, abs_dst);
	return abs_dst;
}