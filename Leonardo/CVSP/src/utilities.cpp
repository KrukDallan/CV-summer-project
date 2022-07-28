#include "utilities.h"

// @Alberto Makosa
int checkInput(int argc, char** argv)
{
	if (argc <= 1) 
	{
		std::cout << "\n" << "Use one of the following" << "\n";
		std::cout << "\n"<< " -f <image's path>" << "\t" << "To use a single file (image)" << "\n";
		std::cout << "\n" << " -s <folder's path>" << "\t" << "To use a set of images" << "\n";
		
		return -1;
	}

	for (int i = 0; i < argc; i++)
	{
		if (strcmp(argv[i], "-f") == 0) 
		{
			return 1;
		}	
		if (strcmp(argv[i], "-s") == 0)
		{
			return 20;
		}
	}

	// If no -f or -s is found
	std::cout << "\n" << "Use one of the following" << "\n";
	std::cout << "\n" << " -f <image's path>" << "\t" << "To use a single file (image)" << "\n";
	std::cout << "\n" << " -s <folder's path>" << "\t" << "To use a set of images" << "\n";
	return -1;
}