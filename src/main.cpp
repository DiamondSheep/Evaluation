
#include "dataloader.hpp" 
#include <string>
#include <fstream>

int main(){ 
	std::cout << " -- Project: evaluate" << std::endl; 
	std::string images_dir = "/mnt/data/dataset/imagenet/images_val/"; 
	evaluate::DataLoader loader;
	if (loader.open(images_dir) == 0) {
		std::cout << "DataLoader opened." << std::endl;
	}
	cv::Mat mat;
	loader.item(mat);
	loader.item(mat);
	return 0; 
}
