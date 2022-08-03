
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
	float mean_imagenet[3] = {0.485, 0.456, 0.406};
	float std_imagenet[3] = {0.229, 0.224, 0.225};
	loader.set_transform(224, 224, mean_imagenet, std_imagenet);
	ncnn::Mat mat;
	mat = loader.item();
	mat = loader.item();
	return 0; 
}
