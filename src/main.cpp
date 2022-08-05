
#include "dataloader.hpp" 
#include "evaluate.hpp"
#include <string>
#include <fstream>

int main(){ 
	std::cout << " -- Project: evaluate" << std::endl; 
	std::string source_dir = "/mnt/data/dataset/imagenet/images_val/"; 
	//std::string source_dir = "demo/"; 
	evaluate::Evaluate eval(source_dir, "squeezenet_v1.1");
	eval.process();
	return 0; 
}
