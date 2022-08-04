
#include "dataloader.hpp" 
#include "evaluate.hpp"
#include <string>
#include <fstream>

int main(){ 
	std::cout << " -- Project: evaluate" << std::endl; 
	std::string source_dir = "/mnt/data/dataset/imagenet/images_val/"; 

	evaluate::Evaluate eval(source_dir, "squeezenet");
	eval.process();
	return 0; 
}
