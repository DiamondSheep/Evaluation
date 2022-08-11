
#include "dataloader.hpp" 
#include "evaluate.hpp"
#include <string>
#include <fstream>

int main(int argc, char* argv[]){ 
	std::cout << " -- Project: evaluate" << std::endl; 
	if (argc != 3) {
		std::cout << "Too less argments: " << std::endl;
		std::cout << "./evaluate [model_name] [data_path]" << std::endl;
		return 1;
	}
	std::string source_dir = argv[2];  
	// "/mnt/data/dataset/imagenet/images_val/"; 
	evaluate::Evaluate eval(source_dir, argv[1]);
	eval.process();
	return 0; 
}
