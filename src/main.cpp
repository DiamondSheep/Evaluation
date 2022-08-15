
#include "dataloader.hpp" 
#include "evaluate.hpp"
#include <string>
#include <fstream>

int main(int argc, char* argv[]){ 
	std::cout << " -- Project: evaluate" << std::endl; 
	
	std::string model;
	std::string source_dir;  

	if (argc == 3) {
		model = argv[1];
		source_dir = argv[2];  
	}
	else if (argc == 1) {
		std::cout << "Using default setting. " << std::endl;
		model = "squeezenet_v1.1";
		// "/mnt/data/dataset/imagenet/images_val/"; 
		source_dir = "/mnt/data/dataset/imagenet/images_val/";  
	}
	else {
		
		std::cout << "Args count error. " << std::endl;
		std::cout << "./evaluate [model_name] [data_path]" << std::endl;
		return 1;
	}
	
	std::cout << "Evaluating " << model 
			  << " on data path: " << source_dir << std::endl;
	evaluate::Evaluate eval(source_dir, model);
	eval.process();
	
	return 0; 
}
