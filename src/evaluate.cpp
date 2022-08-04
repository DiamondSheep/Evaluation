#include "evaluate.hpp" 

namespace evaluate {
Network::Network()
: is_loaded(false) {
	init();
}
void Network::init() {
	// set model
	if (m_net.load_param("../models/squeezenet_v1.1.param"))
        exit(-1);
    if (m_net.load_model("../models/squeezenet_v1.1.bin"))
        exit(-1);
	is_loaded = true;
}
void Network::process(const ncnn::Mat& image) {
	std::vector<float> cls_scores;
	ncnn::Extractor ex = m_net.create_extractor();
	ex.input("data", image);

	ncnn::Mat out;
	ex.extract("prob", out);
	cls_scores.resize(out.w);
    for (int j = 0; j < out.w; j++)
    {
        cls_scores[j] = out[j];
    }
	std::cout << "Inference Done. " << std::endl;
}

Evaluate::Evaluate(const std::string& source_dir, const std::string& model_name)
: m_dataloader(source_dir), m_network(new Network()) {
	// TODO : init network with model name
	init();
}
void Evaluate::init() {
	// set dataloader
	float mean_imagenet[3] = {0.485, 0.456, 0.406};
	float std_imagenet[3] = {0.229, 0.224, 0.225};
	m_dataloader.set_transform(224, 224, mean_imagenet, std_imagenet);
}
void Evaluate::process() {
	ncnn::Mat mat;
	mat = m_dataloader.item();
	if (m_network->isLoaded()) {
		m_network->process(mat);
	}
}

}