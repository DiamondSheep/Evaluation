#include "evaluate.hpp" 

namespace evaluate {
Network::Network(const std::string& net_name)
: is_loaded(false), m_net_name(net_name) {
	init();
}
void Network::init() {
	std::string net_path = "../models/" + m_net_name;
	std::string net_param_path = net_path + ".param";
	std::string net_bin_path = net_path + ".bin";
	// set model
	if (m_net.load_param(net_param_path.c_str()))
        exit(-1);
    if (m_net.load_model(net_bin_path.c_str()))
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
: m_dataloader(source_dir), m_network(new Network(model_name)) {
	init();
}
void Evaluate::init() {
	// set dataloader
	// TODO: config
	float mean_imagenet[3] = {0.485, 0.456, 0.406};
	float std_imagenet[3] = {0.229, 0.224, 0.225};
	m_dataloader.set_transform(224, 224, mean_imagenet, std_imagenet);
}
void Evaluate::process() {
	std::pair<ncnn::Mat, int> data;
	data = m_dataloader.item();
	if (m_network->isLoaded()) {
		m_network->process(data.first);
	}
}

std::vector<int> Evaluate::topk(int k) {
	/*
	std::vector<std::pair<float, int> > vec;
    vec.resize(size);
    for (int i = 0; i < size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater<std::pair<float, int> >());

    // print topk and score
    for (int i = 0; i < topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        fprintf(stderr, "%d = %f\n", index, score);
    }
	*/
}

}