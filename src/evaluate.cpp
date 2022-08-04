#include "evaluate.hpp" 
#include <stdlib.h>

namespace evaluate {
Network::Network(const std::string& net_name)
: m_net(new ncnn::Net), is_loaded(false), m_net_name(net_name){
	init();
}
void Network::init() {
	std::string net_path = "../models/" + m_net_name;
	std::string net_param_path = net_path + ".param";
	std::string net_bin_path = net_path + ".bin";
	// set model
	if (m_net->load_param(net_param_path.c_str()))
        exit(-1);
    if (m_net->load_model(net_bin_path.c_str()))
        exit(-1);
	m_ex.reset(new ncnn::Extractor(m_net->create_extractor()));
	is_loaded = true;
}
void Network::process(const ncnn::Mat& image) {
	ncnn::Mat out;
	m_ex.reset(new ncnn::Extractor(m_net->create_extractor()));
	m_ex->input("data", image);
	m_ex->extract("prob", out);
	cls_scores.resize(out.w);
    for (int i = 0; i < out.w; i++) {
		cls_scores[i] = std::make_pair(out[i], i);
    }
}
std::vector<int> Network::topk(int k) {
	std::vector<int> result(k, 0);
	std::partial_sort(cls_scores.begin(), cls_scores.begin() + k, cls_scores.end(),
                      [](std::pair<float, int>& a, std::pair<float, int>& b){ return a.first > b.first; });
	for (int i = 0; i < k; i++) {
        float score = cls_scores[i].first;
        int index = cls_scores[i].second;
		result[i] = index;
        //fprintf(stderr, "%d = %f\n", index, score);
    }
	return result;
}
std::vector<int> Network::top5() {
	return topk(5);
}

Evaluate::Evaluate(const std::string& source_dir, const std::string& model_name)
: m_dataloader(source_dir), m_network(new Network(model_name)), top1_accumulate(0.0), top5_accumulate(0.0), m_count(0) {
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
	if (!m_network->isLoaded()) {
		return;
	}
	while (true) {
		data = m_dataloader.item();
		const ncnn::Mat& image = data.first;
		int label = data.second;
		if (label == -1) {
			std::cout << "Evaluation Done." << std::endl;
			return;
		}
		m_network->process(image);
		std::vector<int> result = m_network->top5();
		accumulate(result, label);
		system("clear"); 
		std::cout << "Top1: " << top1_accuracy() << ", Top5: " << top5_accuracy() << std::endl;
	}
}
void Evaluate::accumulate(const std::vector<int>& result, int label) {
	++m_count;
	if (result.size() == 5) {
		auto it = std::find(result.begin(), result.end(), label);
		if (it != result.end()) {
			++top5_accumulate;
		}
	}
	if (result.front() == label) {
		++top1_accumulate;
	}
}
}