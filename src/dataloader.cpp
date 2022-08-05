#include "dataloader.hpp"

namespace evaluate {
// Transform
Transform::Transform() 
: set_flag(false), m_resize(1.0){
    set_size(0, 0);
    for (size_t i = 0; i < 3; ++i) {
        m_mean[i] = 0;
        m_std[i] = 0;
    }
}
Transform::Transform(const int width, const int height, 
                     const float mean_[3], const float std_[3])
: set_flag(false), m_resize(1.0){
    set_size(width, height);
    set_normalize(mean_, std_);
}

// DataLoader
DataLoader::DataLoader()
: m_transform(new Transform()), is_opened(false) {

}
DataLoader::DataLoader(const std::string& source)
: m_transform(new Transform()), is_opened(false) {
    open(source);
}

void DataLoader::open(const std::string& source) {
    //double start = get_current_time();
    m_source = source;
    m_dir = opendir(m_source.c_str());
    //double end = get_current_time();
    //std::cout << "time: " << end - start << std::endl;
    is_opened = true;
}
std::pair<ncnn::Mat, int> DataLoader::item() {
    ncnn::Mat ncnn_mat;
    if (!isOpened()) {
        std::cout << "Error: DataLoader is not opened. " << std::endl;
        return {ncnn_mat, -1};
    }
    if ((m_dir_ptr = readdir(m_dir)) == NULL) {
        std::cout << "All files loaded. " << std::endl;
        return {ncnn_mat, -1};
    }
    if(m_dir_ptr->d_name[0] == '.') {
        return item();
    }
    std::string file_name = std::string(m_source) + std::string(m_dir_ptr->d_name);
    //std::cout << file_name << std::endl;

    // Parse label from filename
    std::size_t label_begin = file_name.find_last_of('_');
    std::size_t label_end = file_name.find_last_of('.');
    int label = atoi(file_name.substr(label_begin+1, label_end - label_begin).c_str());
    // Get image
    cv::Mat cv_mat = cv::imread(file_name.c_str(), cv::IMREAD_COLOR);
    if (!cv_mat.data) {
        std::cout << "Error: File " << file_name << " can not be read. " << std::endl;
        return {ncnn_mat, -1};
    }
    if (m_transform->isSet()) {
        ncnn_mat = m_transform->transform(cv_mat, 224, 224, 0.875);
    }
    return {ncnn_mat, label};
}

void DataLoader::set_transform(const int height, const int width, 
                               const float mean_[3], const float std_[3]) {
    m_transform->set_size(width, height);
    m_transform->set_normalize(mean_, std_);
}

} // end of namespace