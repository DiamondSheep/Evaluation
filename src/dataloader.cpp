#include "dataloader.hpp"

namespace evaluate {
// Transform
Transform::Transform() {
    set_crop_size(0, 0);
    for (size_t i = 0; i < 3; ++i) {
        m_mean[i] = 0;
        m_std[i] = 0;
    }
}
Transform::Transform(const int width, const int height, 
                     const float mean[3], const float std[3]) {
    set_crop_size(width, height);
    set_normalize(mean, std);
}

// DataLoader
bool DataLoader::open(const std::string& source) {
    //double start = get_current_time();
    m_source = source;
    m_dir = opendir(m_source.c_str());
    //double end = get_current_time();
    //std::cout << "time: " << end - start << std::endl;
    return true;
}
ncnn::Mat DataLoader::item() {
    
    // load image file and output ncnn mat

    if ((m_dir_ptr = readdir(m_dir)) == NULL) {
        std::cout << "dataloader load fail." << std::endl;
    }
    if(m_dir_ptr->d_name[0] == '.') {
        return item();
    }
    std::string file_name = std::string(m_source) + std::string(m_dir_ptr->d_name);
    std::cout << "file name: " << file_name << std::endl;

    cv::Mat cv_mat = cv::imread(file_name.c_str(), cv::IMREAD_COLOR);
    std::cout << "mat infor : " << cv_mat.size().width << ", " << cv_mat.size().height << std::endl;
    ncnn::Mat ncnn_mat = ncnn::Mat::from_pixels(cv_mat.data, ncnn::Mat::PIXEL_BGR, cv_mat.cols, cv_mat.rows);
    //m_transform->transform(ncnn_mat);
    return ncnn_mat;
}

void DataLoader::set_transform(const int height, const int width, 
                               const float mean[3], const float std[3]) {
    m_transform->set_crop_size(width, height);
    m_transform->set_normalize(mean, std);
}

} // end of namespace