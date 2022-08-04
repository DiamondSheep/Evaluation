#include "dataloader.hpp"

namespace evaluate {
// Transform
Transform::Transform() 
: set_flag(false){
    set_crop_size(0, 0);
    for (size_t i = 0; i < 3; ++i) {
        m_mean[i] = 0;
        m_std[i] = 0;
    }
}
Transform::Transform(const int width, const int height, 
                     const float mean_[3], const float std_[3])
: set_flag(false){
    set_crop_size(width, height);
    set_normalize(mean_, std_);
}
void Transform::center_crop(ncnn::Mat& mat) {
    int w = mat.w, h = mat.h;
    int roix = (w - m_width) / 2;
    int roiy = (h - m_height) / 2;
    if (m_width <= 0 || m_width > w || m_height <= 0 || m_height > h) {
        std::cout << "Error: Target size (" << m_width << ", " << m_height 
                  << ") is illegal for image size (" << w << ", " << h
                  << "). "
                  << std::endl;
        return;
    }
    ncnn::Mat::from_pixels_roi(mat, ncnn::Mat::PIXEL_BGR, 
                                w, h, roix, roiy,
                                m_width, m_height
                                );
}
void Transform::center_crop(ncnn::Mat& mat, const int width, const int height) {
    set_crop_size(width, height);
    center_crop(mat);
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
ncnn::Mat DataLoader::item() {
    ncnn::Mat ncnn_mat;
    if (!isOpened()) {
        std::cout << "Error: DataLoader is not opened. " << std::endl;
        return ncnn_mat;
    }
    if ((m_dir_ptr = readdir(m_dir)) == NULL) {
        std::cout << "All files loaded. " << std::endl;
    }
    if(m_dir_ptr->d_name[0] == '.') {
        return item();
    }
    std::string file_name = std::string(m_source) + std::string(m_dir_ptr->d_name);
    cv::Mat cv_mat = cv::imread(file_name.c_str(), cv::IMREAD_COLOR);
    if (!cv_mat.data) {
        std::cout << "Error: File " << file_name << " can not be read. " << std::endl;
        return ncnn_mat;
    }
    ncnn_mat = ncnn::Mat::from_pixels(cv_mat.data, ncnn::Mat::PIXEL_BGR, cv_mat.cols, cv_mat.rows);
    if (m_transform->isSet()) {
        m_transform->transform(ncnn_mat);
    }
    return ncnn_mat;
}

void DataLoader::set_transform(const int height, const int width, 
                               const float mean_[3], const float std_[3]) {
    m_transform->set_crop_size(width, height);
    m_transform->set_normalize(mean_, std_);
}

} // end of namespace