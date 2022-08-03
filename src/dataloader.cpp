#include "dataloader.hpp"

namespace evaluate {
bool DataLoader::open(const std::string& source) {
    //double start = get_current_time();
    m_source = source;

    m_dir = opendir(m_source.c_str());

    //double end = get_current_time();
    //std::cout << "time: " << end - start << std::endl;
    return true;
}
void DataLoader::item(cv::Mat& mat) {
    if ((m_dir_ptr = readdir(m_dir)) == NULL) {
        std::cout << "dataloader load fail." << std::endl;
    }
    if(m_dir_ptr->d_name[0] == '.') {
        item(mat);
        return;
    }
    std::string file_name = std::string(m_source) + std::string(m_dir_ptr->d_name);
    std::cout << "file name: " << file_name << std::endl;
    mat = cv::imread(file_name.c_str(), cv::IMREAD_COLOR);
    std::cout << "mat infor : " << mat.size().height << ", " << mat.size().width << std::endl;
}
} // end of namespace