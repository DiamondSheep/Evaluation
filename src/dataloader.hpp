#ifndef __DATALOADER_H__
#define __DATALOADER_H__

#include <vector>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/noncopyable.hpp>
#include <dirent.h>
// ncnn 
#include "mat.h"
#include "layer/crop.h"
// ours
#include "utils.hpp"

namespace evaluate {

class Transform {
public:
    struct image_size {
        image_size(int w, int h)
        : width(w), height(h){}
        int width;
        int height;
    };
    
    Transform(const int width, const int height, 
              const float mean[3], const float std[3])
    : m_size(width, height) {
        for (size_t i = 0; i < 3; ++i) {
            m_mean[i] = mean[i];
            m_std[i] = std[i];
        }
    }
    void load_cv_mat(cv::Mat& image) {
        m_input = &image;
    }
    void to_ncnn_mat(ncnn::Mat& mat) {
        mat = ncnn::Mat::from_pixels(m_input->data, ncnn::Mat::PIXEL_BGR, m_input->cols, m_input->rows);
    }
    void center_crop() {

    }
    void normalize(ncnn::Mat& mat, const float mean[3], const float std[3]) {
        for (size_t i = 0; i < 3; ++i) {
            m_mean[i] = mean[i];
            m_std[i] = std[i];
        }
        normalize(mat);
    }
    void normalize(ncnn::Mat& mat) {
        mat.substract_mean_normalize(m_mean, m_std);
    }
private:
    image_size m_size;
    float m_mean[3];
    float m_std[3];
    cv::Mat* m_input;
};

class DataLoader : boost::noncopyable {
public:
    //DataLoader()=default;
    bool open(const std::string& source);
    void item(cv::Mat& mat);
    void transform(cv::Mat& mat) {}
private:
    std::string m_source;
    DIR* m_dir;
    struct dirent* m_dir_ptr;
};

} // end of namespace

#endif