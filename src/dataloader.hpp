#ifndef __DATALOADER_H__
#define __DATALOADER_H__

#include <vector>
#include <string>
#include <memory>
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
    typedef std::shared_ptr<Transform> ptr;
    Transform();
    Transform(const int width, const int height, 
              const float mean_[3], const float std_[3]);

    // Settings
    void set_size(const int width, const int height) {
        m_width = width;
        m_height = height;
    }
    void set_normalize(const float mean_[3], const float std_[3]) {
        for (size_t i = 0; i < 3; ++i) {
            m_mean[i] = mean_[i];
            m_std[i] = std_[i];
        }
        set_flag = true;
    }
    bool isSet() {
        return set_flag;
    }
    
    // Operations
    void normalize(ncnn::Mat& mat) {
        mat.substract_mean_normalize(m_mean, m_std);
    }
    void normalize(ncnn::Mat& mat, const float mean_[3], const float std_[3]) {
        set_normalize(mean_, std_);
        normalize(mat);
    }
    ncnn::Mat transform(cv::Mat& cv_mat, int target_width, int target_height, float portion=0.875) {
        int origin_width = cv_mat.cols, origin_height = cv_mat.rows;
        int temp_width = int(target_width / portion);
        int temp_height = int(target_height / portion);
        cv::Size temp_size = cv::Size(temp_width, temp_height);
        
        int roix = (temp_width - target_width) / 2;
        int roiy = (temp_height - target_height) / 2;
        int roiw = target_width, roih = target_height;

        cv::Mat cropped_mat;
        cv::resize(cv_mat, cropped_mat, temp_size, 0, 0);
        ncnn::Mat ncnn_mat = ncnn::Mat::from_pixels_roi_resize(cropped_mat.data, ncnn::Mat::PIXEL_RGB, 
                                cropped_mat.cols, cropped_mat.rows, roix, roiy, roiw, roih, 
                                target_width, target_height);
        normalize(ncnn_mat);
        return ncnn_mat;
    }
private:
    bool set_flag;
    float m_resize;
    int m_width;
    int m_height;

    float m_mean[3];
    float m_std[3];
};

class DataLoader : boost::noncopyable {
public:
    DataLoader();
    DataLoader(const std::string& source);
    void open(const std::string& source);
    bool isOpened() const { return is_opened; }
    std::pair<ncnn::Mat, int> item();
    void set_transform(const int height, const int width, const float mean[3], const float std[3]);
private:
    bool is_opened;
    std::string m_source;
    DIR* m_dir;
    struct dirent* m_dir_ptr;
    Transform::ptr m_transform;
};

} // end of namespace

#endif