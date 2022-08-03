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
    void set_crop_size(const int width, const int height) {
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
    
    // Operations
    void center_crop(ncnn::Mat& mat) {
        int w = mat.w, h = mat.h;
        int roix = (w - m_width) / 2;
        int roiy = (h - m_height) / 2;
        ncnn::Mat::from_pixels_roi(mat, ncnn::Mat::PIXEL_BGR, 
                                   w, h, roix, roiy,
                                   m_width, m_height
                                   );
    }
    void center_crop(ncnn::Mat& mat, const int width, const int height) {
        set_crop_size(width, height);
        center_crop(mat);
    }
    void normalize(ncnn::Mat& mat) {
        mat.substract_mean_normalize(m_mean, m_std);
    }
    void normalize(ncnn::Mat& mat, const float mean_[3], const float std_[3]) {
        set_normalize(mean_, std_);
        normalize(mat);
    }
    void transform(ncnn::Mat& ncnn_mat) {
        normalize(ncnn_mat);
        std::cout << "crop" << std::endl;
        center_crop(ncnn_mat);
    }
private:
    bool set_flag;
    int m_width;
    int m_height;

    float m_mean[3];
    float m_std[3];
};

class DataLoader : boost::noncopyable {
public:
    //DataLoader()=default;
    bool open(const std::string& source);
    ncnn::Mat item();
    void set_transform(const int height, const int width, const float mean[3], const float std[3]);
private:
    std::string m_source;
    DIR* m_dir;
    struct dirent* m_dir_ptr;
    Transform::ptr m_transform;
};

} // end of namespace

#endif