#ifndef __DATALOADER_H__
#define __DATALOADER_H__

#include <vector>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/noncopyable.hpp>
#include <dirent.h>

#include "utils.hpp"

namespace evaluate {

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