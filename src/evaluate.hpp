#ifndef __EVALUATE__ 
#define __EVALUATE__ 

#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <boost/noncopyable.hpp>
// ncnn
#include "net.h"
// ours
#include "dataloader.hpp"

namespace evaluate {
// Dataset path: /mnt/data/dataset/imagenet/
class Network : boost::noncopyable {
public:
    typedef std::shared_ptr<Network> ptr;
    Network();
    void init();
    void process(const ncnn::Mat& image);
    bool isLoaded() { return is_loaded; }
private:
    ncnn::Net m_net;
    bool is_loaded;
};
class Evaluate : boost::noncopyable {
public:
    Evaluate(const std::string& source_dir, const std::string& net_name);
    void init();
    void process();
private:
    Network::ptr m_network;
    DataLoader m_dataloader;
};
} // end of namespace
#endif
