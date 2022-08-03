#ifndef __EVALUATE__ 
#define __EVALUATE__ 

#include <iostream>
#include <fstream>
#include <string>
#include <boost/noncopyable.hpp>

#include "net.h"
namespace evaluate {
// Dataset path: /mnt/data/dataset/imagenet/
class Network : boost::noncopyable {
public:
    Network();
    void init();
    void process();
private:
    ncnn::Net m_net;
    //ncnn::Extractor m_ex;
};
} // end of namespace
#endif
