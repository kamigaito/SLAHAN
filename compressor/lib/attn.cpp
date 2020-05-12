#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/expr.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <type_traits>
#include <time.h>

#include <boost/serialization/vector.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options.hpp>

#include "s2s/dynet/dict.h"
#include "s2s/nn/attn.hpp"
#include "s2s/learner/attn.hpp"
#include "s2s/corpus/corpora.hpp"
#include "s2s/corpus/options.hpp"
#include "s2s/train/train.hpp"
#include "s2s/predict/predict.hpp"
#include "s2s/test/test.hpp"

int main(int argc, char** argv) {
    namespace po = boost::program_options;
    po::options_description bpo("h");
    s2s::s2s_options opts;
    s2s::set_s2s_options(&bpo, &opts);
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, bpo), vm);
    po::notify(vm);
    if(vm.at("mode").as<std::string>() == "train"){
        s2s::add_s2s_options_train(&vm, &opts);
        s2s::check_s2s_options_train(&vm, opts);
        std::string file_name = opts.rootdir + "/options.txt";
        struct stat st;
        if(stat(opts.rootdir.c_str(), &st) != 0){
            mkdir(opts.rootdir.c_str(), 0775);
        }
        std::ofstream out(file_name);
        boost::archive::text_oarchive oa(out);
        oa << opts;
        out.close();
        dynet::initialize(argc, argv);
        s2s::train<s2s::encoder_decoder_attn, s2s::attn_learner>(opts);
    }else if(vm.at("mode").as<std::string>() == "predict"){
        s2s::check_s2s_options_predict(&vm, opts);
        dynet::initialize(argc, argv);
        s2s::predict<s2s::encoder_decoder_attn>(opts);
    }else if(vm.at("mode").as<std::string>() == "test"){
        s2s::add_s2s_options_train(&vm, &opts);
        s2s::check_s2s_options_train(&vm, opts);
        dynet::initialize(argc, argv);
        s2s::test<s2s::encoder_decoder_attn>(opts);
    }else{
        std::cerr << "Mode does not exist !"<< std::endl;
        assert(false);
    }
}
