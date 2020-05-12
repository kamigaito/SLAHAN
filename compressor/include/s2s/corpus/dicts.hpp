#include "dynet/expr.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <type_traits>
#include <random>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/program_options.hpp>

#include "s2s/dynet/dict.h"
#include "s2s/corpus/dict_set.hpp"

#ifndef INCLUDE_GUARD_S2S_DICTS_HPP
#define INCLUDE_GUARD_S2S_DICTS_HPP

namespace s2s {

class dicts {

public:

    dict_set_token dict_set_src;
    dict_set_label dict_set_trg;

    void set(const s2s_options &opts){
        // src
        /* resize vectors */
        dict_set_src.init(opts.enc_feature_vocab_size.size());
        /* construct dictionary */
        std::cerr << "Reading source language training text from " << opts.srcfile << "...\n";
        dict_set_src.freq_cut(opts, opts.srcfile, opts.enc_word_vocab_size, opts.enc_char_vocab_size, opts.enc_feature_vocab_size);
        // trg
        dict_set_trg.set_id(opts);
    }

    void load(const s2s_options &opts){
        // src
        dict_set_src.init(opts.enc_feature_vocab_size.size());
        std::string file_prefix = opts.rootdir + "/" + opts.dict_prefix + "src_";
        dict_set_src.load(opts, file_prefix);
        // trg
        dict_set_trg.set_id(opts);
    }

    void save(const s2s_options &opts){
        // src
        std::string file_prefix = opts.rootdir + "/" + opts.dict_prefix + "src_";
        dict_set_src.save(opts, file_prefix);
    }

};

};

#endif

