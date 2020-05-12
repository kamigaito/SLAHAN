#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options.hpp>
#include <boost/regex.hpp>
#include <boost/algorithm/string/regex.hpp>

#include "s2s/dynet/dict.h"
#include "s2s/corpus/comp.hpp"

#ifndef INCLUDE_GUARD_S2S_TOKEN_HPP
#define INCLUDE_GUARD_S2S_TOKEN_HPP

namespace s2s {

  class token {

    public:

    unsigned int word_id;
    std::vector<unsigned int> char_id_seq;
    std::vector<unsigned int> feat_id_list;

  };

};

#endif
