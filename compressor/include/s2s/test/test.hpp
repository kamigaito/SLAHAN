#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/rnn.h"
#include "dynet/gru.h"
#include "dynet/lstm.h"
#include "dynet/fast-lstm.h"
#include "dynet/expr.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <type_traits>
#include <time.h>

#include <boost/serialization/vector.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options.hpp>

#include "s2s/dynet/dict.h"
#include "s2s/corpus/corpora.hpp"
#include "s2s/corpus/comp.hpp"
#include "s2s/corpus/options.hpp"
#include "s2s/corpus/batch.hpp"
#include "s2s/test/corp.hpp"
#include "s2s/test/encdec.hpp"

#ifndef INCLUDE_GUARD_S2S_TEST_HPP
#define INCLUDE_GUARD_S2S_TEST_HPP

namespace s2s {

    template<class T_EncDec>
    void test(s2s_options &opts){
        dicts dicts;
        dicts.set(opts);
        parallel_corpus para_corp(opts, dicts);
        // corpora
        test_corp(opts, dicts, para_corp); 
        // encdec
        dynet::ParameterCollection model;
        std::unique_ptr<T_EncDec> encdec(new T_EncDec(model, opts, dicts));
        test_encdec(opts, dicts, para_corp, encdec); 
        std::cerr << "### All tests were passed. ###" << std::endl;
    }

};

#endif

