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
#include "s2s/corpus/corpora_utils.hpp"
#include "s2s/corpus/batch.hpp"
#include "s2s/corpus/dicts.hpp"

#ifndef INCLUDE_GUARD_S2S_CORPUS_HPP
#define INCLUDE_GUARD_S2S_CORPUS_HPP

namespace s2s {

class monoling_corpus {

    public:

    std::vector<std::vector<token > > src;
    std::vector<std::vector<unsigned int> > align;
    unsigned int index;
    std::vector<unsigned int> sents_order;
    std::vector<std::pair<unsigned int, unsigned int> > batch_order;

    const s2s_options& opts;
    dicts& d;

    monoling_corpus(const s2s_options& opts, dicts& d) : opts(opts),d(d) {
        index = 0;
    }

    void load_src(const std::string srcfile){
        load_corpus_src(srcfile, opts, d.dict_set_src, src);
        sents_order.resize(src.size());
        std::iota(sents_order.begin(), sents_order.end(), 0);
    }

    void load_align(const std::string alignfile){
        load_align_corpus(alignfile, opts, align);
    }

    void load_check_with_align(){
        assert(src.size() == align.size()); // sentence size does not match!
        // check
        for(unsigned int sid = 0; sid < src.size(); sid++){
            assert(src.at(sid).size() == align.at(sid).size()); // alignment size
            for(const unsigned int tok : align.at(sid)){
                assert(0 <= tok && tok < src.at(sid).size()); // alignment range
            }
        }
    }

    bool next_mono_batch(batch& mono_batch){
        if(index < batch_order.size()){
            mono_batch.set(sents_order, batch_order.at(index).first, batch_order.at(index).second, src, align, d);
            index++;
            return true;
        }
        return false;
    }

    void reset_index(){
        index = 0;
    }

    void sort_mono_sent(const std::string shuffle_type){
        if(shuffle_type == "sort_default"){
            std::vector<std::pair<unsigned int, std::pair<unsigned int, unsigned int > > > vec_len(src.size());
            for(unsigned int sid = 0; sid < src.size(); sid++){
                vec_len[sid].first = sid;
                vec_len[sid].second.first = src.at(sid).size();
            }
            CompareLength comp_len;
            sort(vec_len.begin(), vec_len.end(), comp_len);
            for(unsigned int sid = 0; sid < src.size(); sid++){
                sents_order[sid] = vec_len.at(sid).first;
            }
        }else if(shuffle_type == "default"){
        }else{
            std::cerr << "shuffle_type does not match." << std::endl;
            assert(false);
        }
    }

    void set_mono_batch_order(const unsigned int max_batch_size, const unsigned int src_tok_lim, const std::string batch_type){
        batch_order.clear();
        if(batch_type == "default"){
            unsigned int batch_start = 0, batch_size = 0, src_tok = 0;
            for(unsigned int sid = 0; sid < sents_order.size(); sid++){
                unsigned int cur_len_src = src.at(sents_order.at(sid)).size();
                if(batch_size + 1 <= max_batch_size && src_tok + cur_len_src <= src_tok_lim){
                    src_tok += cur_len_src;
                    batch_size++;
                }else{
                    batch_order.push_back(std::pair<unsigned int, unsigned int>(batch_start, batch_size));
                    batch_start = sid;
                    src_tok = cur_len_src;
                    batch_size = 1;
                }
                if(sid == sents_order.size() - 1){
                    batch_order.push_back(std::pair<unsigned int, unsigned int>(batch_start, batch_size));
                }
            }
        }else if(batch_type == "same_length"){
            unsigned int batch_start = 0, batch_size = 0, src_tok = 0;
            unsigned int cur_len = src.at(sents_order.at(0)).size();
            for(unsigned int sid = 0; sid < sents_order.size(); sid++){
                unsigned int cur_len_src = src.at(sents_order.at(sid)).size();
                if(cur_len_src == cur_len && batch_size + 1 <= max_batch_size && src_tok + cur_len_src <= src_tok_lim){
                    src_tok += cur_len_src;
                    batch_size++;
                }else{
                    batch_order.push_back(std::pair<unsigned int, unsigned int>(batch_start, batch_size));
                    batch_start = sid;
                    cur_len = cur_len_src;
                    src_tok = cur_len_src;
                    batch_size = 1;
                }
                if(sid == sents_order.size() - 1){
                    batch_order.push_back(std::pair<unsigned int, unsigned int>(batch_start, batch_size));
                }
            }
        }else{
            std::cerr << "batch_type does not match." << std::endl;
            assert(false);
        }
    }

};

class parallel_corpus : public monoling_corpus {

    public:

    std::vector<std::vector<token> > trg;

    parallel_corpus(const s2s_options& opts, dicts& d) : monoling_corpus(opts, d) {}

    void load_trg(const std::string trgfile){
        load_corpus_trg(trgfile, opts, d.dict_set_trg, trg);
    }

    void load_check(){
        // check
        assert(src.size() == trg.size()); // sentence size does not match!
    }

    void sort_para_sent(const std::string shuffle_type, const unsigned int max_batch_size, const unsigned int src_tok_lim, const unsigned int trg_tok_lim){
        if(shuffle_type == "random"){
            srand(unsigned(time(NULL)));
            std::random_shuffle(sents_order.begin(), sents_order.end());
        }else if(shuffle_type == "sort_default"){
            std::vector<std::pair<unsigned int, std::pair<unsigned int, unsigned int > > > vec_len(src.size());
            for(unsigned int sid = 0; sid < src.size(); sid++){
                vec_len[sid].first = sid;
                vec_len[sid].second.first = src.at(sid).size();
                vec_len[sid].second.second = trg.at(sid).size();
            }
            CompareLength comp_len;
            sort(vec_len.begin(), vec_len.end(), comp_len);
            for(unsigned int sid = 0; sid < src.size(); sid++){
                sents_order[sid] = vec_len.at(sid).first;
            }
        }else if(shuffle_type == "sort_random"){
            std::vector<unsigned int> sents_order_local(src.size());
            std::vector<std::pair<unsigned int, std::pair<unsigned int, unsigned int > > > vec_len(src.size());
            for(unsigned int sid = 0; sid < src.size(); sid++){
                vec_len[sid].first = sid;
                vec_len[sid].second.first = src.at(sid).size();
                vec_len[sid].second.second = trg.at(sid).size();
            }
            CompareLength comp_len;
            sort(vec_len.begin(), vec_len.end(), comp_len);
            for(unsigned int sid = 0; sid < src.size(); sid++){
                sents_order_local[sid] = vec_len.at(sid).first;
            }
            sents_order.clear();
            std::vector<unsigned int> vec_sents;
            std::pair<unsigned int, unsigned int> cur_len(src.at(sents_order_local.at(0)).size(), trg.at(sents_order_local.at(0)).size());
            for(unsigned int sid = 0; sid < sents_order_local.size(); sid++){
                unsigned int cur_len_src = src.at(sents_order_local.at(sid)).size();
                unsigned int cur_len_trg = trg.at(sents_order_local.at(sid)).size();
                if(cur_len_src == cur_len.first && cur_len_trg == cur_len.second){
                    vec_sents.push_back(sents_order_local.at(sid));
                }else{
                    cur_len.first = cur_len_src;
                    cur_len.second = cur_len_trg;
                    srand(unsigned(time(NULL)));
                    std::random_shuffle(vec_sents.begin(), vec_sents.end());
                    sents_order.insert(sents_order.end(), vec_sents.begin(), vec_sents.end());
                    vec_sents.clear();
                    vec_sents.push_back(sents_order_local.at(sid));
                }
                if(sid == sents_order_local.size() - 1){
                    srand(unsigned(time(NULL)));
                    std::random_shuffle(vec_sents.begin(), vec_sents.end());
                    sents_order.insert(sents_order.end(), vec_sents.begin(), vec_sents.end());
                }
            }
        }else if(shuffle_type == "default"){
        }else{
            std::cerr << "shuffle_type does not match." << std::endl;
            assert(false);
        }
    }

    void shuffle_batch(const std::string shuffle_type){
        if(shuffle_type == "random"){
            srand(unsigned(time(NULL)));
            std::random_shuffle(batch_order.begin(), batch_order.end());
        }else if(shuffle_type == "default"){
        }else{
            std::cerr << "shuffle_type does not match." << std::endl;
            assert(false);
        }
    }

    bool next_batch_para(batch& para_batch){
        if(index < batch_order.size()){
            para_batch.set(sents_order, batch_order.at(index).first, batch_order.at(index).second, src, trg, align, d);
            index++;
            return true;
        }
        return false;
    }

    void set_para_batch_order(const unsigned int max_batch_size, const unsigned int src_tok_lim, const unsigned int trg_tok_lim, const std::string batch_type){
        batch_order.clear();
        if(batch_type == "default"){
            unsigned int batch_start = 0;
            unsigned int batch_size = 0;
            unsigned int src_tok = 0;
            unsigned int trg_tok = 0;
            for(unsigned int sid = 0; sid < sents_order.size(); sid++){
                unsigned int cur_len_src = src.at(sents_order.at(sid)).size();
                unsigned int cur_len_trg = trg.at(sents_order.at(sid)).size();
                if(batch_size + 1 <= max_batch_size && src_tok + cur_len_src <= src_tok_lim && trg_tok + cur_len_trg <= trg_tok_lim){
                    src_tok += cur_len_src;
                    trg_tok += cur_len_trg;
                    batch_size++;
                }else{
                    batch_order.push_back(std::pair<unsigned int, unsigned int>(batch_start, batch_size));
                    batch_start = sid;
                    src_tok = cur_len_src;
                    trg_tok = cur_len_trg;
                    batch_size = 1;
                }
                if(sid == sents_order.size() - 1){
                    batch_order.push_back(std::pair<unsigned int, unsigned int>(batch_start, batch_size));
                }
            }
        }else if(batch_type == "same_length"){
            unsigned int batch_start = 0;
            unsigned int batch_size = 0;
            unsigned int src_tok = 0;
            unsigned int trg_tok = 0;
            std::pair<unsigned int, unsigned int> cur_len(src.at(sents_order.at(0)).size(), trg.at(sents_order.at(0)).size());
            for(unsigned int sid = 0; sid < sents_order.size(); sid++){
                unsigned int cur_len_src = src.at(sents_order.at(sid)).size();
                unsigned int cur_len_trg = trg.at(sents_order.at(sid)).size();
                if(cur_len_src == cur_len.first && cur_len_trg == cur_len.second && batch_size + 1 <= max_batch_size && src_tok + cur_len_src <= src_tok_lim && trg_tok + cur_len_trg <= trg_tok_lim){
                    src_tok += cur_len_src;
                    trg_tok += cur_len_trg;
                    batch_size++;
                }else{
                    batch_order.push_back(std::pair<unsigned int, unsigned int>(batch_start, batch_size));
                    batch_start = sid;
                    cur_len.first = cur_len_src;
                    cur_len.second = cur_len_trg;
                    src_tok = cur_len_src;
                    trg_tok = cur_len_trg;
                    batch_size = 1;
                }
                if(sid == sents_order.size() - 1){
                    batch_order.push_back(std::pair<unsigned int, unsigned int>(batch_start, batch_size));
                }
            }
        }else{
            std::cerr << "batch_type does not match." << std::endl;
            assert(false);
        }
    }

};

};

#endif
