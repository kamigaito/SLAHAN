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

#ifndef INCLUDE_GUARD_S2S_TEST_CORP_HPP
#define INCLUDE_GUARD_S2S_TEST_CORP_HPP

namespace s2s {

    void test_batch_random_sub(s2s_options &opts, dicts &dicts, const unsigned int alpha, unsigned int &cnt_unk, unsigned int &cnt_not_unk, parallel_corpus &para_corp){
        opts.drop_word_alpha = alpha;
        para_corp.reset_index();
        para_corp.set_para_batch_order(opts.max_batch_train, opts.src_tok_lim_train, opts.trg_tok_lim_train, opts.batch_type_train);
        batch batch_local(opts, dicts);
        unsigned int batch_cnt = 0;
        while(para_corp.next_batch_para(batch_local)){
            // word drop test
            batch_local.drop_word();
            // src feat
            for(unsigned int tid = 0; tid < batch_local.sent_word.size(); tid++){
                for(unsigned int sid = 0; sid < batch_local.sent_id.size(); sid++){
                    if(batch_local.sent_word.at(tid).at(sid) == dicts.dict_set_src.unk_id_word){
                        cnt_unk++;
                    }else{
                        cnt_not_unk++;
                    }
                }
            }
            batch_cnt++;
        }
        assert(batch_cnt > 0);
        std::cerr << "All: " << (cnt_unk + cnt_not_unk) << ", " << "Unk: " << cnt_unk << std::endl;
        para_corp.reset_index();
    }

    void test_batch_random(s2s_options &opts, dicts &dicts, parallel_corpus &para_corp){
        // alpha = 0.0
        {
            unsigned int cnt_unk = 0;
            unsigned int cnt_not_unk = 0;
            test_batch_random_sub(opts, dicts, 0.f, cnt_unk, cnt_not_unk, para_corp);
            assert(cnt_unk == 0);
            assert(cnt_not_unk > 0);
        }
        // alpha = 10000.0
        {
            unsigned int cnt_unk = 0;
            unsigned int cnt_not_unk = 0;
            test_batch_random_sub(opts, dicts, 10000.f, cnt_unk, cnt_not_unk, para_corp);
            assert(cnt_unk > 0);
        }
        // alpha = 1.0
        {
            unsigned int cnt_unk = 0;
            unsigned int cnt_not_unk = 0;
            test_batch_random_sub(opts, dicts, 1.f, cnt_unk, cnt_not_unk, para_corp);
            assert(cnt_unk > 0);
            assert(cnt_not_unk > 0);
        }
    }

    void test_corp_load_files(s2s_options &opts, dicts &dicts, parallel_corpus &para_corp){
        // load test
        para_corp.load_src(opts.srcfile);
        para_corp.load_trg(opts.trgfile);
        para_corp.load_check();
        if(opts.guided_alignment == true){
            para_corp.load_align(opts.alignfile);
            para_corp.load_check_with_align();
        }

        std::ifstream in_src(opts.srcfile);
        assert(in_src);
        std::ifstream in_trg(opts.trgfile);
        assert(in_trg);
        std::ifstream in_align(opts.alignfile);
        if(opts.guided_alignment == true){
            assert(in_align);
        }
        std::string line_src = "";
        std::string line_trg = "";
        std::string line_align = "";
        int sid = 0;
        while(getline(in_src, line_src) && getline(in_trg, line_trg) && (opts.guided_alignment == false || getline(in_align, line_align))) {
            // src
            {
                std::vector<std::string> str_tokens;
                boost::algorithm::split_regex(str_tokens, line_src, boost::regex(" "));
                std::vector<token > &sent = para_corp.src.at(sid);
                assert(sent.size() == str_tokens.size());
                for(unsigned int tok_id = 0; tok_id < sent.size(); tok_id++){
                    // word
                    std::string str_conv_word = dicts.dict_set_src.d_word.convert(sent.at(tok_id).word_id);
                    // feat
                    std::string str_conv = str_conv_word;
                    bool is_feat_unk = false;
                    for(unsigned int feat_type = 0; feat_type < sent.at(tok_id).feat_id_list.size(); feat_type++){
                        str_conv += "-|-";
                        std::string str_conv_local = dicts.dict_set_src.d_feat[feat_type].convert(sent.at(tok_id).feat_id_list.at(feat_type));
                        str_conv += str_conv_local;
                        if(str_conv_local == opts.unk_symbol){
                           is_feat_unk = true; 
                        }
                    }
                    if(str_conv_word != opts.unk_symbol && is_feat_unk == false && str_conv != str_tokens.at(tok_id)){
                        std::cerr << str_conv << " " << str_tokens.at(tok_id) << std::endl;
                        assert(false);
                    }
                    // char
                    std::string str_char = "";
                    bool is_char_unk = false;
                    for(unsigned int char_id : sent.at(tok_id).char_id_seq){
                        if(char_id != dicts.dict_set_src.start_id_char && char_id != dicts.dict_set_src.end_id_char){
                            std::string str_char_local = dicts.dict_set_src.d_char.convert(char_id);
                            str_char += str_char_local;
                            if(str_char_local == opts.unk_symbol){
                                is_char_unk = true; 
                            }
                        }
                    }
                    if(str_conv_word != opts.unk_symbol && is_char_unk == false && dicts.dict_set_src.d_word.convert(sent.at(tok_id).word_id) != str_char){
                        std::cerr << dicts.dict_set_src.d_word.convert(sent.at(tok_id).word_id) << " " << str_char << std::endl;
                        assert(false);
                    }
                }
            }
            // trg
            {
                std::vector<std::string> str_tokens;
                boost::algorithm::split_regex(str_tokens, line_trg, boost::regex(" "));
                std::vector<token > &sent = para_corp.trg.at(sid);
                assert(sent.size() == str_tokens.size());
                for(unsigned int tok_id = 0; tok_id < sent.size(); tok_id++){
                    // word
                    std::string str_conv = dicts.dict_set_trg.d_word.convert(sent.at(tok_id).word_id);
                    if(str_conv != opts.unk_symbol && str_conv != str_tokens.at(tok_id)){
                        std::cerr << str_conv << " " << str_tokens.at(tok_id) << std::endl;
                        assert(false);
                    }
                }
            }
            // align
            if(opts.guided_alignment == true){
                std::vector<std::string> str_tokens;
                boost::algorithm::split_regex(str_tokens, line_align, boost::regex(" "));
                std::vector<unsigned int> aligns(str_tokens.size());
                std::vector<unsigned int > &sent = para_corp.align.at(sid);
                for(unsigned int tok_id=0; tok_id < sent.size(); tok_id++){
                    std::string str_link = std::to_string(sent[tok_id]) + "-" + std::to_string(tok_id);
                    if(str_tokens[tok_id] != str_link){
                        std::cerr << str_tokens[tok_id] << ", " << str_link << std::endl;
                        assert(false);
                    }
                }
            }
            sid++;
        }
        in_src.close();
        in_trg.close();
        in_align.close();
        assert(sid > 0);
    }
    
    void test_corp_to_batch(s2s_options &opts, dicts &dicts, parallel_corpus &para_corp){
        // batch test
        unsigned int batch_cnt = 0;
        batch batch_local(opts, dicts);
        para_corp.reset_index();
        para_corp.set_para_batch_order(opts.max_batch_train, opts.src_tok_lim_train, opts.trg_tok_lim_train, opts.batch_type_train);
        while(para_corp.next_batch_para(batch_local)){
            // src word
            for(unsigned int tid = 0; tid < batch_local.sent_word.size(); tid++){
                unsigned int sid = 0;
                for(const unsigned int sent_id : batch_local.sent_id){
                    if(tid >= para_corp.src.at(sent_id).size()){
                        if(batch_local.sent_word.at(tid).at(sid) != dicts.dict_set_src.end_id_word){
                            std::cerr << batch_local.sent_word.at(tid).at(sid) << ", " << dicts.dict_set_src.end_id_word << std::endl;
                            assert(false);
                        }
                    }else{
                        if(batch_local.sent_word.at(tid).at(sid) != para_corp.src.at(sent_id).at(tid).word_id){
                            std::cerr << batch_local.sent_word.at(tid).at(sid) << ", " << para_corp.src.at(sent_id).at(tid).word_id << std::endl;
                            assert(false);
                        }
                    }
                    sid++;
                }
            }
            // src feat
            for(unsigned int tid = 0; tid < batch_local.sent_feat.size(); tid++){
                for(unsigned int fid = 0; fid < dicts.dict_set_src.d_feat.size(); fid++){
                    unsigned int sid = 0;
                    for(const unsigned int sent_id : batch_local.sent_id){
                        if(tid >= para_corp.src.at(sent_id).size()){
                            if(batch_local.sent_feat.at(fid).at(tid).at(sid) != dicts.dict_set_src.end_id_feat[fid]){
                                std::cerr << dicts.dict_set_src.d_feat[fid].convert(batch_local.sent_feat.at(fid).at(tid).at(sid)) << ", " << dicts.dict_set_src.d_feat[fid].convert(dicts.dict_set_src.end_id_feat[fid]) << std::endl;
                                assert(false);
                            }
                        }else{
                            if(batch_local.sent_feat.at(fid).at(tid).at(sid) != para_corp.src.at(sent_id).at(tid).feat_id_list.at(fid)){
                                std::cerr << dicts.dict_set_src.d_feat[fid].convert(batch_local.sent_feat.at(fid).at(tid).at(sid)) << ", " << dicts.dict_set_src.d_feat[fid].convert(para_corp.src.at(sent_id).at(tid).feat_id_list.at(fid)) << std::endl;
                                assert(false);
                            }
                        }
                        sid++;
                    }
                }
            }
            // src char
            for(unsigned int tid = 0; tid < batch_local.sent_char.size(); tid++){
                for(unsigned int cid = 0; cid < batch_local.sent_char.at(tid).size(); cid++){
                    unsigned int sid = 0;
                    for(const unsigned int sent_id : batch_local.sent_id){
                        if(tid >= para_corp.src.at(sent_id).size()){
                            if(batch_local.sent_char.at(tid).at(cid).at(sid) != dicts.dict_set_src.end_id_char){
                                std::cerr << batch_local.sent_char.at(tid).at(cid).at(sid) << ", " << dicts.dict_set_src.end_id_char << std::endl;
                                assert(false);
                            }
                        }else{
                            if(tid > 0 && tid < para_corp.src.at(sent_id).size() - 1 && batch_local.sent_char.at(tid).at(cid).at(sid) != dicts.dict_set_src.pad_id_char && batch_local.sent_char.at(tid).at(cid).at(sid) != para_corp.src.at(sent_id).at(tid).char_id_seq.at(cid)){
                                std::cerr << dicts.dict_set_src.d_char.convert(batch_local.sent_char.at(tid).at(cid).at(sid)) << ", " << dicts.dict_set_src.d_char.convert(para_corp.src.at(sent_id).at(tid).char_id_seq.at(cid)) << std::endl;
                                assert(false);
                            }
                        }
                        sid++;
                    }
                }
            }
            // trg word
            for(unsigned int tid = 0; tid < batch_local.sent_label.size(); tid++){
                unsigned int sid = 0;
                for(const unsigned int sent_id : batch_local.sent_id){
                    if(tid >= para_corp.trg.at(sent_id).size()){
                        if(batch_local.sent_label.at(tid).at(sid) != dicts.dict_set_trg.end_id_word){
                            std::cerr << batch_local.sent_label.at(tid).at(sid) << ", " << dicts.dict_set_trg.end_id_word << std::endl;
                            assert(false);
                        }
                    }else{
                        if(batch_local.sent_label.at(tid).at(sid) != para_corp.trg.at(sent_id).at(tid).word_id){
                            std::cerr << batch_local.sent_label.at(tid).at(sid) << ", " << para_corp.trg.at(sent_id).at(tid).word_id << std::endl;
                            assert(false);
                        }
                    }
                    sid++;
                }
            }
            batch_cnt++;
        }
        assert(batch_cnt > 0);
    }

    void test_corp_sort(s2s_options &opts, dicts &dicts, parallel_corpus &para_corp){
        // sort test
        para_corp.sort_para_sent(opts.sort_sent_type_train, opts.max_batch_train, opts.src_tok_lim_train, opts.trg_tok_lim_train);
        para_corp.set_para_batch_order(opts.max_batch_train, opts.src_tok_lim_train, opts.trg_tok_lim_train, opts.batch_type_train);
        para_corp.shuffle_batch(opts.shuffle_batch_type_train);
    }

    void test_corp(s2s_options &opts, dicts &dicts, parallel_corpus &para_corp){
        // file loading
        test_corp_load_files(opts, dicts, para_corp);
        // corpus to batch
        test_corp_to_batch(opts, dicts, para_corp);
        // word dropout
        test_batch_random(opts, dicts, para_corp);
        // sort sentences
        test_corp_sort(opts, dicts, para_corp);
    }

};

#endif
