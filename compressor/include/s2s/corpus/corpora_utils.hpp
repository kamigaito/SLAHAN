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
#include "s2s/corpus/dict_set.hpp"
#include "s2s/corpus/token.hpp"

#ifndef INCLUDE_GUARD_S2T_CORPORA_UTILS_HPP
#define INCLUDE_GUARD_S2T_CORPORA_UTILS_HPP

namespace s2s {

    void load_corpus_src(const std::string file_path, const s2s_options& opts, dict_set_token& d_set, std::vector<std::vector<token> >& corpus){
        std::ifstream in(file_path);
        assert(in);
        int sid = 0;
        int ttoks = 0;
        std::string line;
        while(getline(in, line)) {
            std::vector<token> token_list;
            std::vector<std::string> str_tokens;
            boost::algorithm::split_regex(str_tokens, line, boost::regex(" "));
            ttoks += str_tokens.size();
            for(unsigned int token_id = 0; token_id < str_tokens.size(); token_id++) {
                std::vector<std::string> str_cols;
                boost::algorithm::split_regex(str_cols, str_tokens.at(token_id), boost::regex("-\\|-"));
                token tok;
                // word
                tok.word_id = d_set.d_word.convert(str_cols[0]);
                // feat
                for(unsigned int col_id = 1; col_id < str_cols.size(); col_id++){
                    unsigned int feat_id = col_id - 1;
                    tok.feat_id_list.emplace_back(d_set.d_feat[feat_id].convert(str_cols.at(col_id)));
                }
                // char
                if(opts.lookup_type == "char" || opts.lookup_type == "word+char"){
                    tok.char_id_seq.emplace_back(d_set.start_id_char);
                    for(unsigned int char_id = 0; char_id < str_cols.at(0).size(); char_id++){
                        tok.char_id_seq.emplace_back(d_set.d_char.convert(str_cols.at(0).substr(char_id, 1)));
                    }
                    tok.char_id_seq.emplace_back(d_set.end_id_char);
                }
                token_list.push_back(tok);
            }
            corpus.push_back(token_list);
/*
            for(unsigned int feature_id = 0; feature_id < start.size(); feature_id++){
                if (corpus_src.back().front().at(feature_id) != start.at(feature_id) && corpus_src.back().back().at(feature_id) != end.at(feature_id)) {
                    std::cerr << "Sentence in " << file_path << ":" << tlc << " didn't start or end with <s>, </s>\n";
                    abort();
                }
            }
*/
            sid++;
        }
        in.close();
        std::cerr << sid << " lines, " << ttoks << " tokens, " << d_set.d_word.size() << " types\n";
    }

    void load_corpus_trg(const std::string file_path, const s2s_options& opts, dict_set_label& d_set, std::vector<std::vector<token> >& corpus){
        std::ifstream in(file_path);
        assert(in);
        int sid = 0;
        int ttoks = 0;
        std::string line;
        while(getline(in, line)) {
            std::vector<token> token_list;
            std::vector<std::string> str_tokens;
            boost::algorithm::split_regex(str_tokens, line, boost::regex(" "));
            ttoks += str_tokens.size();
            for(unsigned int token_id = 0; token_id < str_tokens.size(); token_id++) {
                token tok;
                // word
                tok.word_id = d_set.d_word.convert(str_tokens[token_id]);
                token_list.push_back(tok);
            }
            corpus.push_back(token_list);
            sid++;
        }
        in.close();
        std::cerr << sid << " lines, " << ttoks << " tokens, " << d_set.d_word.size() << " types\n";
    }

    void load_align_corpus(const std::string file_path, const s2s_options& opts, std::vector<std::vector<unsigned int> >& corpus){
        std::ifstream in(file_path);
        assert(in);
        int sid = 0;
        int ttoks = 0;
        std::string line = "";
        while(getline(in, line)) {
            std::vector<std::string> tokens;
            boost::algorithm::split_regex(tokens, line, boost::regex(" "));
            std::vector<unsigned int> aligns(tokens.size());
            unsigned int tok_id = 0;
            for(const std::string token : tokens){
                std::vector<std::string> w2w;
                boost::algorithm::split_regex(w2w, token, boost::regex("-"));
                aligns[stoi(w2w.at(1))] = stoi(w2w.at(0));
                tok_id++;
            }
            corpus.push_back(aligns);
            ttoks += tokens.size();
            sid++;
        }
        in.close();
        std::cerr << sid << " lines, " << ttoks << " tokens\n";
    }

};

#endif
