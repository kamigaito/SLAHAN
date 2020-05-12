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
#include "s2s/corpus/utils.hpp"

#ifndef INCLUDE_GUARD_S2S_DICT_SET_HPP
#define INCLUDE_GUARD_S2S_DICT_SET_HPP

namespace s2s {

class dict_set_token {

public:

    dynet::Dict d_word;
    dynet::Dict d_char;
    std::vector<dynet::Dict> d_feat;

    unsigned int start_id_word;
    unsigned int end_id_word;
    unsigned int unk_id_word;
    unsigned int pad_id_word;

    unsigned int start_id_char;
    unsigned int end_id_char;
    unsigned int unk_id_char;
    unsigned int pad_id_char;

    std::vector<unsigned int> start_id_feat;
    std::vector<unsigned int> end_id_feat;
    std::vector<unsigned int> unk_id_feat;
    std::vector<unsigned int> pad_id_feat;

    std::vector<unsigned int> word_freq;
    std::vector<unsigned int> char_freq;
    std::vector<std::vector<unsigned int>> feat_freq;

    void init(const unsigned int feature_size){ // opts.enc_feature_vocab_size.size()
        /* resize vectors */
        d_feat.resize(feature_size);
        start_id_feat.resize(feature_size);
        end_id_feat.resize(feature_size);
        unk_id_feat.resize(feature_size);
        pad_id_feat.resize(feature_size);
    }

    /* set start and end of sentence id */
    void set_id(const s2s_options &opts){
        // word
        start_id_word = d_word.convert(opts.start_symbol);
        end_id_word = d_word.convert(opts.end_symbol);
        pad_id_word = d_word.convert(opts.pad_symbol);
        // char
        start_id_char = d_char.convert(opts.start_symbol);
        end_id_char = d_char.convert(opts.end_symbol);
        pad_id_char = d_char.convert(opts.pad_symbol);
        // feat
        for(unsigned int feat_id = 0; feat_id < d_feat.size(); feat_id++){
            start_id_feat[feat_id] = d_feat[feat_id].convert(opts.start_symbol);
            end_id_feat[feat_id] = d_feat[feat_id].convert(opts.end_symbol);
            pad_id_feat[feat_id] = d_feat[feat_id].convert(opts.pad_symbol);
        }
    }

    void set_unk_id(const s2s_options &opts){
        unk_id_word = d_word.convert(opts.unk_symbol);
        unk_id_char = d_char.convert(opts.unk_symbol);
        for(unsigned int feat_id = 0; feat_id < d_feat.size(); feat_id++){
            unk_id_feat[feat_id] = d_feat[feat_id].convert(opts.unk_symbol);
        }
    }

    std::string fname_d_word(const s2s_options &opts, std::string file_prefix){
        std::string file_name = file_prefix + "word.txt";
        return file_name;
    }

    std::string fname_d_char(const s2s_options &opts, std::string file_prefix){
        std::string file_name = file_prefix + "char.txt";
        return file_name;
    }

    std::string fname_d_feat(const s2s_options &opts, const unsigned int feat_id, std::string file_prefix){
        std::string file_name = file_prefix + "feat_" + std::to_string(feat_id) + ".txt";
        return file_name;
    }

    void load(const s2s_options &opts, std::string file_prefix){
        {
            std::string file_name = fname_d_word(opts, file_prefix);
            std::cerr << "Loading word dictionary from " << file_name << "...\n";
            std::ifstream in(file_name);
            boost::archive::text_iarchive ia(in);
            ia >> d_word;
            in.close();
        }
        {
            std::string file_name = fname_d_char(opts, file_prefix);
            std::cerr << "Loading character dictionary from " << file_name << "...\n";
            std::ifstream in(file_name);
            boost::archive::text_iarchive ia(in);
            ia >> d_char;
            in.close();
        }
        for(unsigned int feat_id=0; feat_id < d_feat.size(); feat_id++){
            // std::string file_name = opts.rootdir + "/" + opts.dict_prefix + "src_" + std::to_string(i) + ".txt";
            std::string file_name = fname_d_feat(opts, feat_id, file_prefix);
            std::cerr << "Loading feature dictionary from " << file_name << "...\n";
            std::ifstream in(file_name);
            boost::archive::text_iarchive ia(in);
            ia >> d_feat[feat_id];
            in.close();
        }
        set_id(opts);
        set_unk_id(opts);
    }

    void save(const s2s_options &opts, std::string file_prefix){
        {
            std::string file_name = fname_d_word(opts, file_prefix);
            std::cerr << "Saving word dictionary from " << file_name << "...\n";
            std::ofstream out(file_name);
            boost::archive::text_oarchive oa(out);
            oa << d_word;
            out.close();
        }
        {
            std::string file_name = fname_d_char(opts, file_prefix);
            std::cerr << "Saving character dictionary from " << file_name << "...\n";
            std::ofstream out(file_name);
            boost::archive::text_oarchive oa(out);
            oa << d_char;
            out.close();
        }
        for(unsigned int feat_id=0; feat_id < d_feat.size(); feat_id++){
            std::string file_name = fname_d_feat(opts, feat_id, file_prefix);
            std::cerr << "Saving feature dictionary to " << file_name << "...\n";
            std::ofstream out(file_name);
            boost::archive::text_oarchive oa(out);
            oa << d_feat[feat_id];
            out.close();
        }
    }

    void freq_cut(const s2s_options &opts, const std::string file_path, const int word_vocab_size, const int char_vocab_size, const std::vector<int>& feature_vocab_size){
        std::ifstream in(file_path);
        assert(in);
        set_id(opts);
        std::vector<std::map<std::string, unsigned int> > vec_str_freq;
        std::map<std::string, unsigned int>  str_char_freq;
        std::string line;
        // count frequencies
        while(getline(in, line)) {
            std::vector<std::string>  tokens;
            boost::algorithm::split_regex(tokens, line, boost::regex(" "));
            for(const std::string token : tokens){
                std::vector<std::string> col;
                boost::algorithm::split_regex(col, token, boost::regex("-\\|-"));
                assert(col.size() == d_feat.size() + 1);
                if(col.size() != vec_str_freq.size()){
                    vec_str_freq.resize(col.size());
                }
                for(unsigned int col_id = 0; col_id < col.size(); col_id++){
                    vec_str_freq[col_id][col.at(col_id)]++;
                }
                std::vector<std::string> chars = split_to_char(col.at(0));
                for(auto elem : chars){
                    str_char_freq[elem]++;
                }
            }
        }
        in.close();
        // cutting words and features
        for(unsigned int col_id = 0; col_id < d_feat.size() + 1; col_id++){
            std::vector<std::pair<std::string, unsigned int> > str_vec;
            for(auto& p1: vec_str_freq.at(col_id)){
                str_vec.push_back(std::pair<std::string, unsigned int>(p1.first, p1.second));
            }
            CompareString comp;
            sort(str_vec.begin(), str_vec.end(), comp);
            if(col_id > 0){
                for(auto& p1 : str_vec){
                    if(feature_vocab_size.at(col_id-1) >= 0 && d_feat[col_id-1].size() + 1 >= (unsigned int)(feature_vocab_size.at(col_id-1))){ // -1 for <UNK>
                        break;
                    }
                    d_feat[col_id-1].convert(p1.first);
                }
            }else{
                for(auto& p1 : str_vec){
                    if(word_vocab_size >= 0 && d_word.size() + 1 >= (unsigned int)(word_vocab_size)){ // -1 for <UNK>
                        break;
                    }
                    d_word.convert(p1.first);
                }
            }
        }
        // cutting chars
        {
            std::vector<std::pair<std::string, unsigned int> > str_vec;
            for(auto& p1: str_char_freq){
                str_vec.push_back(std::pair<std::string, unsigned int>(p1.first, p1.second));
            }
            CompareString comp;
            sort(str_vec.begin(), str_vec.end(), comp);
            for(auto& p1 : str_vec){
                if(char_vocab_size >= 0 && d_char.size() + 1 >= (unsigned int)(char_vocab_size)){ // -1 for <UNK>
                    break;
                }
                d_char.convert(p1.first);
            }
        }
        // fix the vocabulary sizes
        d_word.freeze();
        d_word.set_unk(opts.unk_symbol);
        d_char.freeze();
        d_char.set_unk(opts.unk_symbol);
        for(unsigned int feat_id = 0; feat_id < d_feat.size(); feat_id++){
            d_feat[feat_id].freeze(); // no new word types allowed
            d_feat[feat_id].set_unk(opts.unk_symbol);
        }
        set_unk_id(opts);
        // set freq
        word_freq.resize(d_word.size());
        for(auto& p1: vec_str_freq.at(0)){
            word_freq[d_word.convert(p1.first)] = p1.second;
        }
        char_freq.resize(d_char.size());
        for(auto& p1: str_char_freq){
            char_freq[d_char.convert(p1.first)] = p1.second;
        }
        feat_freq.resize(d_feat.size());
        for(unsigned int feat_id = 0; feat_id < d_feat.size(); feat_id++){
            feat_freq[feat_id].resize(d_feat.at(feat_id).size());
            for(auto& p1: vec_str_freq.at(feat_id + 1)){
                feat_freq[feat_id][d_feat[feat_id].convert(p1.first)] = p1.second;
            }
        }
    }
};
  
class dict_set_label {

public:

    dynet::Dict d_word;

    unsigned int start_id_word;
    unsigned int keep_id_word;
    unsigned int delete_id_word;
    unsigned int end_id_word;

    /* set start and end of sentence id */
    void set_id(const s2s_options &opts){
        // word
        start_id_word = d_word.convert(opts.start_symbol);
        keep_id_word = d_word.convert(opts.keep_symbol);
        delete_id_word = d_word.convert(opts.delete_symbol);
        end_id_word = d_word.convert(opts.end_symbol);
        d_word.freeze();
    }

};


};
#endif
