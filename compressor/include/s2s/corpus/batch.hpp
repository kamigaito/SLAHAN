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
#include "s2s/corpus/dicts.hpp"
#include "s2s/corpus/elmo.hpp"

#ifndef INCLUDE_GUARD_S2S_MANUAL_BATCH_HPP
#define INCLUDE_GUARD_S2S_MANUAL_BATCH_HPP

namespace s2s {

    class batch {

    public:

        std::vector<unsigned int> sent_id;
        // src
        std::vector<std::vector<unsigned int> > sent_word;
        std::vector<std::vector<std::vector<unsigned int> > > sent_char;
        std::vector<std::vector<std::vector<unsigned int> > > sent_feat;
        // elmo_type, token_id, sentence_id, layer_id, float_vector
        std::vector<std::vector<std::vector<std::vector<std::vector<float> > > > > sent_elmo;
        // trg
        std::vector<std::vector<unsigned int> > sent_label;
        // align
        std::vector<std::vector<unsigned int> > align;
        std::vector<unsigned int> len_src;
        std::vector<unsigned int> len_trg;
        // ELMo style vectors
        std::vector<ELMo> elmos;
        // random
        std::random_device rd;
        std::mt19937 mt;
        // const
        const s2s_options& opts;
        dicts& d;

        batch(const s2s_options& opts, dicts& d, const std::vector<std::string> &elmo_hdf5_files) : rd(), mt(rd()), opts(opts), d(d) {
            if(opts.elmo_hdf5_files.size() > 0){
                for(unsigned int e_id = 0; e_id < elmo_hdf5_files.size(); e_id++){
                    std::cout << "Loading word vectors from " << elmo_hdf5_files[e_id] << std::endl;
                    elmos.push_back(ELMo(elmo_hdf5_files[e_id]));
                }
            }
        }

        batch(const s2s_options& opts, dicts& d) : rd(), mt(rd()), opts(opts), d(d) {
            if(opts.elmo_hdf5_files.size() > 0){
                for(unsigned int e_id = 0; e_id < opts.elmo_hdf5_files.size(); e_id++){
                    std::cout << "Loading word vectors from " << opts.elmo_hdf5_files[e_id] << std::endl;
                    elmos.push_back(ELMo(opts.elmo_hdf5_files[e_id]));
                }
            }
        }

        void set(
            const std::vector<unsigned int> sents_order,
            const unsigned int index,
            const unsigned int batch_size_local,
            const std::vector<std::vector<token > > &src_input,
            const std::vector<std::vector<token > > &trg_input,
            const std::vector<std::vector<unsigned int> > &align_input,
            dicts& d
        ){
            word2batch(index, batch_size_local, sents_order, src_input);
            feat2batch(index, batch_size_local, sents_order, src_input);
            if(opts.lookup_type == "char" || opts.lookup_type == "word+char"){
                char2batch(index, batch_size_local, sents_order, src_input);
            }
            if(opts.elmo_hdf5_dims.size() > 0){
                elmo2batch(index, batch_size_local, sents_order, src_input);
            }
            label2batch(index, batch_size_local, sents_order, trg_input);
            align = align2batch(sents_order, index, batch_size_local, align_input);
            len_src = src2len(sents_order, index, batch_size_local, src_input);
            len_trg = trg2len(sents_order, index, batch_size_local, trg_input);
            sent_id.clear();
            for(unsigned int sid = 0; sid < batch_size_local; sid++){
                sent_id.push_back(sents_order.at(sid + index));
            }
        }

        void set(
            const std::vector<unsigned int> sents_order,
            const unsigned int index,
            const unsigned int batch_size_local,
            const std::vector<std::vector<token > > &src_input,
            const std::vector<std::vector<unsigned int> > &align_input,
            dicts& d
        ){
            word2batch(index, batch_size_local, sents_order, src_input);
            feat2batch(index, batch_size_local, sents_order, src_input);
            if(opts.lookup_type == "char" || opts.lookup_type == "word+char"){
                char2batch(index, batch_size_local, sents_order, src_input);
            }
            if(opts.elmo_hdf5_dims.size() > 0){
                elmo2batch(index, batch_size_local, sents_order, src_input);
            }
            align = align2batch(sents_order, index, batch_size_local, align_input);
            len_src = src2len(sents_order, index, batch_size_local, src_input);
            sent_id.clear();
            for(unsigned int sid = 0; sid < batch_size_local; sid++){
                sent_id.push_back(sents_order.at(sid + index));
            }
        }

        unsigned int batch_size() const {
            return sent_word.at(0).size();
        }

        // transpose word matrix
        void word2batch(const unsigned int index, const unsigned int batch_size_local, const std::vector<unsigned int>& sents_order, const std::vector<std::vector<token> >& vec_input){
            // max token size
            unsigned int max_len = 0;
            for(unsigned int sid = 0; sid < batch_size_local; sid++){
                unsigned int cur_len = vec_input.at(sents_order.at(sid + index)).size();
                if(cur_len > max_len){
                    max_len = cur_len;
                }
            }
            sent_word.resize(max_len);
            for(unsigned int pos = 0; pos < max_len; pos++){
                sent_word[pos].resize(batch_size_local);
                for(unsigned int sid = 0; sid < batch_size_local; sid++){
                    if(pos < vec_input.at(sents_order.at(sid + index)).size()){
                        sent_word[pos][sid] = vec_input.at(sents_order.at(sid + index)).at(pos).word_id;
                    }else{
                        sent_word[pos][sid] = d.dict_set_src.end_id_word;
                    }
                }
            }
        }

        // transpose character matrix
        void char2batch(const unsigned int index, const unsigned int batch_size_local, const std::vector<unsigned int>& sents_order, const std::vector<std::vector<token> >& vec_input){
            // calculate max token size
            unsigned int max_len = 0;
            for(unsigned int sid = 0; sid < batch_size_local; sid++){
                unsigned int cur_len = vec_input.at(sents_order.at(sid + index)).size();
                if(cur_len > max_len){
                    max_len = cur_len;
                }
            }
            // calculate max character size
            sent_char.resize(max_len);
            std::vector<unsigned int> vec_max_len(max_len, 0);
            for(unsigned int pos = 0; pos < max_len; pos++){
                sent_char[pos].resize(batch_size_local);
                for(unsigned int sid = 0; sid < batch_size_local; sid++){
                    if(pos < vec_input.at(sents_order.at(sid + index)).size()){
                        unsigned int cur_size = vec_input.at(sents_order.at(sid + index)).at(pos).char_id_seq.size();
                        if(cur_size > vec_max_len.at(pos)){
                            vec_max_len[pos] = cur_size;
                        }
                    }
                }
            }
            for(unsigned int tok_pos = 0; tok_pos < max_len; tok_pos++){
                sent_char[tok_pos].resize(vec_max_len.at(tok_pos));
                for(unsigned int char_pos = 0; char_pos < vec_max_len.at(tok_pos); char_pos++){
                    sent_char[tok_pos][char_pos].resize(batch_size_local);
                    for(unsigned int sid = 0; sid < batch_size_local; sid++){
                        if(tok_pos == 0){
                            sent_char[tok_pos][char_pos][sid] = d.dict_set_src.start_id_char;
                        }else if(0 < tok_pos && tok_pos < vec_input.at(sents_order.at(sid + index)).size() - 1){
                            if(char_pos < vec_input.at(sents_order.at(sid + index)).at(tok_pos).char_id_seq.size()){
                                sent_char[tok_pos][char_pos][sid] = vec_input.at(sents_order.at(sid + index)).at(tok_pos).char_id_seq.at(char_pos);
                            }else{
                                sent_char[tok_pos][char_pos][sid] = d.dict_set_src.pad_id_char;
                            }
                        }else{
                            sent_char[tok_pos][char_pos][sid] = d.dict_set_src.end_id_char;
                        }
                    }
                }
            }
        }

        // transpose feature matrix
        void feat2batch(const unsigned int index, const unsigned int batch_size_local, const std::vector<unsigned int>& sents_order, const std::vector<std::vector<token> >& vec_input){
            // max token size
            unsigned int max_len = 0;
            for(unsigned int sid = 0; sid < batch_size_local; sid++){
                unsigned int cur_len = vec_input.at(sents_order.at(sid + index)).size();
                if(cur_len > max_len){
                    max_len = cur_len;
                }
            }
            sent_feat.resize(d.dict_set_src.d_feat.size());
            for(unsigned int feat_id = 0; feat_id < d.dict_set_src.d_feat.size(); feat_id++){
                sent_feat[feat_id].resize(max_len);
                for(unsigned int tok_pos = 0; tok_pos < max_len; tok_pos++){
                    sent_feat[feat_id][tok_pos].resize(batch_size_local);
                    for(unsigned int sid = 0; sid < batch_size_local; sid++){
                        if(tok_pos < vec_input.at(sents_order.at(sid + index)).size()){
                            sent_feat[feat_id][tok_pos][sid] = vec_input.at(sents_order.at(sid + index)).at(tok_pos).feat_id_list.at(feat_id);
                        }else{
                            sent_feat[feat_id][tok_pos][sid] = d.dict_set_src.end_id_feat.at(feat_id);
                        }
                    }
                }
            }
        }

        // transpose elmo matrix
        void elmo2batch(const unsigned int index, const unsigned int batch_size_local, const std::vector<unsigned int>& sents_order, const std::vector<std::vector<token> >& vec_input){
            // max token size
            unsigned int max_len = 0;
            for(unsigned int sid = 0; sid < batch_size_local; sid++){
                unsigned int cur_len = vec_input.at(sents_order.at(sid + index)).size();
                if(cur_len > max_len){
                    max_len = cur_len;
                }
            }
            // extract elmo vectors
            // elmo_type, sentence_id, layer_id, token_id, float_vector
            std::vector<std::vector<std::vector<std::vector<std::vector<float> > > > > elmo_vectors(elmos.size());
            std::vector<std::vector<unsigned int> > elmo_lens(elmos.size());
            for(unsigned int eid = 0; eid < elmos.size(); eid++){
                elmo_vectors[eid].resize(batch_size_local);
                for(unsigned int sid = 0; sid < batch_size_local; sid++){
                    unsigned int sen_len = vec_input.at(sents_order.at(sid + index)).size();
                    elmos[eid].get(sents_order.at(sid + index), elmo_vectors[eid][sid]);
                }
            }
            // convert elmo vectors into a batch
            // elmo_type, token_id, sentence_id, layer_id, float_vector
            sent_elmo.resize(elmos.size());
            // elmo type
            for(unsigned int eid=0; eid < elmos.size(); eid++){
                // token_id
                sent_elmo[eid].resize(max_len);
                for(unsigned int tid=0; tid < max_len; tid++){
                    // sentence_id
                    sent_elmo[eid][tid].resize(batch_size_local);
                    for(unsigned int sid=0; sid < batch_size_local; sid++){
                        unsigned int slen = vec_input.at(sents_order.at(sid + index)).size();
                        // layer_id
                        sent_elmo[eid][tid][sid].resize(elmo_vectors[eid][sid].size());
                        for(unsigned int lid=0; lid < opts.elmo_hdf5_layers[eid]; lid++){
                            unsigned int elen = elmo_vectors[eid][sid][lid].size();
                            // float_vector
                            // std::cerr << eid << ", "<<  tid  << ", " << sents_order.at(sid + index) << ", " << lid << std::endl;
                            if(elen == slen - 2){ // like ELMo
                                if(tid == 0 || tid >= slen - 1){
                                    sent_elmo[eid][tid][sid][lid] = std::vector<float>(elmo_vectors[eid][sid][lid][0].size(), 0.f);
                                }else{
                                    sent_elmo[eid][tid][sid][lid] = elmo_vectors[eid][sid][lid][tid-1];
                                }
                            }else if(elen == slen){ // like BERT
                                if(tid >= slen){
                                    sent_elmo[eid][tid][sid][lid] = elmo_vectors[eid][sid][lid][slen-1];
                                }else{
                                    sent_elmo[eid][tid][sid][lid] = elmo_vectors[eid][sid][lid][tid];
                                }
                            }else{
                                std::cerr << elen << ", " << slen << std::endl;
                                assert(false);
                            }
                        }
                    }
                }
            }

/*
            for(unsigned int elmo_id = 0; elmo_id < elmos.size(); elmo_id++)
                for(unsigned int layer_id = 0; layer_id < d.dict_set_src.d_feat.size(); feat_id++){
                    sent_feat[feat_id].resize(max_len);
                    for(unsigned int tok_pos = 0; tok_pos < max_len; tok_pos++){
                        sent_feat[feat_id][tok_pos].resize(batch_size_local);
                        for(unsigned int sid = 0; sid < batch_size_local; sid++){
                            if(tok_pos < vec_input.at(sents_order.at(sid + index)).size()){
                                sent_feat[feat_id][tok_pos][sid] = vec_input.at(sents_order.at(sid + index)).at(tok_pos).feat_id_list.at(feat_id);
                            }else{
                                sent_feat[feat_id][tok_pos][sid] = d.dict_set_src.end_id_feat.at(feat_id);
                            }
                        }
                    }
                }
            }
*/
        }
        
        // transpose output label matrix
        void label2batch(const unsigned int index, const unsigned int batch_size_local, const std::vector<unsigned int>& sents_order, const std::vector<std::vector<token> >& vec_input){
            // max token size
            unsigned int max_len = 0;
            for(unsigned int sid = 0; sid < batch_size_local; sid++){
                unsigned int cur_len = vec_input.at(sents_order.at(sid + index)).size();
                if(cur_len > max_len){
                    max_len = cur_len;
                }
            }
            sent_label.resize(max_len);
            for(unsigned int pos = 0; pos < max_len; pos++){
                sent_label[pos].resize(batch_size_local);
                for(unsigned int sid = 0; sid < batch_size_local; sid++){
                    if(pos < vec_input.at(sents_order.at(sid + index)).size()){
                        sent_label[pos][sid] = vec_input.at(sents_order.at(sid + index)).at(pos).word_id;
                    }else{
                        sent_label[pos][sid] = d.dict_set_trg.end_id_word;
                    }
                }
            }
        }

        // transpose alignment matrix
        std::vector<std::vector<unsigned int > > align2batch(const std::vector<unsigned int> sents_order, const unsigned int index, const unsigned int batch_size_local, const std::vector<std::vector<unsigned int> > &vec_input){
            unsigned int max_len = 0;
            for(unsigned int sid = 0; sid < batch_size_local && sid < vec_input.size(); sid++){
                unsigned int cur_len = vec_input.at(sents_order.at(sid + index)).size();
                if(cur_len > max_len){
                    max_len = cur_len;
                }
            }
            std::vector<std::vector<unsigned int>> col(max_len, std::vector<unsigned int>(batch_size_local, 0));
            for(unsigned int sid = 0; sid < batch_size_local && sid < vec_input.size(); sid++){
                for(unsigned int pos = 0; pos < max_len; pos++){
                    if(pos < vec_input.at(sents_order.at(sid + index)).size()){
                        col[pos][sid] = vec_input.at(sents_order.at(sid + index)).at(pos);
                    }else{
                        col[pos][sid] = vec_input.at(sents_order.at(sid + index)).back();
                    }
                }
            }
            return col;
        }

        // return maximum sentence length
        std::vector<unsigned int > trg2len(const std::vector<unsigned int> sents_order, const unsigned int index, const unsigned int batch_size_local, const std::vector<std::vector<token> > &vec_input){
            std::vector<unsigned int > len_vec;
            for(unsigned int sid = 0; sid < batch_size_local; sid++){
                unsigned int cur_len = vec_input.at(sents_order.at(sid + index)).size();
                len_vec.push_back(cur_len);
            }
            return len_vec;
        }

        // return maximum sentence length
        std::vector<unsigned int > src2len(const std::vector<unsigned int> sents_order, const unsigned int index, const unsigned int batch_size_local, const std::vector<std::vector<token> > &vec_input){
            std::vector<unsigned int > len_vec;
            for(unsigned int sid = 0; sid < batch_size_local; sid++){
                unsigned int cur_len = vec_input.at(sents_order.at(sid + index)).size();
                len_vec.push_back(cur_len);
            }
            return len_vec;
        }

        // Kiperwasser and Goldberg 2016
        void drop_word(){
            if(opts.drop_word_alpha > 0.0){
                std::uniform_real_distribution<float> prob(0.0, 1.0);
                for(unsigned int s_index = 0; s_index < sent_word.size(); s_index++){
                    for(unsigned int b_id = 0; b_id < sent_word.at(s_index).size(); b_id++){
                        unsigned int w_id = sent_word.at(s_index).at(b_id);
                        if(
                            w_id != d.dict_set_src.start_id_word
                            && w_id != d.dict_set_src.end_id_word
                            && w_id != d.dict_set_src.unk_id_word
                            && w_id != d.dict_set_src.pad_id_word
                        ){
                            float alpha = opts.drop_word_alpha;
                            float drop_prob = alpha / (alpha + (float)(d.dict_set_src.word_freq[w_id]));
                            if(prob(mt) < drop_prob){
                                sent_word[s_index][b_id] = d.dict_set_src.unk_id_word;
                            }
                        }
                    }
                }
            }
        }

    };

};

#endif
