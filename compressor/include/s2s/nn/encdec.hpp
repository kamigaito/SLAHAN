#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/rnn.h"
#include "dynet/gru.h"
#include "dynet/lstm.h"
#include "dynet/fast-lstm.h"
#include "dynet/expr.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <type_traits>
#include <sys/types.h>
#include <sys/stat.h>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options.hpp>

#include "s2s/dynet/dict.h"
#include "s2s/corpus/options.hpp"
#include "s2s/corpus/corpora.hpp"
#include "s2s/corpus/dicts.hpp"
#include "s2s/corpus/batch.hpp"

#include "word2vec/readbin.hpp"

#ifndef INCLUDE_GUARD_ENCODER_DECODER_ENCODER_HPP
#define INCLUDE_GUARD_ENCODER_DECODER_ENCODER_HPP

namespace s2s {

class encoder_decoder {

public:

    dynet::VanillaLSTMBuilder rev_enc_builder;
    dynet::VanillaLSTMBuilder fwd_enc_builder;
    
    dynet::VanillaLSTMBuilder dec_builder;

    dynet::VanillaLSTMBuilder rev_char_enc_builder;
    dynet::VanillaLSTMBuilder fwd_char_enc_builder;
    
    dynet::LookupParameter p_word_enc;
    dynet::LookupParameter p_char_enc;
    std::vector<dynet::LookupParameter> p_feat_enc;
    std::vector<dynet::Parameter> p_dec_init_bias;
    std::vector<dynet::Parameter> p_dec_init_w;
    std::vector<std::vector<dynet::Parameter> > p_layer_w;
    dynet::Parameter p_add_W;
    dynet::Parameter p_add_bias;
    dynet::Parameter p_out_R;
    dynet::Parameter p_out_bias;

    const s2s_options& opts;
    dicts& d;
    
    std::vector<dynet::Expression> h_lookup;

    bool flag_drop_out;
    unsigned int slen;
    unsigned int batch_size;

    explicit encoder_decoder(dynet::ParameterCollection& model, const s2s_options& opts, dicts& d) : opts(opts),d(d) {

        flag_drop_out = true;

        unsigned int enc_input_size = 0;

        assert(opts.enc_feature_vocab_size.size() == opts.enc_feature_vec_size.size());
        //  word embedding
        if(opts.lookup_type == "word" || opts.lookup_type == "word+char"){
            p_word_enc = model.add_lookup_parameters(d.dict_set_src.d_word.size(), {opts.enc_word_vec_size});
            enc_input_size += opts.enc_word_vec_size;
        }
        // charactor embedding
        if(opts.lookup_type == "char" || opts.lookup_type == "word+char"){
            enc_input_size += opts.char_rnn_size * 2;
            p_char_enc = model.add_lookup_parameters(d.dict_set_src.d_char.size(), {opts.enc_char_vec_size});
        }
        // feature embedding
        for(unsigned int feat_id = 0; feat_id < opts.enc_feature_vec_size.size(); feat_id++){
            p_feat_enc.push_back(model.add_lookup_parameters(d.dict_set_src.d_feat.at(feat_id).size(), {opts.enc_feature_vec_size.at(feat_id)}));
            enc_input_size += opts.enc_feature_vec_size.at(feat_id);
        }
        // elmo style embedding
        p_layer_w.resize(opts.elmo_hdf5_layers.size());
        for(unsigned int eid = 0; eid < p_layer_w.size(); eid++){
            p_layer_w[eid].resize(opts.elmo_hdf5_layers.at(eid));
            for(unsigned int lid = 0; lid < p_layer_w.at(eid).size(); lid++){
                p_layer_w[eid][lid] = model.add_parameters({opts.elmo_hdf5_dims.at(eid)});
            }
            enc_input_size += opts.elmo_hdf5_dims.at(eid);
        }
        // charactor level LSTM
        rev_char_enc_builder = dynet::VanillaLSTMBuilder(
            opts.char_num_layers,
            opts.enc_char_vec_size,
            opts.char_rnn_size,
            model
        );
        fwd_char_enc_builder = dynet::VanillaLSTMBuilder(
            opts.char_num_layers,
            opts.enc_char_vec_size,
            opts.char_rnn_size,
            model
        );
        // word level LSTM
        rev_enc_builder = dynet::VanillaLSTMBuilder(
            opts.num_layers,
            enc_input_size,
            opts.rnn_size,
            model
        );
        fwd_enc_builder = dynet::VanillaLSTMBuilder(
            opts.num_layers,
            enc_input_size,
            opts.rnn_size,
            model
        );
    }
    
    void init (const batch &batch_local, dynet::ComputationGraph& cg){
        batch_size = batch_local.batch_size();
        if(opts.lookup_type == "word" || opts.lookup_type == "word+char" || opts.lookup_type == "none"){
            slen = batch_local.sent_word.size();
        }else if(opts.lookup_type == "char"){
            slen = batch_local.sent_char.size();
        }else{
            assert(false);
        }
    }

    void embedding (const batch &batch_local, dynet::ComputationGraph& cg){

        h_lookup.resize(slen);

        std::vector<std::vector<dynet::Expression> > vec_lookup(slen);
        // word
        if(opts.lookup_type == "word" || opts.lookup_type == "word+char"){
            for(unsigned int t_i = 0; t_i < slen; t_i++){
                for(unsigned int b_i = 0; b_i < batch_size; b_i++){
                    if(batch_local.sent_word.at(t_i).at(b_i) >= p_word_enc.dim().d[1]){
                        std::cerr << "0 < " << batch_local.sent_word.at(t_i).at(b_i) << " < " << p_word_enc.dim().d[1] << std::endl;
                        assert(false);
                    }
                }
                vec_lookup[t_i].push_back(lookup(cg, p_word_enc, batch_local.sent_word[t_i]));
            }
        }else if(opts.lookup_type != "char" and opts.lookup_type != "none"){
            assert(false);
        }
        // char
        if(opts.lookup_type == "char" || opts.lookup_type == "word+char"){
            fwd_char_enc_builder.new_graph(cg);
            rev_char_enc_builder.new_graph(cg);
            for(unsigned int t_i = 0; t_i < slen; t_i++){
                fwd_char_enc_builder.start_new_sequence();
                rev_char_enc_builder.start_new_sequence();
                if(flag_drop_out == true && opts.dropout_rate_lstm_char > 0.f){
                    fwd_char_enc_builder.set_dropout_masks(batch_size);
                    rev_char_enc_builder.set_dropout_masks(batch_size);
                }
                unsigned int char_len = batch_local.sent_char[t_i].size();
                for(unsigned int c_i = 0; c_i < char_len; c_i++){
                    // std::cerr << __LINE__ << ", " << c_i << std::endl;
                    for(unsigned int b_i = 0; b_i < batch_size; b_i++){
                        if(batch_local.sent_char.at(t_i).at(c_i).at(b_i) >= p_char_enc.dim().d[1]){
                            std::cerr << "0 < " << batch_local.sent_char.at(t_i).at(c_i).at(b_i) << " < " << p_char_enc.dim().d[1] << std::endl;
                            assert(false);
                        }
                    }
                    fwd_char_enc_builder.add_input(lookup(cg, p_char_enc, batch_local.sent_char[t_i][c_i]));
                }
                for(unsigned int c_i_d = 0; c_i_d < char_len; c_i_d++){
                    unsigned int c_i = (char_len - 1) - c_i_d;
                    // std::cerr << __LINE__ << ", " << c_i << std::endl;
                    for(unsigned int b_i = 0; b_i < batch_size; b_i++){
                        if(batch_local.sent_char.at(t_i).at(c_i).at(b_i) >= p_char_enc.dim().d[1]){
                            std::cerr << "0 < " << batch_local.sent_char.at(t_i).at(c_i).at(b_i) << " < " << p_char_enc.dim().d[1] << std::endl;
                            assert(false);
                        }
                    }
                    rev_char_enc_builder.add_input(lookup(cg, p_char_enc, batch_local.sent_char[t_i][c_i]));
                }
                vec_lookup[t_i].push_back(concatenate(std::vector<dynet::Expression>({fwd_char_enc_builder.back(), rev_char_enc_builder.back()})));
            }
        }else if(opts.lookup_type != "word" and opts.lookup_type != "none"){
            assert(false);
        }
        // feature
        for(unsigned int f_id = 0; f_id < batch_local.sent_feat.size(); f_id++){
            for(unsigned int tok_pos = 0; tok_pos < slen; tok_pos++){
                for(unsigned int b_i = 0; b_i < batch_size; b_i++){
                    if(batch_local.sent_feat.at(f_id).at(tok_pos).at(b_i) >= p_feat_enc.at(f_id).dim().d[1]){
                        std::cerr << "0 < " << batch_local.sent_feat.at(f_id).at(tok_pos).at(b_i) << " < " << p_feat_enc.at(f_id).dim().d[1] << std::endl;
                        assert(false);
                    }
                }
                vec_lookup[tok_pos].push_back(lookup(cg, p_feat_enc[f_id], batch_local.sent_feat[f_id][tok_pos]));
            }
        }
        // elmo style embeddings
        // elmo_type, token_id, sentence_id, layer_id, float_vector
        for(unsigned int tid = 0; tid < batch_local.sent_elmo.at(0).size(); tid++){
            for(unsigned int eid = 0; eid < batch_local.sent_elmo.size(); eid++){
                const unsigned int dim_size = opts.elmo_hdf5_dims.at(eid);
                const unsigned int layer_size = opts.elmo_hdf5_layers.at(eid);
                const unsigned int slen = batch_local.sent_elmo.at(eid).at(tid).size();
                // extract elmo layers
                std::vector<float> layers;
                layers.reserve(dim_size*layer_size*batch_size);
                for(unsigned int sid = 0; sid < slen; sid++){
                    for(unsigned int lid = 0; lid < opts.elmo_hdf5_layers[eid]; lid++){
                        const std::vector<float> &flt_vec = batch_local.sent_elmo.at(eid).at(tid).at(sid).at(lid);
                        layers.insert(layers.end(), flt_vec.begin(), flt_vec.end());
                    }
                }
                assert(layers.size() == dim_size*layer_size*batch_size);
                dynet::Expression i_elmo_layers = dynet::input(cg, dynet::Dim({layer_size, dim_size}, batch_size), layers);
                // calculate weights for each layer
                std::vector<dynet::Expression> elmo_weights;
                for(unsigned int lid = 0; lid < opts.elmo_hdf5_layers[eid]; lid++){
                    std::vector<float> layer;
                    layer.reserve(dim_size*batch_size);
                    for(unsigned int sid = 0; sid < slen; sid++){
                        const std::vector<float> &flt_vec = batch_local.sent_elmo.at(eid).at(tid).at(sid).at(lid);
                        layer.insert(layer.end(), flt_vec.begin(), flt_vec.end());
                    }
                    assert(layer.size() == dim_size*batch_size);
                    dynet::Expression i_elmo_layer = dynet::input(cg, dynet::Dim({dim_size}, batch_size), layer);
                    elmo_weights.push_back(transpose(parameter(cg, p_layer_w[eid][lid])) * i_elmo_layer);
                }
                dynet::Expression i_elmo_weights = concatenate_cols(elmo_weights);
                // calculate weighted sum
                dynet::Expression i_elmo_alpha = softmax(i_elmo_weights);
                dynet::Expression i_elmo_weighted_layers = i_elmo_alpha * i_elmo_layers;
                vec_lookup[tid].push_back(transpose(i_elmo_weighted_layers));
            }
        }
        // concatenate all input vectors
        for(unsigned int tok_pos = 0; tok_pos < slen; tok_pos++){
            h_lookup[tok_pos] = concatenate(vec_lookup[tok_pos]);
        }
    }

    virtual void encoder(const batch &batch_local, dynet::ComputationGraph& cg)=0;

    virtual void disable_dropout()=0;

    virtual void enable_dropout()=0;

    void read_pre_train_src(const std::string file_name, dynet::Dict& d_word_src){
        std::unordered_map<unsigned int, std::vector<float > > init_lookup;
        unsigned int embed_size = p_word_enc.dim().d[0];
        word2vec::readbin(init_lookup, file_name, d_word_src, embed_size);
        for(std::pair<unsigned int, std::vector<float> > p : init_lookup){
            p_feat_enc[0].initialize(p.first, p.second);
        }
    }

};

}

#endif

