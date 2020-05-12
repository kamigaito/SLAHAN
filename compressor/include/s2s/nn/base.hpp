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
#include "s2s/nn/encdec.hpp"

#ifndef INCLUDE_GUARD_ENCODER_DECODER_BASE_HPP
#define INCLUDE_GUARD_ENCODER_DECODER_BASE_HPP

namespace s2s {

class encoder_decoder_base : public encoder_decoder {

public:

    std::vector<dynet::Expression> h_bi;
    dynet::Expression i_feed;

    explicit encoder_decoder_base(dynet::ParameterCollection& model, const s2s_options& opts, dicts& d) : encoder_decoder(model,opts,d) {

        flag_drop_out = true;
        unsigned int dec_feeding_size = 0;

        assert(opts.dec_feature_vocab_size.size() == opts.dec_feature_vec_size.size());
        unsigned int cell_ratio = 2; // In GRU, cell_ratio = 1;

        p_dec_init_w.resize(opts.num_layers * cell_ratio);
        for(unsigned int i = 0; i < opts.num_layers * cell_ratio; i++){
            p_dec_init_w[i] = model.add_parameters({opts.rnn_size, opts.rnn_size});
        }
        p_dec_init_bias.resize(opts.num_layers * cell_ratio);
        for(unsigned int i = 0; i < opts.num_layers * cell_ratio; i++){
            p_dec_init_bias[i] = model.add_parameters({opts.rnn_size});
        }
        if(opts.additional_output_layer == true){
            dec_feeding_size = opts.rnn_size;
        }else{
            dec_feeding_size = opts.rnn_size * 3;
        }
        if(opts.additional_output_layer == true){
            p_add_W = model.add_parameters({dec_feeding_size, opts.rnn_size * 3});
            p_add_bias = model.add_parameters({dec_feeding_size});
            p_out_R = model.add_parameters({d.dict_set_trg.d_word.size(), opts.rnn_size});
            p_out_bias = model.add_parameters({d.dict_set_trg.d_word.size()});
        }else{
            p_out_R = model.add_parameters({d.dict_set_trg.d_word.size(), dec_feeding_size});
            p_out_bias = model.add_parameters({d.dict_set_trg.d_word.size()});
        }

        dec_builder = dynet::VanillaLSTMBuilder(
            opts.num_layers,
            (rev_enc_builder.input_dim + dec_feeding_size + 3),
            opts.rnn_size,
            model
        );
    
    }

    encoder_decoder_base(const encoder_decoder_base&) = delete;
    encoder_decoder_base& operator=(const encoder_decoder_base&) = delete;
    ~encoder_decoder_base() = default;

    void encoder(const batch &batch_local, dynet::ComputationGraph& cg) {
        // initialize
        init(batch_local, cg);
        // embedding
        embedding(batch_local, cg);

        std::vector<dynet::Expression> h_fwd(slen);
        std::vector<dynet::Expression> h_bwd(slen);
        h_bi.resize(slen);

        // forward encoder
        fwd_enc_builder.new_graph(cg);
        fwd_enc_builder.start_new_sequence();
        if(flag_drop_out == true && opts.dropout_rate_lstm_word > 0.f){
            fwd_enc_builder.set_dropout_masks(batch_size);
        }
        for (unsigned int t_i = 0; t_i < slen; ++t_i) {
            fwd_enc_builder.add_input(h_lookup[t_i]);
            h_fwd[t_i] = fwd_enc_builder.back();
        }
        // backward encoder
        rev_enc_builder.new_graph(cg);
        rev_enc_builder.start_new_sequence();
        if(flag_drop_out == true && opts.dropout_rate_lstm_word > 0.f){
            rev_enc_builder.set_dropout_masks(batch_size);
        }
        for (unsigned int ind = 0; ind < slen; ++ind) {
            unsigned int t_i = (slen - 1) - ind;
            rev_enc_builder.add_input(h_lookup[t_i]);
            h_bwd[t_i] = rev_enc_builder.back();
        }
        // bidirectional encoding
        for (unsigned int t_i = 0; t_i < slen; ++t_i) {
            h_bi[t_i] = concatenate(std::vector<dynet::Expression>({h_fwd[t_i], h_bwd[t_i]}));
        }
        // Initialize decoder
        std::vector<dynet::Expression> vec_enc_final_state;
        vec_enc_final_state = rev_enc_builder.final_s();
        dec_builder.new_graph(cg);
        if(opts.additional_connect_layer){
            std::vector<dynet::Expression> vec_dec_init_state;
            for (unsigned int i = 0; i < vec_enc_final_state.size(); i++){
                dynet::Expression i_dec_init_w = parameter(cg, p_dec_init_w[i]);
                dynet::Expression i_dec_init_bias = parameter(cg, p_dec_init_bias[i]);
                vec_dec_init_state.push_back(tanh(i_dec_init_w * vec_enc_final_state[i] + i_dec_init_bias));
            }
            dec_builder.start_new_sequence(vec_dec_init_state);
        }else{
            dec_builder.start_new_sequence(vec_enc_final_state);
        }
        if(flag_drop_out == true && opts.dropout_rate_lstm_word > 0.f){
            dec_builder.set_dropout_masks(batch_size);
        }
        init_feed(batch_local, cg);
    }
    
    void init_feed(const batch &batch_local, dynet::ComputationGraph& cg){
        i_feed = dynet::zeroes(cg, dynet::Dim({p_out_R.dim().d[1]}, batch_local.batch_size()));
    }

    dynet::Expression decoder_output(dynet::ComputationGraph& cg,  const std::vector<unsigned int> prev, const unsigned int t){
        return decoder_output(cg, prev, t, dec_builder.state());
    }

    dynet::Expression decoder_output(dynet::ComputationGraph& cg,  const std::vector<unsigned int> prev, const unsigned int t, const dynet::RNNPointer pointer_prev){
        // t should be always larger than 0.
        assert(t > 0);
        // convert previous label to the bit features.
        std::vector<dynet::real> bit_features;
        const unsigned int start_id = d.dict_set_trg.start_id_word;
        const unsigned int kep_id = d.dict_set_trg.keep_id_word;
        const unsigned int del_id = d.dict_set_trg.delete_id_word;
        for(unsigned int batch_id = 0; batch_id < prev.size(); batch_id++){
            // start
            if(prev[batch_id] == start_id){
                bit_features.push_back(opts.bit_size_flt);
            }else{
                bit_features.push_back(0.0f);
            }
            // keep
            if(prev[batch_id] == kep_id){
                bit_features.push_back(opts.bit_size_flt);
            }else{
                bit_features.push_back(0.0f);
            }
            // delete
            if(prev[batch_id] == del_id){
                bit_features.push_back(opts.bit_size_flt);
            }else{
                bit_features.push_back(0.0f);
            }
        }
        dynet::Expression i_x_t = dynet::input(cg, dynet::Dim({3}, batch_size), bit_features);
        dynet::Expression i_dec_input = concatenate(std::vector<dynet::Expression>({i_x_t, i_feed}));
        i_dec_input = concatenate(std::vector<dynet::Expression>({i_dec_input, h_lookup[t]}));
        dec_builder.add_input(pointer_prev, i_dec_input);
        dynet::Expression i_h_dec = dec_builder.h.back().back();
        if(opts.additional_output_layer){
            dynet::Expression i_add_W = parameter(cg, p_add_W);
            dynet::Expression i_add_bias = parameter(cg, p_add_bias);
            i_feed = tanh(i_add_W * concatenate(std::vector<dynet::Expression>({i_h_dec, h_bi[t]})) + i_add_bias);
        }else{
            i_feed = concatenate(std::vector<dynet::Expression>({i_h_dec, h_bi[t]}));
        }
        dynet::Expression i_out_R = parameter(cg, p_out_R);
        dynet::Expression i_out_bias = parameter(cg, p_out_bias);
        dynet::Expression i_out_pred_t = i_out_R * i_feed + i_out_bias;
        return i_out_pred_t;
    }

    void disable_dropout(){
        fwd_char_enc_builder.disable_dropout();
        rev_char_enc_builder.disable_dropout();
        fwd_enc_builder.disable_dropout();
        rev_enc_builder.disable_dropout();
        dec_builder.disable_dropout();
        flag_drop_out = false;
    }

    void enable_dropout(){
        fwd_char_enc_builder.set_dropout(opts.dropout_rate_lstm_char, 0.f);
        rev_char_enc_builder.set_dropout(opts.dropout_rate_lstm_char, 0.f);
        fwd_enc_builder.set_dropout(opts.dropout_rate_lstm_word, 0.f);
        rev_enc_builder.set_dropout(opts.dropout_rate_lstm_word, 0.f);
        dec_builder.set_dropout(opts.dropout_rate_lstm_word, 0.f);
        flag_drop_out = true;
    }

private:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) {
        ar & p_word_enc;
        ar & p_char_enc;
        ar & p_feat_enc;
        ar & p_dec_init_bias;
        ar & p_dec_init_w;
        ar & p_layer_w;
        ar & p_add_W;
        ar & p_add_bias;
        ar & p_out_R;
        ar & p_out_bias;
        ar & dec_builder;
        ar & rev_char_enc_builder;
        ar & fwd_char_enc_builder;
        ar & rev_enc_builder;
        ar & fwd_enc_builder;
    }

};

}

#endif
