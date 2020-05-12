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

#ifndef INCLUDE_GUARD_LSTM_HPP
#define INCLUDE_GUARD_LSTM_HPP

namespace s2s {

class encoder_decoder_lstm : public encoder_decoder {

public:

    dynet::Expression i_h_lookup_cols;

    explicit encoder_decoder_lstm(dynet::ParameterCollection& model, const s2s_options& opts, dicts& d) : encoder_decoder(model,opts,d) {

        unsigned int dec_input_size = 0;
        unsigned int enc_input_size = rev_enc_builder.input_dim;
        flag_drop_out = true;
        // decoder input
        dec_input_size = 3;
        if(opts.flag_bit_feature == true){
            dec_input_size += 3;
        }
        if(opts.flag_par_embed == true){
            dec_input_size += enc_input_size * 2;
        }else{
            dec_input_size += enc_input_size;
        }
        assert(opts.dec_feature_vocab_size.size() == opts.dec_feature_vec_size.size());
        unsigned int cell_ratio = 2; // In GRU, cell_ration = 1;

        p_dec_init_w.resize(opts.num_layers * cell_ratio);
        for(unsigned int i = 0; i < opts.num_layers * cell_ratio; i++){
            p_dec_init_w[i] = model.add_parameters({opts.rnn_size, opts.rnn_size});
        }
        p_dec_init_bias.resize(opts.num_layers * cell_ratio);
        for(unsigned int i = 0; i < opts.num_layers * cell_ratio; i++){
            p_dec_init_bias[i] = model.add_parameters({opts.rnn_size});
        }

        if(opts.additional_output_layer == true){
            p_add_W = model.add_parameters({opts.rnn_size, opts.rnn_size});
            p_add_bias = model.add_parameters({opts.rnn_size});
            p_out_R = model.add_parameters({d.dict_set_trg.d_word.size(), opts.rnn_size});
            p_out_bias = model.add_parameters({d.dict_set_trg.d_word.size()});
        }else{
            p_out_R = model.add_parameters({d.dict_set_trg.d_word.size(), opts.rnn_size});
            p_out_bias = model.add_parameters({d.dict_set_trg.d_word.size()});
        }

        dec_builder = dynet::VanillaLSTMBuilder(
            opts.num_layers,
            dec_input_size,
            opts.rnn_size,
            model
        );
    }

    encoder_decoder_lstm(const encoder_decoder_lstm&) = delete;
    encoder_decoder_lstm& operator=(const encoder_decoder_lstm&) = delete;
    ~encoder_decoder_lstm() = default;

    void encoder(const batch &batch_local, dynet::ComputationGraph& cg) {
        // initialize
        init(batch_local, cg);
        // embedding
        embedding(batch_local, cg);
        i_h_lookup_cols = concatenate_cols(h_lookup);
        // backward encoder
        rev_enc_builder.new_graph(cg);
        rev_enc_builder.start_new_sequence();
        if(flag_drop_out == true && opts.dropout_rate_lstm_word > 0.f){
            rev_enc_builder.set_dropout_masks(batch_size);
        }
        for (unsigned int ind = 0; ind < slen; ++ind) {
            unsigned int t_i = (slen - 1) - ind;
            rev_enc_builder.add_input(h_lookup[t_i]);
        };
        std::vector<dynet::Expression> vec_enc_final_state = rev_enc_builder.final_s();
        if(opts.flag_bi_enc == true){
            // forward encoder
            fwd_enc_builder.new_graph(cg);
            fwd_enc_builder.start_new_sequence();
            if(flag_drop_out == true && opts.dropout_rate_lstm_word > 0.f){
                fwd_enc_builder.set_dropout_masks(batch_size);
            }
            for (unsigned int t_i = 0; t_i < slen; ++t_i) {
                fwd_enc_builder.add_input(h_lookup[t_i]);
            };
            std::vector<dynet::Expression> vec_fwd_enc_final_state = fwd_enc_builder.final_s();
            for (unsigned int lay_i = 0; lay_i < vec_fwd_enc_final_state.size(); lay_i++){
                vec_enc_final_state[lay_i] = vec_enc_final_state[lay_i] + vec_fwd_enc_final_state.at(lay_i);
            }
        }
        // Initialize decoder
        dec_builder.new_graph(cg);
        if(opts.additional_connect_layer){
            std::vector<dynet::Expression> vec_dec_init_state;
            for (unsigned int lay_i = 0; lay_i < vec_enc_final_state.size(); lay_i++){
                dynet::Expression i_dec_init_w = parameter(cg, p_dec_init_w[lay_i]);
                dynet::Expression i_dec_init_bias = parameter(cg, p_dec_init_bias[lay_i]);
                vec_dec_init_state.push_back(tanh(i_dec_init_w * vec_enc_final_state[lay_i] + i_dec_init_bias));
            }
            dec_builder.start_new_sequence(vec_dec_init_state);
        }else{
            dec_builder.start_new_sequence(vec_enc_final_state);
        }
        if(flag_drop_out == true && opts.dropout_rate_lstm_word > 0.f){
            dec_builder.set_dropout_masks(batch_size);
        }
    }

    dynet::Expression decoder_output(dynet::ComputationGraph& cg, const unsigned int position, const std::vector<unsigned int>& head, const std::vector<dynet::real>& bit_features){
        return decoder_output(cg, position, head, bit_features, dec_builder.state());
    }

    dynet::Expression decoder_output(dynet::ComputationGraph& cg, const unsigned int position, const std::vector<unsigned int> &head, const std::vector<dynet::real>& bit_features, const dynet::RNNPointer pointer_prev){
        dynet::Expression i_dec_input = h_lookup.at(position);
        if(opts.flag_bit_feature){
            i_dec_input = concatenate(std::vector<dynet::Expression>({i_dec_input, dynet::input(cg, dynet::Dim({6}, batch_size), bit_features)}));
        }else{
            i_dec_input = concatenate(std::vector<dynet::Expression>({i_dec_input, dynet::input(cg, dynet::Dim({3}, batch_size), bit_features)}));
        }
        if(opts.flag_par_embed){
            i_dec_input = concatenate(std::vector<dynet::Expression>({i_dec_input, pick(transpose(i_h_lookup_cols), head)}));
        }
        dynet::Expression i_h_dec = dec_builder.add_input(pointer_prev, i_dec_input);
        dynet::Expression i_final;
        if(opts.additional_output_layer){
            dynet::Expression i_add_W = parameter(cg, p_add_W);
            dynet::Expression i_add_bias = parameter(cg, p_add_bias);
            i_final = tanh(i_add_W * i_h_dec + i_add_bias);
        }else{
            i_final = i_h_dec;
        }
        dynet::Expression i_out_R = parameter(cg, p_out_R);
        dynet::Expression i_out_bias = parameter(cg, p_out_bias);
        dynet::Expression i_out_pred_t = i_out_R * i_final + i_out_bias;
        return i_out_pred_t;
    }

    std::vector<dynet::real > bit_features(const unsigned int t, const std::vector<std::vector<unsigned int> >& output, const std::vector<unsigned int>& head){
        // t should be always larger than 0.
        assert(t > 0);
        // convert previous label to the bit features.
        std::vector<dynet::real> features;
        const unsigned int start_id = d.dict_set_trg.start_id_word;
        const unsigned int kep_id = d.dict_set_trg.keep_id_word;
        const unsigned int del_id = d.dict_set_trg.delete_id_word;
        for(unsigned int batch_id = 0; batch_id < batch_size; batch_id++){
            // start
            if(t > 0 && output[t-1][batch_id] == start_id){
                features.push_back(opts.bit_size_flt);
            }else{
                features.push_back(0.0f);
            }
            // keep
            if(t > 0 && output[t-1][batch_id] == kep_id){
                features.push_back(opts.bit_size_flt);
            }else{
                features.push_back(0.0f);
            }
            // delete
            if(t > 0 && output[t-1][batch_id] == del_id){
                features.push_back(opts.bit_size_flt);
            }else{
                features.push_back(0.0f);
            }
            if(opts.flag_bit_feature == true){
                // The parent word was kept.
                if(t == 0 || (head[batch_id] < t && output[head[batch_id]][batch_id] == kep_id)){
                    features.push_back(opts.bit_size_flt);
                }else{
                    features.push_back(0.0f);
                }
                // The parent word was deleted.
                if(t != 0 || (head[batch_id] < t && output[head[batch_id]][batch_id] == del_id)){
                    features.push_back(opts.bit_size_flt);
                }else{
                    features.push_back(0.0f);
                }
                // The parent word comes later.
                if(t != 0 || head[batch_id] > t){
                    features.push_back(opts.bit_size_flt);
                }else{
                    features.push_back(0.0f);
                }
            }
        }
        return features;
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
        ar & p_layer_w;
        ar & p_add_W;
        ar & p_add_bias;
        ar & p_dec_init_bias;
        ar & p_dec_init_w;
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
