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

#ifndef INCLUDE_GUARD_ENCODER_DECODER_LAHAN_HPP
#define INCLUDE_GUARD_ENCODER_DECODER_LAHAN_HPP

namespace s2s {

class encoder_decoder_slahan : public encoder_decoder {

public:

    dynet::Parameter p_Wad;
    dynet::Parameter p_Uad;
    dynet::Parameter p_vad;

    dynet::Parameter p_Wctx;
    dynet::Parameter p_bctx;
    
    std::vector<dynet::Parameter> p_Wahf;

    std::vector<dynet::Parameter> p_Wahb;

    dynet::Parameter p_select_W; // {rnn_size*2,rnn_size*7}
    dynet::Parameter p_select_b; // {rnn_size*2}

    std::vector<dynet::Expression> h_bi;
    std::vector<std::vector<dynet::Expression> > f_alpha_d;
    std::vector<std::vector<dynet::Expression> > b_alpha_d;

    dynet::Expression i_h_enc;
    dynet::Expression i_h_fin;
    dynet::Expression i_feed;

    // for visualization
    std::vector<dynet::Expression> gate_list;

    explicit encoder_decoder_slahan(dynet::ParameterCollection& model, const s2s_options& opts, dicts& d) : encoder_decoder(model,opts,d) {

        flag_drop_out = true;

        unsigned int dec_context_size = 0;
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
        
        p_Wad = model.add_parameters({opts.att_size, opts.rnn_size * 2});
        p_Uad = model.add_parameters({opts.att_size, opts.rnn_size * 2});
        p_vad = model.add_parameters({opts.att_size});

        p_Wctx = model.add_parameters({opts.rnn_size, opts.rnn_size*5});
        p_bctx = model.add_parameters({opts.rnn_size});

        for(unsigned int d = 0; d < opts.rec_attn_dep.size(); d++){
            p_Wahf.push_back(model.add_parameters({opts.rnn_size, opts.rnn_size * 2}));
        }

        for(unsigned int d = 0; d < opts.rec_attn_dep.size(); d++){
            p_Wahb.push_back(model.add_parameters({opts.rnn_size, opts.rnn_size * 2}));
        }

        if(opts.context_type == "both"){
            p_select_W = model.add_parameters({opts.rnn_size*2, opts.rnn_size*5});
            p_select_b = model.add_parameters({opts.rnn_size*2});
            dec_context_size = opts.rnn_size * 5;
        }else if(opts.context_type == "parent" or opts.context_type == "child"){
            p_select_W = model.add_parameters({opts.rnn_size*2, opts.rnn_size*3});
            p_select_b = model.add_parameters({opts.rnn_size*2});
            dec_context_size = opts.rnn_size * 5;
        }else{
            assert(false);
        }
        
        if(opts.additional_output_layer == true){
            dec_feeding_size = opts.rnn_size;
            p_add_W = model.add_parameters({dec_feeding_size, dec_context_size});
            p_add_bias = model.add_parameters({dec_feeding_size});
            p_out_R = model.add_parameters({d.dict_set_trg.d_word.size(), dec_feeding_size});
            p_out_bias = model.add_parameters({d.dict_set_trg.d_word.size()});
        }else{
            dec_feeding_size = dec_context_size;
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

    encoder_decoder_slahan(const encoder_decoder_slahan&) = delete;
    encoder_decoder_slahan& operator=(const encoder_decoder_slahan&) = delete;
    ~encoder_decoder_slahan() = default;

    void init_feed(const batch &batch_local, dynet::ComputationGraph& cg){
        i_feed = dynet::zeroes(cg, dynet::Dim({p_out_R.dim().d[1]}, batch_local.batch_size()));
    }

    void encoder(const batch &batch_local, dynet::ComputationGraph& cg) {
        // initialize
        init(batch_local, cg);
        // embedding
        embedding(batch_local, cg);
        init_feed(batch_local, cg);
        gate_list.clear();

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
        h_bi[0] = dynet::zeroes(cg, dynet::Dim({opts.rnn_size * 2}, batch_size));
        for (unsigned int t_i = 1; t_i < slen; ++t_i) {
            h_bi[t_i] = concatenate(std::vector<dynet::Expression>({h_fwd[t_i], h_bwd[t_i]}));
        }
        i_h_enc = concatenate_cols(h_bi);
        i_h_fin = concatenate(std::vector<dynet::Expression>({rev_enc_builder.h.back().back(), fwd_enc_builder.h.back().back()}));
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
    }

    std::vector<dynet::Expression> attention(dynet::ComputationGraph& cg, const batch &batch_local){
        if(opts.self_attn_type == "given"){
            return given_attention(cg, batch_local);
        }else if(opts.self_attn_type == "self"){
            return self_attention(cg, batch_local);
        }else{
            std::cerr << "Attention type does not match." << std::endl;
            assert(false);
        }
    }

    std::vector<dynet::Expression> given_attention(dynet::ComputationGraph& cg, const batch &batch_local){
        std::vector<dynet::Expression> h_att_self(slen);
        for(unsigned int child_pos = 0; child_pos < slen; child_pos++){
            std::vector<dynet::real> mask_real_zero_one(batch_size * slen);
            std::vector<dynet::real> mask_real_zero_min_flt_max(batch_size * slen);
            for(unsigned int batch_id = 0; batch_id < batch_size; batch_id++){
                for(unsigned int parent_pos = 0; parent_pos < slen; parent_pos++){
                    unsigned int v_idx = slen * batch_id + parent_pos;
                    if(parent_pos == batch_local.align[child_pos][batch_id]){
                        mask_real_zero_one[v_idx] = 0.0f;
                        mask_real_zero_min_flt_max[v_idx] = 0.0f;
                    }else{
                        mask_real_zero_one[v_idx] = 0.0f;
                        mask_real_zero_min_flt_max[v_idx] = opts.dummy_flt_min;
                    }
                }
            }
            h_att_self[child_pos] = dynet::input(cg, dynet::Dim({slen}, batch_size), mask_real_zero_one);
            h_att_self[child_pos] = h_att_self[child_pos] + dynet::input(cg, dynet::Dim({slen}, batch_size), mask_real_zero_min_flt_max);
        }
        return h_att_self;
    }

    std::vector<dynet::Expression> self_attention(dynet::ComputationGraph& cg, const batch &batch_local){
        std::vector<dynet::Expression> h_att_self(slen);
        dynet::Expression i_vad = parameter(cg, p_vad);
        dynet::Expression i_Wad = parameter(cg, p_Wad);
        dynet::Expression i_Uad = parameter(cg, p_Uad);
        dynet::Expression i_Uadhj = i_Uad * i_h_enc;
        for(unsigned int child_pos = 0; child_pos < slen; child_pos++){
            std::vector<dynet::real> mask_real_zero_one(batch_size * slen);
            std::vector<dynet::real> mask_real_zero_min_flt_max(batch_size * slen);
            for(unsigned int batch_id = 0; batch_id < batch_size; batch_id++){
                for(unsigned int parent_pos = 0; parent_pos < slen; parent_pos++){
                    unsigned int v_idx = slen * batch_id + parent_pos;
                    if(child_pos > 0 && parent_pos == child_pos){
                        mask_real_zero_one[v_idx] = 0.0f;
                        mask_real_zero_min_flt_max[v_idx] = opts.dummy_flt_min;
                    }else if(child_pos > 0 && parent_pos != child_pos){
                        mask_real_zero_one[v_idx] = 1.0f;
                        mask_real_zero_min_flt_max[v_idx] = 0.0f;
                    }else if(child_pos == 0 && parent_pos == child_pos){
                        mask_real_zero_one[v_idx] = 1.0f;
                        mask_real_zero_min_flt_max[v_idx] = 0.0f;
                    }else if(child_pos == 0 && parent_pos != child_pos){
                        mask_real_zero_one[v_idx] = 0.0f;
                        mask_real_zero_min_flt_max[v_idx] = opts.dummy_flt_min;
                    }
                }
            }
            dynet::Expression i_wadh = i_Wad * h_bi[child_pos];
            dynet::Expression i_Wadh = concatenate_cols(std::vector<dynet::Expression>(slen, i_wadh));
            dynet::Expression i_att = transpose(tanh(i_Wadh + i_Uadhj)) * i_vad;
            h_att_self[child_pos] = cmult(i_att, dynet::input(cg, dynet::Dim({slen}, batch_size), mask_real_zero_one));
            h_att_self[child_pos] = h_att_self[child_pos] + dynet::input(cg, dynet::Dim({slen}, batch_size), mask_real_zero_min_flt_max);
        }
        return h_att_self;
    }

    void recursive_attention(dynet::ComputationGraph& cg, std::vector<dynet::Expression>& h_att_self){
        // initialize forward and backward attention at d=0
        f_alpha_d.resize(opts.rec_attn_dep.back());
        b_alpha_d.resize(opts.rec_attn_dep.back());
        f_alpha_d[0].resize(slen);
        b_alpha_d[0].resize(slen);
        std::vector<dynet::Expression> i_f_alpha_d(opts.rec_attn_dep.back());
        for(unsigned int child_pos = 0; child_pos < slen; child_pos++){
            f_alpha_d[0][child_pos] = softmax(h_att_self[child_pos]);
        }
        i_f_alpha_d[0] = concatenate_cols(f_alpha_d[0]);
        for(unsigned int child_pos = 0; child_pos < slen; child_pos++){
            b_alpha_d[0][child_pos] = pick(i_f_alpha_d[0], child_pos, 0);
        }
        // calculate recursive attentions
        for(unsigned int d = 1; d < opts.rec_attn_dep.back(); d++){
            // To avoid the rounding errors
            unsigned int order = d + 1;
            unsigned int d_left = (order / 2 + order % 2) - 1;
            unsigned int d_right = order / 2 - 1;
            i_f_alpha_d[d] = i_f_alpha_d[d_left] * i_f_alpha_d[d_right];
            f_alpha_d[d].resize(slen);
            b_alpha_d[d].resize(slen);
            for(unsigned int child_pos = 0; child_pos < slen; child_pos++){
                f_alpha_d[d][child_pos] = pick(i_f_alpha_d[d], child_pos, 1);
                b_alpha_d[d][child_pos] = pick(i_f_alpha_d[d], child_pos, 0);
            }
        }
    }

    dynet::Expression decoder_output(dynet::ComputationGraph& cg, const unsigned int t, const std::vector<unsigned int>& prev){
        return decoder_output(cg, t, prev, dec_builder.state());
    }

    dynet::Expression decoder_output(dynet::ComputationGraph& cg, const unsigned int t, const std::vector<unsigned int>& prev, const dynet::RNNPointer pointer_prev){
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
        // construct vectors for decoder feeding
        dynet::Expression i_x_t = dynet::input(cg, dynet::Dim({3}, batch_size), bit_features);
        dynet::Expression i_dec_input = concatenate(std::vector<dynet::Expression>({i_x_t, i_feed}));
        i_dec_input = concatenate(std::vector<dynet::Expression>({i_dec_input, h_lookup[t]}));
        // update decoder states
        dec_builder.add_input(pointer_prev, i_dec_input);
        dynet::Expression i_h_dec = dec_builder.h.back().back();
        // create a current state vector
        dynet::Expression i_h_ctx = concatenate(std::vector<dynet::Expression>({i_h_dec, i_h_fin}));
        i_h_ctx = concatenate(std::vector<dynet::Expression>({i_h_ctx, h_bi[t]}));
        dynet::Expression i_Wctx = parameter(cg, p_Wctx);
        dynet::Expression i_bctx = parameter(cg, p_bctx);
        dynet::Expression i_h_ctx_attn = tanh(i_Wctx * i_h_ctx + i_bctx);
        // extract encoder hidden states from a forward attention layer
        dynet::Expression i_f_c_t = forward_attention(cg, t, i_h_ctx_attn);
        // extract encoder hidden states from a backward attention layer
        dynet::Expression i_b_c_t = backward_attention(cg, t, i_h_ctx_attn);
        // balance gate
        dynet::Expression i_c_t = balance_layer(cg, i_f_c_t, i_b_c_t, i_h_ctx_attn);
        // final context
        i_c_t = concatenate(std::vector<dynet::Expression>({h_bi[t], i_c_t}));
        i_c_t = concatenate(std::vector<dynet::Expression>({i_h_dec, i_c_t}));
        // to output layer
        if(opts.additional_output_layer){
            dynet::Expression i_add_W = parameter(cg, p_add_W);
            dynet::Expression i_add_bias = parameter(cg, p_add_bias);
            i_feed = tanh(i_add_W * i_c_t + i_add_bias);
        }else{
            i_feed = i_c_t;
        }
        // output layer
        dynet::Expression i_out_R = parameter(cg, p_out_R);
        dynet::Expression i_out_bias = parameter(cg, p_out_bias);
        dynet::Expression i_out_pred_t = i_out_R * i_feed + i_out_bias;
        
        return i_out_pred_t;
    }

    // forward attention layer
    dynet::Expression forward_attention(dynet::ComputationGraph& cg, const unsigned int t, dynet::Expression& i_h_ctx_attn){
        std::vector<dynet::Expression> f_c_t(opts.rec_attn_dep.size());
        std::vector<dynet::Expression> h_f(opts.rec_attn_dep.size());
        for(unsigned int d = 0; d < opts.rec_attn_dep.size(); d++){
            f_c_t[d] = i_h_enc * f_alpha_d[opts.rec_attn_dep[d]-1][t];
            h_f[d] = parameter(cg, p_Wahf[d]) * f_c_t[d];
        }
        dynet::Expression i_h_f_a = concatenate_cols(h_f);
        // hierarchical general attention
        dynet::Expression i_f_alpha_t = softmax(transpose(i_h_f_a) * i_h_ctx_attn);
        dynet::Expression i_f_c_t = concatenate_cols(f_c_t);
        i_f_c_t = i_f_c_t * i_f_alpha_t;
        return i_f_c_t;
    }

    // backward attention layer
    dynet::Expression backward_attention(dynet::ComputationGraph& cg, const unsigned int t, dynet::Expression& i_h_ctx_attn){
        std::vector<dynet::Expression> b_c_t(opts.rec_attn_dep.size());
        std::vector<dynet::Expression> h_b(opts.rec_attn_dep.size());
        for(unsigned int d = 0; d < opts.rec_attn_dep.size(); d++){
            // reflect recursive attention weights
            dynet::Expression i_filtered = cmult(transpose(b_alpha_d[opts.rec_attn_dep[d]-1][t]), i_h_enc);
            // general attention
            b_c_t[d] = maxpooling2d(i_filtered, std::vector<unsigned>({1,slen}), std::vector<unsigned>({1,slen}));
            h_b[d] = parameter(cg, p_Wahb[d]) * b_c_t[d];
        }
        dynet::Expression i_h_b_a = concatenate_cols(h_b);
        // hierarchical general attention
        dynet::Expression i_b_alpha_t = softmax(transpose(i_h_b_a) * i_h_ctx_attn);
        // output
        dynet::Expression i_b_c_t = concatenate_cols(b_c_t);
        i_b_c_t = i_b_c_t * i_b_alpha_t;
        return i_b_c_t;
    }

    // balance gate
    dynet::Expression balance_layer(dynet::ComputationGraph& cg, dynet::Expression& i_f_c_t, dynet::Expression& i_b_c_t, dynet::Expression& i_h_ctx_attn){
        dynet::Expression i_c_t;
        dynet::Expression i_select_W = parameter(cg, p_select_W);
        dynet::Expression i_select_b = parameter(cg, p_select_b);
        if(opts.context_type == "both"){
            dynet::Expression i_fb_c_t = concatenate(std::vector<dynet::Expression>({i_f_c_t, i_b_c_t}));
            if(opts.selective_gate == true){
                dynet::Expression i_fb = dynet::affine_transform({i_select_b, i_select_W, concatenate(std::vector<dynet::Expression>({i_fb_c_t, i_h_ctx_attn}))});
                dynet::Expression i_s = logistic(i_fb);
                gate_list.push_back(i_s);
                i_c_t = cmult(i_s, i_f_c_t) + cmult(1.f - i_s, i_b_c_t);
            }else{
                i_c_t = (i_f_c_t + i_b_c_t) / 2.f;
            }
        }else if(opts.context_type == "parent"){
            if(opts.selective_gate == true){
                dynet::Expression i_f = dynet::affine_transform({i_select_b, i_select_W, concatenate(std::vector<dynet::Expression>({i_f_c_t, i_h_ctx_attn}))});
                dynet::Expression i_s = logistic(i_f);
                gate_list.push_back(i_s);
                i_c_t = cmult(i_s, i_f_c_t);
            }else{
                i_c_t = i_f_c_t;
            }
        }else if(opts.context_type == "child"){
            if(opts.selective_gate == true){
                dynet::Expression i_b = dynet::affine_transform({i_select_b, i_select_W, concatenate(std::vector<dynet::Expression>({i_b_c_t, i_h_ctx_attn}))});
                dynet::Expression i_s = logistic(i_b);
                gate_list.push_back(i_s);
                i_c_t = cmult(i_s, i_b_c_t);
            }else{
                i_c_t = i_b_c_t;
            }
        }else{
            assert(false);
        }
        return i_c_t;
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
        ar & p_select_W;
        ar & p_select_b;
        ar & p_Wad;
        ar & p_Uad;
        ar & p_vad;
        ar & p_Wctx;
        ar & p_bctx;
        ar & p_Wahf;
        ar & p_Wahb;
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
