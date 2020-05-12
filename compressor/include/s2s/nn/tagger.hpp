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

#ifndef INCLUDE_GUARD_TAGGER_HPP
#define INCLUDE_GUARD_TAGGER_HPP

namespace s2s {

class tagger : public encoder_decoder {

public:

    std::vector<dynet::Expression> h_bi;

    explicit tagger(dynet::ParameterCollection& model, const s2s_options& opts, dicts& d) : encoder_decoder(model,opts,d) {

        flag_drop_out = true;

        if(opts.additional_output_layer == true){
            p_add_W = model.add_parameters({opts.rnn_size, opts.rnn_size * 2});
            p_add_bias = model.add_parameters({opts.rnn_size});
            p_out_R = model.add_parameters({d.dict_set_trg.d_word.size(), opts.rnn_size});
            p_out_bias = model.add_parameters({d.dict_set_trg.d_word.size()});
        }else{
            p_out_R = model.add_parameters({d.dict_set_trg.d_word.size(), opts.rnn_size * 2});
            p_out_bias = model.add_parameters({d.dict_set_trg.d_word.size()});
        }

    }

    tagger(const tagger&) = delete;
    tagger& operator=(const tagger&) = delete;
    ~tagger() = default;

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
    }

    dynet::Expression output(dynet::ComputationGraph& cg, const unsigned int t){
        dynet::Expression i_final;
        if(opts.additional_output_layer){
            dynet::Expression i_add_W = parameter(cg, p_add_W);
            dynet::Expression i_add_bias = parameter(cg, p_add_bias);
            i_final = tanh(i_add_W * h_bi[t] + i_add_bias);
        }else{
            i_final = h_bi[t];
        }
        dynet::Expression i_out_R = parameter(cg, p_out_R);
        dynet::Expression i_out_bias = parameter(cg, p_out_bias);
        dynet::Expression i_out_pred_t = i_out_R * i_final + i_out_bias;
        return i_out_pred_t;
    }

    void disable_dropout(){
        fwd_char_enc_builder.disable_dropout();
        rev_char_enc_builder.disable_dropout();
        fwd_enc_builder.disable_dropout();
        rev_enc_builder.disable_dropout();
        flag_drop_out = false;
    }

    void enable_dropout(){
        fwd_char_enc_builder.set_dropout(opts.dropout_rate_lstm_char, 0.f);
        rev_char_enc_builder.set_dropout(opts.dropout_rate_lstm_char, 0.f);
        fwd_enc_builder.set_dropout(opts.dropout_rate_lstm_word, 0.f);
        rev_enc_builder.set_dropout(opts.dropout_rate_lstm_word, 0.f);
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
        ar & p_out_R;
        ar & p_out_bias;
        ar & rev_char_enc_builder;
        ar & fwd_char_enc_builder;
        ar & rev_enc_builder;
        ar & fwd_enc_builder;
    }

};

};

#endif

