#include "dynet/nodes.h"
#include "dynet/exec.h"
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
#include <memory>
#include <type_traits>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options.hpp>

#include "s2s/dynet/dict.h"
#include "s2s/nn/attn.hpp"
#include "s2s/nn/base.hpp"
#include "s2s/nn/slahan.hpp"
#include "s2s/nn/lstm.hpp"
#include "s2s/nn/tagger.hpp"
#include "s2s/corpus/comp.hpp"
#include "s2s/corpus/dicts.hpp"
#include "s2s/decode/hypothesis.hpp"

#ifndef INCLUDE_GUARD_DECODER_GREEDY_HPP
#define INCLUDE_GUARD_DECODER_GREEDY_HPP

namespace s2s {

    void greedy_decode_ensemble(const batch& batch_local, std::vector<hypothesis<encoder_decoder_attn>>& hyp_stack, std::vector<std::unique_ptr<encoder_decoder_attn> > &encdec, dicts &d, const s2s_options &opts){
        dynet::ComputationGraph cg;
        hyp_stack.resize(1);
        hyp_stack[0].init(batch_local, d, encdec, cg);
        for (unsigned int t = 1; t < encdec[0]->slen; ++t) {
            std::vector<std::vector<float> > output(batch_local.batch_size(), std::vector<float>(d.dict_set_trg.d_word.size(), 0.f));
            // for each model
            for(unsigned int model_id = 0; model_id < encdec.size(); model_id++){
                dynet::Expression i_att_t = encdec[model_id]->decoder_attention(cg, t, hyp_stack[0].prev());
                dynet::Expression i_out_t = encdec[model_id]->decoder_output(cg, t, i_att_t);
                dynet::Expression pred_att = softmax(i_att_t);
                std::vector<dynet::Tensor> results_att = cg.incremental_forward(pred_att).batch_elems();
                dynet::Expression pred_out = softmax(i_out_t);
                std::vector<dynet::Tensor> results_out = cg.incremental_forward(pred_out).batch_elems();
                // for each sentence
                for(unsigned int sent_id = 0; sent_id < results_out.size(); sent_id++){
                    hyp_stack[0].dist_att[model_id][sent_id].push_back(as_vector(results_att.at(sent_id)));
                    hyp_stack[0].dist_out[model_id][sent_id].push_back(as_vector(results_out.at(sent_id)));
                    for(unsigned int vocab_id = 0; vocab_id < hyp_stack[0].dist_out[model_id][sent_id][t-1].size(); vocab_id++){
                        output[sent_id][vocab_id] += log(hyp_stack[0].dist_out[model_id][sent_id][t-1][vocab_id]);
                    }
                }
            }
            // for each sentence
            for(unsigned int sent_id = 0; sent_id < output.size(); sent_id++){
                for(unsigned int vocab_id = 0; vocab_id < output[sent_id].size(); vocab_id++){
                    output[sent_id][vocab_id] /= (float)(encdec.size());
                }
            }
            // output vocab
            for(unsigned int sent_id = 0; sent_id < hyp_stack[0].output.size(); sent_id++){
                unsigned int w_id = 0;
                float w_prob = -FLT_MAX;
                if(t + 1 == batch_local.len_src[sent_id] || hyp_stack[0].output[sent_id].back() == d.dict_set_trg.end_id_word){
                    w_id = d.dict_set_trg.end_id_word;
                    w_prob = output[sent_id][d.dict_set_trg.end_id_word];
                }else{
                    for(unsigned int vocab_id = 0; vocab_id < output[sent_id].size(); vocab_id++){
                        if(vocab_id == d.dict_set_trg.end_id_word){
                            continue;
                        }
                        if(output[sent_id][vocab_id] > w_prob){
                            w_id = vocab_id;
                            w_prob = output[sent_id][vocab_id];
                        }
                    }
                }
                hyp_stack[0].prob[sent_id] += w_prob;
                hyp_stack[0].output[sent_id].push_back(w_id);
            }
            // end check
            unsigned int num_end = 0;
            for(unsigned int sent_id = 0; sent_id < hyp_stack[0].output.size(); sent_id++){
                if(hyp_stack[0].output[sent_id].back() == d.dict_set_trg.end_id_word){
                    num_end++;
                }
            }
            if(num_end == hyp_stack[0].output.size()){
                break;
            }
        }
    }
    
    void greedy_decode_ensemble(const batch& batch_local, std::vector<hypothesis<encoder_decoder_base>>& hyp_stack, std::vector<std::unique_ptr<encoder_decoder_base> > &encdec, dicts &d, const s2s_options &opts){
        dynet::ComputationGraph cg;
        hyp_stack.resize(1);
        hyp_stack[0].init(batch_local, d, encdec, cg);
        for (unsigned int t = 1; t < encdec[0]->slen; ++t) {
            std::vector<std::vector<float> > output(batch_local.batch_size(), std::vector<float>(d.dict_set_trg.d_word.size(), 0.f));
            // for each model
            for(unsigned int model_id = 0; model_id < encdec.size(); model_id++){
                dynet::Expression i_out_t = encdec[model_id]->decoder_output(cg, hyp_stack[0].prev(), t);
                dynet::Expression pred_out = softmax(i_out_t);
                std::vector<dynet::Tensor> results_out = cg.incremental_forward(pred_out).batch_elems();
                // for each sentence
                for(unsigned int sent_id = 0; sent_id < results_out.size(); sent_id++){
                    hyp_stack[0].dist_out[model_id][sent_id].push_back(as_vector(results_out.at(sent_id)));
                    for(unsigned int vocab_id = 0; vocab_id < hyp_stack[0].dist_out[model_id][sent_id][t-1].size(); vocab_id++){
                        output[sent_id][vocab_id] += log(hyp_stack[0].dist_out[model_id][sent_id][t-1][vocab_id]);
                    }
                }
            }
            // for each sentence
            for(unsigned int sent_id = 0; sent_id < output.size(); sent_id++){
                for(unsigned int vocab_id = 0; vocab_id < output[sent_id].size(); vocab_id++){
                    output[sent_id][vocab_id] /= (float)(encdec.size());
                }
            }
            // output vocab
            for(unsigned int sent_id = 0; sent_id < hyp_stack[0].output.size(); sent_id++){
                unsigned int w_id = 0;
                float w_prob = -FLT_MAX;
                if(t + 1 == batch_local.len_src[sent_id] || hyp_stack[0].output[sent_id].back() == d.dict_set_trg.end_id_word){
                    w_id = d.dict_set_trg.end_id_word;
                    w_prob = output[sent_id][d.dict_set_trg.end_id_word];
                }else{
                    for(unsigned int vocab_id = 0; vocab_id < output[sent_id].size(); vocab_id++){
                        if(vocab_id == d.dict_set_trg.end_id_word){
                            continue;
                        }
                        if(output[sent_id][vocab_id] > w_prob){
                            w_id = vocab_id;
                            w_prob = output[sent_id][vocab_id];
                        }
                    }
                }
                hyp_stack[0].prob[sent_id] += w_prob;
                hyp_stack[0].output[sent_id].push_back(w_id);
            }
            // end check
            unsigned int num_end = 0;
            for(unsigned int sent_id = 0; sent_id < hyp_stack[0].output.size(); sent_id++){
                if(hyp_stack[0].output[sent_id].back() == d.dict_set_trg.end_id_word){
                    num_end++;
                }
            }
            if(num_end == hyp_stack[0].output.size()){
                break;
            }
        }
    }
    
    void greedy_decode_ensemble(const batch& batch_local, std::vector<hypothesis<encoder_decoder_slahan>>& hyp_stack, std::vector<std::unique_ptr<encoder_decoder_slahan> > &encdec, dicts &d, const s2s_options &opts){
        dynet::ComputationGraph cg;
        hyp_stack.resize(1);
        hyp_stack[0].init(batch_local, d, encdec, cg);
        std::vector<std::vector<dynet::Expression> > h_att_self_all(encdec.size());
        for(unsigned int model_id = 0; model_id < encdec.size(); model_id++){
            h_att_self_all[model_id] = encdec[model_id]->attention(cg, batch_local);
            encdec[model_id]->recursive_attention(cg, h_att_self_all[model_id]);
        }
        for (unsigned int t = 1; t < encdec[0]->slen; ++t) {
            std::vector<std::vector<float> > output(batch_local.batch_size(), std::vector<float>(d.dict_set_trg.d_word.size(), 0.f));
            // for each model
            for(unsigned int model_id = 0; model_id < encdec.size(); model_id++){
                dynet::Expression i_out_t = encdec[model_id]->decoder_output(cg, t, hyp_stack[0].prev());
                dynet::Expression pred_out = softmax(i_out_t);
                std::vector<dynet::Tensor> results_out = cg.incremental_forward(pred_out).batch_elems();
                dynet::Expression pred_att = softmax(h_att_self_all[model_id][t]);
                std::vector<dynet::Tensor> results_att = cg.incremental_forward(pred_att).batch_elems();
                // for each sentence
                for(unsigned int sent_id = 0; sent_id < results_out.size(); sent_id++){
                    hyp_stack[0].dist_att[model_id][sent_id].push_back(as_vector(results_att.at(sent_id)));
                    hyp_stack[0].dist_out[model_id][sent_id].push_back(as_vector(results_out.at(sent_id)));
                    for(unsigned int vocab_id = 0; vocab_id < hyp_stack[0].dist_out[model_id][sent_id][t-1].size(); vocab_id++){
                        output[sent_id][vocab_id] += log(hyp_stack[0].dist_out[model_id][sent_id][t-1][vocab_id]);
                    }
                }
            }
            // for each sentence
            for(unsigned int sent_id = 0; sent_id < output.size(); sent_id++){
                for(unsigned int vocab_id = 0; vocab_id < output[sent_id].size(); vocab_id++){
                    output[sent_id][vocab_id] /= (float)(encdec.size());
                }
            }
            // output vocab
            for(unsigned int sent_id = 0; sent_id < hyp_stack[0].output.size(); sent_id++){
                unsigned int w_id = 0;
                float w_prob = -FLT_MAX;
                if(t + 1 == batch_local.len_src[sent_id] || hyp_stack[0].output[sent_id].back() == d.dict_set_trg.end_id_word){
                    w_id = d.dict_set_trg.end_id_word;
                    w_prob = output[sent_id][d.dict_set_trg.end_id_word];
                }else{
                    for(unsigned int vocab_id = 0; vocab_id < output[sent_id].size(); vocab_id++){
                        if(vocab_id == d.dict_set_trg.end_id_word){
                            continue;
                        }
                        if(output[sent_id][vocab_id] > w_prob){
                            w_id = vocab_id;
                            w_prob = output[sent_id][vocab_id];
                        }
                    }
                }
                hyp_stack[0].prob[sent_id] += w_prob;
                hyp_stack[0].output[sent_id].push_back(w_id);
            }
            // end check
            unsigned int num_end = 0;
            for(unsigned int sent_id = 0; sent_id < hyp_stack[0].output.size(); sent_id++){
                if(hyp_stack[0].output[sent_id].back() == d.dict_set_trg.end_id_word){
                    num_end++;
                }
            }
            if(num_end == hyp_stack[0].output.size()){
                break;
            }
        }
    }
  
    void greedy_decode_ensemble(const batch& batch_local, std::vector<hypothesis<encoder_decoder_lstm>>& hyp_stack, std::vector<std::unique_ptr<encoder_decoder_lstm> > &encdec, dicts &d, const s2s_options &opts){
        dynet::ComputationGraph cg;
        hyp_stack.resize(1);
        hyp_stack[0].init(batch_local, d, encdec, cg);
        for (unsigned int t = 1; t < encdec[0]->slen; ++t) {
            std::vector<std::vector<float> > output(batch_local.batch_size(), std::vector<float>(d.dict_set_trg.d_word.size(), 0.f));
            // for each model
            for(unsigned int model_id = 0; model_id < encdec.size(); model_id++){
                std::vector<dynet::real> bit_features = encdec[model_id]->bit_features(t, hyp_stack[0].prevs, batch_local.align[t]);
                dynet::Expression i_out_t = encdec[model_id]->decoder_output(cg, t, batch_local.align[t], bit_features);
                dynet::Expression pred_out = softmax(i_out_t);
                std::vector<dynet::Tensor> results_out = cg.incremental_forward(pred_out).batch_elems();
                // for each sentence
                for(unsigned int sent_id = 0; sent_id < results_out.size(); sent_id++){
                    hyp_stack[0].dist_out[model_id][sent_id].push_back(as_vector(results_out.at(sent_id)));
                    for(unsigned int vocab_id = 0; vocab_id < hyp_stack[0].dist_out[model_id][sent_id][t-1].size(); vocab_id++){
                        output[sent_id][vocab_id] += log(hyp_stack[0].dist_out[model_id][sent_id][t-1][vocab_id]);
                    }
                }
            }
            // for each sentence
            for(unsigned int sent_id = 0; sent_id < output.size(); sent_id++){
                for(unsigned int vocab_id = 0; vocab_id < output[sent_id].size(); vocab_id++){
                    output[sent_id][vocab_id] /= (float)(encdec.size());
                }
            }
            // output vocab
            for(unsigned int sent_id = 0; sent_id < hyp_stack[0].output.size(); sent_id++){
                unsigned int w_id = 0;
                float w_prob = -FLT_MAX;
                if(t + 1 == batch_local.len_src[sent_id] || hyp_stack[0].output[sent_id].back() == d.dict_set_trg.end_id_word){
                    w_id = d.dict_set_trg.end_id_word;
                    w_prob = output[sent_id][d.dict_set_trg.end_id_word];
                }else{
                    for(unsigned int vocab_id = 0; vocab_id < output[sent_id].size(); vocab_id++){
                        if(vocab_id == d.dict_set_trg.end_id_word){
                            continue;
                        }
                        if(output[sent_id][vocab_id] > w_prob){
                            w_id = vocab_id;
                            w_prob = output[sent_id][vocab_id];
                        }
                    }
                }
                hyp_stack[0].prob[sent_id] += w_prob;
                hyp_stack[0].output[sent_id].push_back(w_id);
            }
            hyp_stack[0].update_prevs();
            // end check
            unsigned int num_end = 0;
            for(unsigned int sent_id = 0; sent_id < hyp_stack[0].output.size(); sent_id++){
                if(hyp_stack[0].output[sent_id].back() == d.dict_set_trg.end_id_word){
                    num_end++;
                }
            }
            if(num_end == hyp_stack[0].output.size()){
                break;
            }
        }
    }
    
    void greedy_decode_ensemble(const batch& batch_local, std::vector<hypothesis<tagger>>& hyp_stack, std::vector<std::unique_ptr<tagger> > &encdec, dicts &d, const s2s_options &opts){
        dynet::ComputationGraph cg;
        hyp_stack.resize(1);
        hyp_stack[0].init(batch_local, d, encdec, cg);
        for (unsigned int t = 0; t < encdec[0]->slen; ++t) {
            std::vector<std::vector<float> > output(batch_local.batch_size(), std::vector<float>(d.dict_set_trg.d_word.size(), 0.f));
            // for each model
            for(unsigned int model_id = 0; model_id < encdec.size(); model_id++){
                dynet::Expression i_out_t = encdec[model_id]->output(cg, t);
                dynet::Expression pred_out = softmax(i_out_t);
                std::vector<dynet::Tensor> results_out = cg.incremental_forward(pred_out).batch_elems();
                // for each sentence
                for(unsigned int sent_id = 0; sent_id < results_out.size(); sent_id++){
                    hyp_stack[0].dist_out[model_id][sent_id].push_back(as_vector(results_out.at(sent_id)));
                    for(unsigned int vocab_id = 0; vocab_id < hyp_stack[0].dist_out[model_id][sent_id][t].size(); vocab_id++){
                        output[sent_id][vocab_id] += log(hyp_stack[0].dist_out[model_id][sent_id][t][vocab_id]);
                    }
                }
            }
            // for each sentence
            for(unsigned int sent_id = 0; sent_id < output.size(); sent_id++){
                for(unsigned int vocab_id = 0; vocab_id < output[sent_id].size(); vocab_id++){
                    output[sent_id][vocab_id] /= (float)(encdec.size());
                }
            }
            // output vocab
            for(unsigned int sent_id = 0; sent_id < hyp_stack[0].output.size(); sent_id++){
                unsigned int w_id = 0;
                float w_prob = -FLT_MAX;
                if(t == 0){
                    w_id = d.dict_set_trg.start_id_word;
                    w_prob = output[sent_id][d.dict_set_trg.start_id_word];
                }else if(t == batch_local.len_src[sent_id] - 1 || hyp_stack[0].output[sent_id].back() == d.dict_set_trg.end_id_word){
                    w_id = d.dict_set_trg.end_id_word;
                    w_prob = output[sent_id][d.dict_set_trg.end_id_word];
                }else{
                    for(unsigned int vocab_id = 0; vocab_id < output[sent_id].size(); vocab_id++){
                        if(vocab_id == d.dict_set_trg.end_id_word){
                            continue;
                        }
                        if(output[sent_id][vocab_id] > w_prob){
                            w_id = vocab_id;
                            w_prob = output[sent_id][vocab_id];
                        }
                    }
                }
                hyp_stack[0].prob[sent_id] += w_prob;
                if(t == 0){
                    hyp_stack[0].output[sent_id][0] = w_id;
                }else{
                    hyp_stack[0].output[sent_id].push_back(w_id);
                }
            }
        }
    }

    template<class T_EncDec>
    void greedy_decode_ensemble(const batch& batch_local, std::vector<hypothesis<T_EncDec>>& hyp_stack, std::unique_ptr<T_EncDec>& encdec, dicts &d, const s2s_options &opts){
        std::vector<std::unique_ptr<T_EncDec> > encdec_vec;
        encdec_vec.push_back(std::move(encdec));
        greedy_decode_ensemble(batch_local, hyp_stack, encdec_vec, d, opts);
        encdec = std::move(encdec_vec[0]);
    }

};

#endif // INCLUDE_GUARD_DECODER_GREEDY_HPP

