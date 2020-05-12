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
#include "s2s/decode/hypothesis.hpp"
#include "s2s/corpus/comp.hpp"
#include "s2s/corpus/dicts.hpp"

#ifndef INCLUDE_GUARD_DECODER_BEAM_HPP
#define INCLUDE_GUARD_DECODER_BEAM_HPP

namespace s2s {
    

    void beam_decode_ensemble(const batch& batch_local, std::vector<hypothesis<encoder_decoder_attn>>& hyp_stack, std::vector<std::unique_ptr<encoder_decoder_attn> > &encdec, dicts &d, const s2s_options &opts){
        if(batch_local.batch_size() > 1){
            std::cerr << "batch size should be 1 on beam decoding." << std::endl;
            assert(false);
        }
        dynet::ComputationGraph cg;
        CompareHyp<encoder_decoder_attn> comp;
        hyp_stack.resize(1);
        hyp_stack[0].init(batch_local, d, encdec, cg);
        for (unsigned int t = 1; t < encdec[0]->slen; ++t) {
            std::vector<hypothesis<encoder_decoder_attn>> hyp_stack_local;
            for(hypothesis<encoder_decoder_attn> hyp : hyp_stack){
                std::vector<hypothesis<encoder_decoder_attn>> hyp_stack_tmp;
                std::vector<float> output(d.dict_set_trg.d_word.size(), 0.f);
                for(unsigned int model_id = 0; model_id < encdec.size(); model_id++){
                    dynet::Expression i_att_t = encdec[model_id]->decoder_attention(cg, t, hyp.prev(), hyp.pointer_prev[model_id]);
                    dynet::Expression i_out_t = encdec[model_id]->decoder_output(cg, t, i_att_t);
                    hyp.pointer_prev[model_id] = encdec[model_id]->dec_builder.state();
                    dynet::Expression pred_att = softmax(i_att_t);
                    std::vector<dynet::Tensor> results_att = cg.incremental_forward(pred_att).batch_elems();
                    dynet::Expression pred_out = softmax(i_out_t);
                    std::vector<dynet::Tensor> results_out = cg.incremental_forward(pred_out).batch_elems();
                    hyp.dist_att[model_id][0].push_back(as_vector(results_att.at(0)));
                    hyp.dist_out[model_id][0].push_back(as_vector(results_out.at(0)));
                    for(unsigned int vocab_id = 0; vocab_id < hyp.dist_out[model_id][0][t-1].size(); vocab_id++){
                        output[vocab_id] += log(hyp.dist_out[model_id][0][t-1][vocab_id]);
                    }
                }
                for(unsigned int vocab_id = 0; vocab_id < output.size(); vocab_id++){
                    output[vocab_id] /= (float)(encdec.size());
                }
                // output
                if(t == batch_local.len_src[0] - 1 || hyp.output[0].back() == d.dict_set_trg.end_id_word){
                  hypothesis<encoder_decoder_attn> hyp_tmp = hyp;
                  hyp_tmp.output[0].push_back(d.dict_set_trg.end_id_word);
                  hyp_tmp.prob[0] += output[d.dict_set_trg.end_id_word];
                  hyp_stack_tmp.push_back(hyp_tmp);
                }else{
                  for(unsigned int vocab_id = 0; vocab_id < output.size(); vocab_id++){
                    if(vocab_id == d.dict_set_trg.end_id_word){
                        continue;
                    }
                    hypothesis<encoder_decoder_attn> hyp_tmp = hyp;
                    hyp_tmp.output[0].push_back(vocab_id);
                    hyp_tmp.prob[0] += output[vocab_id];
                    hyp_stack_tmp.push_back(hyp_tmp);
                  }
                }
                sort(hyp_stack_tmp.begin(), hyp_stack_tmp.end(), comp);
                for(unsigned int hyp_id = 0; hyp_id < opts.beam_size && hyp_id < hyp_stack_tmp.size(); hyp_id++){
                    hyp_stack_local.push_back(hyp_stack_tmp[hyp_id]);
                }
            }
            hyp_stack.clear();
            sort(hyp_stack_local.begin(), hyp_stack_local.end(), comp);
            for(unsigned int hyp_id = 0; hyp_id < opts.beam_size && hyp_id < hyp_stack_local.size(); hyp_id++){
                hyp_stack.push_back(hyp_stack_local[hyp_id]);
            }
            // end check
            unsigned int num_end = 0;
            for(hypothesis<encoder_decoder_attn> hyp : hyp_stack){
                if(hyp.output[0].back() == d.dict_set_trg.end_id_word){
                    num_end++;
                }
            }
            if(num_end == hyp_stack.size()){
                break;
            }
        }
    }
    
    void beam_decode_ensemble(const batch& batch_local, std::vector<hypothesis<encoder_decoder_base>>& hyp_stack, std::vector<std::unique_ptr<encoder_decoder_base> > &encdec, dicts &d, const s2s_options &opts){
      if(batch_local.batch_size() > 1){
        std::cerr << "batch size should be 1 on beam decoding." << std::endl;
        assert(false);
      }
      dynet::ComputationGraph cg;
      CompareHyp<encoder_decoder_base> comp;
      hyp_stack.resize(1);
      hyp_stack[0].init(batch_local, d, encdec, cg);
      for (unsigned int t = 1; t < encdec[0]->slen; ++t) {
        std::vector<hypothesis<encoder_decoder_base>> hyp_stack_local;
        for(hypothesis<encoder_decoder_base> hyp : hyp_stack){
          std::vector<hypothesis<encoder_decoder_base>> hyp_stack_tmp;
          std::vector<float> output(d.dict_set_trg.d_word.size(), 0.f);
          for(unsigned int model_id = 0; model_id < encdec.size(); model_id++){
            dynet::Expression i_out_t = encdec[model_id]->decoder_output(cg, hyp.prev(), t, hyp.pointer_prev[model_id]);
            hyp.pointer_prev[model_id] = encdec[model_id]->dec_builder.state();
            dynet::Expression pred_out = softmax(i_out_t);
            std::vector<dynet::Tensor> results_out = cg.incremental_forward(pred_out).batch_elems();
            hyp.dist_out[model_id][0].push_back(as_vector(results_out.at(0)));
            for(unsigned int vocab_id = 0; vocab_id < hyp.dist_out[model_id][0][t-1].size(); vocab_id++){
              output[vocab_id] += log(hyp.dist_out[model_id][0][t-1][vocab_id]);
            }
          }
          for(unsigned int vocab_id = 0; vocab_id < output.size(); vocab_id++){
            output[vocab_id] /= (float)(encdec.size());
          }
          // output
          if(t + 1 == batch_local.len_src[0] || hyp.output[0].back() == d.dict_set_trg.end_id_word){
            hypothesis<encoder_decoder_base> hyp_tmp = hyp;
            hyp_tmp.output[0].push_back(d.dict_set_trg.end_id_word);
            hyp_tmp.prob[0] += output[d.dict_set_trg.end_id_word];
            hyp_stack_tmp.push_back(hyp_tmp);
          }else{
            for(unsigned int vocab_id = 0; vocab_id < output.size(); vocab_id++){
              if(vocab_id == d.dict_set_trg.end_id_word){
                  continue;
              }
              hypothesis<encoder_decoder_base> hyp_tmp = hyp;
              hyp_tmp.output[0].push_back(vocab_id);
              hyp_tmp.prob[0] += output[vocab_id];
              hyp_stack_tmp.push_back(hyp_tmp);
            }
          }
          sort(hyp_stack_tmp.begin(), hyp_stack_tmp.end(), comp);
          for(unsigned int hyp_id = 0; hyp_id < opts.beam_size && hyp_id < hyp_stack_tmp.size(); hyp_id++){
            hyp_stack_local.push_back(hyp_stack_tmp[hyp_id]);
          }
        }
        hyp_stack.clear();
        sort(hyp_stack_local.begin(), hyp_stack_local.end(), comp);
        for(unsigned int hyp_id = 0; hyp_id < opts.beam_size && hyp_id < hyp_stack_local.size(); hyp_id++){
          hyp_stack.push_back(hyp_stack_local[hyp_id]);
        }
      }
    }

    void beam_decode_ensemble(const batch& batch_local, std::vector<hypothesis<encoder_decoder_slahan>>& hyp_stack, std::vector<std::unique_ptr<encoder_decoder_slahan> > &encdec, dicts &d, const s2s_options &opts){
        if(batch_local.batch_size() > 1){
            std::cerr << "batch size should be 1 on beam decoding." << std::endl;
            assert(false);
        }
        dynet::ComputationGraph cg;
        CompareHyp<encoder_decoder_slahan> comp;
        hyp_stack.resize(1);
        hyp_stack[0].init(batch_local, d, encdec, cg);
        std::vector<std::vector<dynet::Expression> > h_att_self_all(encdec.size());
        for(unsigned int model_id = 0; model_id < encdec.size(); model_id++){
          h_att_self_all[model_id] = encdec[model_id]->attention(cg, batch_local);
          encdec[model_id]->recursive_attention(cg, h_att_self_all[model_id]);
        }
        for (unsigned int t = 1; t < encdec[0]->slen; ++t) {
            std::vector<hypothesis<encoder_decoder_slahan>> hyp_stack_local;
            for(hypothesis<encoder_decoder_slahan> hyp : hyp_stack){
                std::vector<hypothesis<encoder_decoder_slahan>> hyp_stack_tmp;
                std::vector<float> output(d.dict_set_trg.d_word.size(), 0.f);
                for(unsigned int model_id = 0; model_id < encdec.size(); model_id++){
                    dynet::Expression i_out_t = encdec[model_id]->decoder_output(cg, t, hyp.prev(), hyp.pointer_prev[model_id]);
                    hyp.pointer_prev[model_id] = encdec[model_id]->dec_builder.state();
                    dynet::Expression pred_out = softmax(i_out_t);
                    std::vector<dynet::Tensor> results_out = cg.incremental_forward(pred_out).batch_elems();
                    dynet::Expression pred_att = softmax(h_att_self_all[model_id][t]);
                    std::vector<dynet::Tensor> results_att = cg.incremental_forward(pred_att).batch_elems();
                    hyp.dist_att[model_id][0].push_back(as_vector(results_att.at(0)));
                    hyp.dist_out[model_id][0].push_back(as_vector(results_out.at(0)));
                    for(unsigned int vocab_id = 0; vocab_id < hyp.dist_out[model_id][0][t-1].size(); vocab_id++){
                        output[vocab_id] += log(hyp.dist_out[model_id][0][t-1][vocab_id]);
                    }
                }
                for(unsigned int vocab_id = 0; vocab_id < output.size(); vocab_id++){
                    output[vocab_id] /= (float)(encdec.size());
                }
                // for debug
                // std::cerr <<  logsumexp<float>(output) << std::endl;
                if(t == batch_local.len_src[0] - 1 || hyp.output[0].back() == d.dict_set_trg.end_id_word){
                  hypothesis<encoder_decoder_slahan> hyp_tmp = hyp;
                  hyp_tmp.output[0].push_back(d.dict_set_trg.end_id_word);
                  hyp_tmp.prob[0] += output[d.dict_set_trg.end_id_word];
                  hyp_stack_tmp.push_back(hyp_tmp);
                }else{
                  for(unsigned int vocab_id = 0; vocab_id < output.size(); vocab_id++){
                    if(vocab_id == d.dict_set_trg.end_id_word){
                        continue;
                    }
                    hypothesis<encoder_decoder_slahan> hyp_tmp = hyp;
                    hyp_tmp.output[0].push_back(vocab_id);
                    hyp_tmp.prob[0] += output[vocab_id];
                    hyp_stack_tmp.push_back(hyp_tmp);
                  }
                }
                sort(hyp_stack_tmp.begin(), hyp_stack_tmp.end(), comp);
                for(unsigned int hyp_id = 0; hyp_id < opts.beam_size && hyp_id < hyp_stack_tmp.size(); hyp_id++){
                    hyp_stack_local.push_back(hyp_stack_tmp[hyp_id]);
                }
            }
            hyp_stack.clear();
            sort(hyp_stack_local.begin(), hyp_stack_local.end(), comp);
            for(unsigned int hyp_id = 0; hyp_id < opts.beam_size && hyp_id < hyp_stack_local.size(); hyp_id++){
                hyp_stack.push_back(hyp_stack_local[hyp_id]);
            }
            // end check
            unsigned int num_end = 0;
            for(hypothesis<encoder_decoder_slahan> hyp : hyp_stack){
                if(hyp.output[0].back() == d.dict_set_trg.end_id_word){
                    num_end++;
                }
            }
            if(num_end == hyp_stack.size()){
                break;
            }
        }
    }

    void beam_decode_ensemble(const batch& batch_local, std::vector<hypothesis<encoder_decoder_lstm>>& hyp_stack, std::vector<std::unique_ptr<encoder_decoder_lstm> > &encdec, dicts &d, const s2s_options &opts){
        if(batch_local.batch_size() > 1){
            std::cerr << "batch size should be 1 on beam decoding." << std::endl;
            assert(false);
        }
        dynet::ComputationGraph cg;
        CompareHyp<encoder_decoder_lstm> comp;
        hyp_stack.resize(1);
        hyp_stack[0].init(batch_local, d, encdec, cg);
        for (unsigned int t = 1; t < encdec[0]->slen; ++t) {
            std::vector<hypothesis<encoder_decoder_lstm>> hyp_stack_local;
            for(hypothesis<encoder_decoder_lstm> hyp : hyp_stack){
                std::vector<hypothesis<encoder_decoder_lstm>> hyp_stack_tmp;
                std::vector<float> output(d.dict_set_trg.d_word.size(), 0.f);
                for(unsigned int model_id = 0; model_id < encdec.size(); model_id++){
                    std::vector<dynet::real> bit_features = encdec[model_id]->bit_features(t, hyp.prevs, batch_local.align[t]);
                    dynet::Expression i_out_t = encdec[model_id]->decoder_output(cg, t, batch_local.align[t], bit_features);
                    //hyp.i_hidden[model_id].push_back(i_out_t);
                    hyp.pointer_prev[model_id] = encdec[model_id]->dec_builder.state();
                    dynet::Expression pred_out = softmax(i_out_t);
                    std::vector<dynet::Tensor> results_out = cg.incremental_forward(pred_out).batch_elems();
                    hyp.dist_out[model_id][0].push_back(as_vector(results_out.at(0)));
                    for(unsigned int vocab_id = 0; vocab_id < hyp.dist_out[model_id][0][t-1].size(); vocab_id++){
                        output[vocab_id] += log(hyp.dist_out[model_id][0][t-1][vocab_id]);
                    }
                }
                for(unsigned int vocab_id = 0; vocab_id < output.size(); vocab_id++){
                    output[vocab_id] /= (float)(encdec.size());
                }
                // output
                if(t + 1 == batch_local.len_src[0] || hyp.output[0].back() == d.dict_set_trg.end_id_word){
                    hypothesis<encoder_decoder_lstm> hyp_tmp = hyp;
                    hyp_tmp.output[0].push_back(d.dict_set_trg.end_id_word);
                    hyp_tmp.prob[0] += output[d.dict_set_trg.end_id_word];
                    hyp_stack_tmp.push_back(hyp_tmp);
                }else{
                    for(unsigned int vocab_id = 0; vocab_id < output.size(); vocab_id++){
                        if(vocab_id == d.dict_set_trg.end_id_word){
                            continue;
                        }
                        hypothesis<encoder_decoder_lstm> hyp_tmp = hyp;
                        hyp_tmp.output[0].push_back(vocab_id);
                        hyp_tmp.prob[0] += output[vocab_id];
                        hyp_stack_tmp.push_back(hyp_tmp);
                    }
                }
                //
                sort(hyp_stack_tmp.begin(), hyp_stack_tmp.end(), comp);
                for(unsigned int hyp_id = 0; hyp_id < opts.beam_size && hyp_id < hyp_stack_tmp.size(); hyp_id++){
                    hyp_stack_local.push_back(hyp_stack_tmp[hyp_id]);
                }
            }
            hyp_stack.clear();
            sort(hyp_stack_local.begin(), hyp_stack_local.end(), comp);
            for(unsigned int hyp_id = 0; hyp_id < opts.beam_size && hyp_id < hyp_stack_local.size(); hyp_id++){
                hyp_stack_local[hyp_id].update_prevs();
                hyp_stack.push_back(hyp_stack_local[hyp_id]);
            }
        }
    }

    void beam_decode_ensemble(const batch& batch_local, std::vector<hypothesis<tagger>>& hyp_stack, std::vector<std::unique_ptr<tagger> > &encdec, dicts &d, const s2s_options &opts){
        assert(false);
    }

    template<class T_EncDec>
    void beam_decode_ensemble(const batch& batch_local, std::vector<hypothesis<T_EncDec>>& hyp_stack, std::unique_ptr<T_EncDec>& encdec, dicts &d, const s2s_options &opts){
        std::vector<std::unique_ptr<T_EncDec> > encdec_vec;
        encdec_vec.push_back(std::move(encdec));
        beam_decode_ensemble(batch_local, hyp_stack, encdec_vec, d, opts);
        encdec = std::move(encdec_vec[0]);
    }

};

#endif // INCLUDE_GUARD_DECODER_BEAM_HPP

