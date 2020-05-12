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

#ifndef INCLUDE_GUARD_DECODER_HYPOTHESIS_HPP
#define INCLUDE_GUARD_DECODER_HYPOTHESIS_HPP

namespace s2s {

    template<class T_EncDec>
    class hypothesis {

        public:

        std::vector<std::vector<unsigned int> > output; // sentence, time
        std::vector<std::vector<unsigned int> > prevs; // time, sentence
        std::vector<float> prob;
        std::vector<std::vector<std::vector<std::vector<float> > > > dist_att; // model, sentence, time, source
        std::vector<std::vector<std::vector<std::vector<float> > > > dist_out; // model, sentence, time, vocab
        std::vector<dynet::RNNPointer> pointer_prev;

        void init(const batch& batch_local, const dicts &d, std::vector<std::unique_ptr<T_EncDec> > &encdec, dynet::ComputationGraph& cg){
            //
            prob.resize(batch_local.batch_size());
            output = std::vector<std::vector<unsigned int > >(batch_local.batch_size(), std::vector<unsigned int>(1, d.dict_set_trg.start_id_word));
            prevs = std::vector<std::vector<unsigned int > >(1, std::vector<unsigned int>(batch_local.batch_size(), d.dict_set_trg.start_id_word));
            //
            pointer_prev.resize(encdec.size());
            //
            dist_att.resize(encdec.size());
            dist_out.resize(encdec.size());
            //
            for(unsigned int sent_id = 0; sent_id < batch_local.batch_size(); sent_id++){
                prob[sent_id] = 0.0;
            }
            for(unsigned int model_id = 0; model_id < encdec.size(); model_id++){
                dist_att[model_id].resize(batch_local.batch_size());
                dist_out[model_id].resize(batch_local.batch_size());
                encdec[model_id]->encoder(batch_local, cg);
                pointer_prev[model_id] = encdec[model_id]->dec_builder.state();
            }
        }

        std::vector<unsigned int> prev(){
            std::vector<unsigned int> prev;
            for(unsigned int i=0; i<output.size(); i++){
                prev.push_back(output[i].back());
            }
            return prev;
        }

        void update_prevs(){
            std::vector<unsigned int> prev(output.size());
            for(unsigned int sid = 0; sid < output.size(); sid++){
                prev[sid] = output[sid].back();
            }
            prevs.push_back(prev);
        }

        std::vector<std::string> print_sents(dicts& d){
            std::vector<std::string> str_sents;
            for(const auto sent : output){
                std::string str_sent = "";
                for(const auto wid : sent){
                    std::string word = d.dict_set_trg.d_word.convert(wid);
                    str_sent += word;
                    if(wid == d.dict_set_trg.end_id_word){
                        break;
                    }
                    str_sent += " ";
                }
                str_sents.push_back(str_sent);
            }
            return str_sents;
        }

    };
    
    template class hypothesis<encoder_decoder_attn>;
    template class hypothesis<encoder_decoder_base>;
    template class hypothesis<encoder_decoder_slahan>;
    template class hypothesis<encoder_decoder_lstm>;
    template class hypothesis<tagger>;

    template<class T_EncDec>
    struct CompareHyp {

        bool operator()(const hypothesis<T_EncDec>& first, const hypothesis<T_EncDec>& second) {
            if(first.prob.size() > 1 || second.prob.size() > 1){
                std::cerr << "Cannot compare." << std::endl;
                assert(false);
            }
            if(first.prob[0] > second.prob[0]){
                return true;
            }
            return false;
        }

    };
    
    template struct CompareHyp<encoder_decoder_attn>;
    template struct CompareHyp<encoder_decoder_base>;
    template struct CompareHyp<encoder_decoder_slahan>;
    template struct CompareHyp<encoder_decoder_lstm>;
    template struct CompareHyp<tagger>;

};

#endif

