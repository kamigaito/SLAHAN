#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/expr.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <type_traits>
#include <time.h>

#include <boost/program_options.hpp>

#include "s2s/dynet/dict.h"
#include "s2s/nn/slahan.hpp"
#include "s2s/corpus/batch.hpp"
#include "s2s/learner/learner.hpp"

#ifndef INCLUDE_GUARD_LEARNER_HISAN_HPP
#define INCLUDE_GUARD_LEARNER_HISAN_HPP

namespace s2s {

class slahan_learner : public learner<encoder_decoder_slahan> {

public:

    slahan_learner(dynet::ParameterCollection& model, const s2s_options& opts, std::unique_ptr<encoder_decoder_slahan>& encdec) : learner(model, opts, encdec) {}

    float train_batch_local(const batch& batch_local){
        return encdec_batch_local(true, batch_local);
    }
    
    float dev_batch_local(const batch& batch_local){
        return encdec_batch_local(false, batch_local);
    }
    
    float encdec_batch_local(const bool is_train, const batch& batch_local){
        auto chrono_start = std::chrono::system_clock::now();
        unsigned int batch_size = batch_local.batch_size();
        float loss_att = 0.0;
        float loss_out = 0.0;
        // enable dropout on training
        if(is_train == true){
            encdec->enable_dropout();
        // disable dropout on prediction
        }else{
            encdec->disable_dropout();
        }
        // print status
        std::cerr << "batch_size: " << batch_size;
        std::cerr << ",\tsource length: " << batch_local.sent_word.size();
        std::cerr << ",\ttarget length: " << batch_local.sent_label.size();
        // network construction
        dynet::ComputationGraph cg;
        std::vector<dynet::Expression> errs_att;
        std::vector<dynet::Expression> errs_out;
        encdec->encoder(batch_local, cg);
        std::vector<dynet::Expression> h_att_self = encdec->attention(cg, batch_local);
        encdec->recursive_attention(cg, h_att_self);
        for (unsigned int t = 1; t < batch_local.sent_label.size(); ++t) {
            if(opts.guided_alignment == true && opts.self_attn_type == "self"){
                for(unsigned int batch_id = 0; batch_id < batch_local.batch_size(); batch_id++){
                    if(batch_local.align.at(t).at(batch_id) >= batch_local.sent_label.size()){
                        std::cerr << "Dependency range is wrong." << std::endl;
                        assert(false);
                    }
                }
                dynet::Expression i_err_att = pickneglogsoftmax(h_att_self[t], batch_local.align.at(t));
                errs_att.push_back(i_err_att);
            }
            dynet::Expression i_out_t = encdec->decoder_output(cg, t, batch_local.sent_label[t-1]);
            dynet::Expression i_err_out = pickneglogsoftmax(i_out_t, batch_local.sent_label[t]);
            errs_out.push_back(i_err_out);
        }
        dynet::Expression i_nerr_out = sum_batches(sum(errs_out)) / (float)(batch_size);
        loss_out = as_scalar(cg.forward(i_nerr_out));
        dynet::Expression i_nerr_all;
        if(opts.guided_alignment == true && opts.self_attn_type == "self"){
            dynet::Expression i_nerr_att = sum_batches(sum(errs_att)) / (float)(batch_size);
            loss_att = as_scalar(cg.incremental_forward(i_nerr_att));
            i_nerr_all = i_nerr_out + align_w * i_nerr_att;
        }else{
            i_nerr_all = i_nerr_out;
        }
        float loss_all = as_scalar(cg.incremental_forward(i_nerr_all));
        std::cerr << ",\tentire loss: " << loss_all;
        std::cerr << ",\toutput loss: " << loss_out;
        std::cerr << ",\tattention loss: " << loss_att;
        // dropout setting
        if(is_train == true){
            cg.backward(i_nerr_all);
            //cg.print_graphviz();
            trainer->update();
        }
        auto chrono_end = std::chrono::system_clock::now();
        auto time_used = (double)std::chrono::duration_cast<std::chrono::milliseconds>(chrono_end - chrono_start).count() / (double)1000;
        std::cerr << ",\ttime: " << time_used << " [s]" << std::endl;
        std::cerr << "[epoch=" << (epoch+1) << " lr=" << trainer->learning_rate << " align_w=" << align_w << " clips=" << trainer->clips_since_status << " updates=" << trainer->updates_since_status << "] " << std::endl;
        return loss_out;
    }

};

};

#endif

