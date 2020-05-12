#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/rnn.h"
#include "dynet/gru.h"
#include "dynet/lstm.h"
#include "dynet/fast-lstm.h"
#include "dynet/expr.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <type_traits>
#include <time.h>
#include <limits>

#include <boost/serialization/vector.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options.hpp>

#include "s2s/dynet/dict.h"
#include "s2s/corpus/corpora.hpp"
#include "s2s/corpus/comp.hpp"
#include "s2s/corpus/options.hpp"
#include "s2s/corpus/batch.hpp"

#ifndef INCLUDE_GUARD_S2S_TEST_ENCDEC_HPP
#define INCLUDE_GUARD_S2S_TEST_ENCDEC_HPP

namespace s2s {
    
    void test_encdec(s2s_options &opts, dicts &dicts, parallel_corpus &para_corp, std::unique_ptr<encoder_decoder_attn>& encdec){
        dynet::ParameterCollection model;
        dynet::Trainer* trainer;
        if(opts.optim == "sgd"){
            trainer = new dynet::SimpleSGDTrainer(model);
        }else if(opts.optim == "momentum_sgd"){
            trainer = new dynet::MomentumSGDTrainer(model);
        }else if(opts.optim == "adagrad"){
            trainer = new dynet::AdagradTrainer(model);
        }else if(opts.optim == "adadelta"){
            trainer = new dynet::AdadeltaTrainer(model);
        }else if(opts.optim == "adam"){
            trainer = new dynet::AdamTrainer(model);
        }else{
            std::cerr << "Trainer does not exist !"<< std::endl;
            assert(false);
        }
        float learning_rate = opts.learning_rate;
        trainer->learning_rate = learning_rate;
        trainer->clipping_enabled = opts.clipping_enabled;
        trainer->clip_threshold = opts.clip_threshold;
        unsigned int batch_cnt = 0;
        batch batch_local(opts, dicts);
        para_corp.reset_index();
        para_corp.set_para_batch_order(opts.max_batch_train, opts.src_tok_lim_train, opts.trg_tok_lim_train, opts.batch_type_train);
        while(para_corp.next_batch_para(batch_local)){
            // network construction
            dynet::ComputationGraph cg;
            std::vector<dynet::Expression> errs_att;
            std::vector<dynet::Expression> errs_out;
            encdec->encoder(batch_local, cg);
            // checking distributions of the higher order attention module
            for (unsigned int t = 1; t < batch_local.sent_label.size(); ++t) {
                dynet::Expression i_att_t = encdec->decoder_attention(cg, t, batch_local.sent_label[t-1]);
                if(opts.guided_alignment == true){
                    for(unsigned int i = 0; i < batch_local.align.at(t).size(); i++){
                        assert(0 <= batch_local.align.at(t).at(i) && batch_local.align.at(t).at(i) < batch_local.sent_word.size());
                    }
                    dynet::Expression i_err = pickneglogsoftmax(i_att_t, batch_local.align.at(t));
                    errs_att.push_back(i_err);
                }
                dynet::Expression i_out_t = encdec->decoder_output(cg, t, i_att_t);
                dynet::Expression i_err = pickneglogsoftmax(i_out_t, batch_local.sent_label[t]);
                errs_out.push_back(i_err);
            }
            dynet::Expression i_nerr_out = sum_batches(sum(errs_out)) / (float)(batch_local.batch_size());
            float loss_out = as_scalar(cg.forward(i_nerr_out));
            float loss_att = 0.f;
            dynet::Expression i_nerr_all;
            if(opts.guided_alignment == true){
                dynet::Expression i_nerr_att = sum_batches(sum(errs_att)) / (float)(batch_local.batch_size());
                loss_att = as_scalar(cg.incremental_forward(i_nerr_att));
                i_nerr_all = i_nerr_out + opts.guided_alignment_weight * i_nerr_att;
            }else{
                i_nerr_all = i_nerr_out;
            }
            float loss_all = as_scalar(cg.incremental_forward(i_nerr_all));
            // checking values
            {
                float inf = std::numeric_limits<float>::infinity();
                float nan = std::numeric_limits<float>::quiet_NaN();
                assert(-inf < loss_out && loss_out < inf);
                assert(-inf < loss_att && loss_att < inf);
                assert(-inf < loss_all && loss_all < inf);
                assert(loss_out != nan);
                assert(loss_att != nan);
                assert(loss_all != nan);
            }
            cg.backward(i_nerr_all);
            //cg.print_graphviz();
            trainer->update();
            std::cerr << "batch id: " << batch_cnt << "\tloss (all): " << loss_all << "\tloss (out): " << loss_out << "\tloss (att): " << loss_att << std::endl;
            batch_cnt++;
        }
        assert(batch_cnt > 0);
        // greedy decoding
        batch_cnt = 0;
        para_corp.reset_index();
        para_corp.set_para_batch_order(opts.max_batch_train, opts.src_tok_lim_train, opts.trg_tok_lim_train, opts.batch_type_train);
        while(para_corp.next_batch_para(batch_local)){
            std::vector<hypothesis<encoder_decoder_attn>> hyp_stack(1);
            s2s::greedy_decode_ensemble(batch_local, hyp_stack, encdec, dicts, opts);
            std::cerr << "batch id: " << batch_cnt << std::endl;
            assert(hyp_stack.size() == 1);
            batch_cnt++;
        }
        assert(batch_cnt > 0);
        // beam decoding
        batch_cnt = 0;
        opts.max_batch_train = 1;
        para_corp.reset_index();
        para_corp.set_para_batch_order(opts.max_batch_train, opts.src_tok_lim_train, opts.trg_tok_lim_train, opts.batch_type_train);
        opts.beam_size = 10;
        while(para_corp.next_batch_para(batch_local)){
            std::vector<hypothesis<encoder_decoder_attn>> hyp_stack(1);
            s2s::beam_decode_ensemble(batch_local, hyp_stack, encdec, dicts, opts);
            std::cerr << "batch id: " << batch_cnt << std::endl;
            if(batch_local.sent_word.size() >= 4 && hyp_stack.size() < opts.beam_size){
                assert(false);
            }
            if(hyp_stack.size() > opts.beam_size){
                assert(false);
            }
            batch_cnt++;
        }
        assert(batch_cnt > 0);
    }
    
    void test_encdec(s2s_options &opts, dicts &dicts, parallel_corpus &para_corp, std::unique_ptr<encoder_decoder_base>& encdec){
        dynet::ParameterCollection model;
        dynet::Trainer* trainer;
        if(opts.optim == "sgd"){
            trainer = new dynet::SimpleSGDTrainer(model);
        }else if(opts.optim == "momentum_sgd"){
            trainer = new dynet::MomentumSGDTrainer(model);
        }else if(opts.optim == "adagrad"){
            trainer = new dynet::AdagradTrainer(model);
        }else if(opts.optim == "adadelta"){
            trainer = new dynet::AdadeltaTrainer(model);
        }else if(opts.optim == "adam"){
            trainer = new dynet::AdamTrainer(model);
        }else{
            std::cerr << "Trainer does not exist !"<< std::endl;
            assert(false);
        }
        float learning_rate = opts.learning_rate;
        trainer->learning_rate = learning_rate;
        trainer->clipping_enabled = opts.clipping_enabled;
        trainer->clip_threshold = opts.clip_threshold;
        unsigned int batch_cnt = 0;
        batch batch_local(opts, dicts);
        para_corp.reset_index();
        para_corp.set_para_batch_order(opts.max_batch_train, opts.src_tok_lim_train, opts.trg_tok_lim_train, opts.batch_type_train);
        while(para_corp.next_batch_para(batch_local)){
            // network construction
            dynet::ComputationGraph cg;
            std::vector<dynet::Expression> errs_att;
            std::vector<dynet::Expression> errs_out;
            encdec->encoder(batch_local, cg);
            // checking distributions of the higher order attention module
            for (unsigned int t = 1; t < batch_local.sent_label.size(); ++t) {
                dynet::Expression i_out_t = encdec->decoder_output(cg, batch_local.sent_label[t-1], t);
                dynet::Expression i_err = pickneglogsoftmax(i_out_t, batch_local.sent_label[t]);
                errs_out.push_back(i_err);
            }
            dynet::Expression i_nerr_out = sum_batches(sum(errs_out)) / (float)(batch_local.batch_size());
            float loss_out = as_scalar(cg.forward(i_nerr_out));
            // checking values
            {
                float inf = std::numeric_limits<float>::infinity();
                float nan = std::numeric_limits<float>::quiet_NaN();
                assert(-inf < loss_out && loss_out < inf);
                assert(loss_out != nan);
            }
            cg.backward(i_nerr_out);
            //cg.print_graphviz();
            trainer->update();
            std::cerr << "batch id: " << batch_cnt << "\tloss (out): " << loss_out << std::endl;
            batch_cnt++;
        }
        assert(batch_cnt > 0);
        // greedy decoding
        batch_cnt = 0;
        para_corp.reset_index();
        para_corp.set_para_batch_order(opts.max_batch_train, opts.src_tok_lim_train, opts.trg_tok_lim_train, opts.batch_type_train);
        while(para_corp.next_batch_para(batch_local)){
            std::vector<hypothesis<encoder_decoder_base>> hyp_stack(1);
            s2s::greedy_decode_ensemble(batch_local, hyp_stack, encdec, dicts, opts);
            std::cerr << "batch id: " << batch_cnt << std::endl;
            assert(hyp_stack.size() == 1);
            batch_cnt++;
        }
        assert(batch_cnt > 0);
        // beam decoding
        batch_cnt = 0;
        opts.max_batch_train = 1;
        para_corp.reset_index();
        para_corp.set_para_batch_order(opts.max_batch_train, opts.src_tok_lim_train, opts.trg_tok_lim_train, opts.batch_type_train);
        opts.beam_size = 10;
        while(para_corp.next_batch_para(batch_local)){
            std::vector<hypothesis<encoder_decoder_base>> hyp_stack(1);
            s2s::beam_decode_ensemble(batch_local, hyp_stack, encdec, dicts, opts);
            std::cerr << "batch id: " << batch_cnt << std::endl;
            if(batch_local.sent_word.size() >= 4 && hyp_stack.size() < opts.beam_size){
                assert(false);
            }
            if(hyp_stack.size() > opts.beam_size){
                assert(false);
            }
            batch_cnt++;
        }
        assert(batch_cnt > 0);
    }
    
    void test_encdec(s2s_options &opts, dicts &dicts, parallel_corpus &para_corp, std::unique_ptr<encoder_decoder_slahan>& encdec){
        dynet::ParameterCollection model;
        dynet::Trainer* trainer;
        if(opts.optim == "sgd"){
            trainer = new dynet::SimpleSGDTrainer(model);
        }else if(opts.optim == "momentum_sgd"){
            trainer = new dynet::MomentumSGDTrainer(model);
        }else if(opts.optim == "adagrad"){
            trainer = new dynet::AdagradTrainer(model);
        }else if(opts.optim == "adadelta"){
            trainer = new dynet::AdadeltaTrainer(model);
        }else if(opts.optim == "adam"){
            trainer = new dynet::AdamTrainer(model);
        }else{
            std::cerr << "Trainer does not exist !"<< std::endl;
            assert(false);
        }
        float learning_rate = opts.learning_rate;
        trainer->learning_rate = learning_rate;
        trainer->clipping_enabled = opts.clipping_enabled;
        trainer->clip_threshold = opts.clip_threshold;
        unsigned int batch_cnt = 0;
        batch batch_local(opts, dicts);
        para_corp.reset_index();
        para_corp.set_para_batch_order(opts.max_batch_train, opts.src_tok_lim_train, opts.trg_tok_lim_train, opts.batch_type_train);
        while(para_corp.next_batch_para(batch_local)){
            // network construction
            dynet::ComputationGraph cg;
            std::vector<dynet::Expression> errs_att;
            std::vector<dynet::Expression> errs_out;
            encdec->encoder(batch_local, cg);
            std::vector<dynet::Expression> h_att_self = encdec->attention(cg, batch_local);
            encdec->recursive_attention(cg, h_att_self);
            // checking distributions of the higher order attention module
            if(opts.self_attn_type == "given"){
                std::vector<std::vector<std::vector<unsigned int> > > h_att_gold(opts.rec_attn_dep.back());
                h_att_gold[0] = batch_local.align;
                for(unsigned int d = 1; d < opts.rec_attn_dep.back(); d++){
                    h_att_gold[d].resize(batch_local.align.size());
                    std::vector<dynet::Expression> errs_att_local;
                    for (unsigned int t = 0; t < batch_local.align.size(); ++t) {
                        h_att_gold[d][t].resize(batch_local.batch_size());
                        for(unsigned int batch_id = 0; batch_id < batch_local.batch_size(); batch_id++){
                            h_att_gold[d][t][batch_id] =  h_att_gold[0][h_att_gold.at(d - 1).at(t).at(batch_id)][batch_id] ;
                        }
                        dynet::Expression i_err_att = pickneglogsoftmax(log(encdec->f_alpha_d[d][t]), h_att_gold[d][t]);
                        errs_att_local.push_back(i_err_att);
                    }
                    dynet::Expression i_nerr_att = sum_batches(sum(errs_att_local)) / (float)(batch_local.batch_size());
                    float loss_att = as_scalar(cg.incremental_forward(i_nerr_att));
                    assert(loss_att == 0.f);
                }
            }
            for (unsigned int t = 1; t < batch_local.sent_label.size(); ++t) {
                for(unsigned int batch_id = 0; batch_id < batch_local.batch_size(); batch_id++){
                    if(batch_local.align.at(t).at(batch_id) >= batch_local.sent_label.size()){
                        std::cerr << "Dependency range is wrong." << std::endl;
                        assert(false);
                    }
                }
                // calculate attention loss
                dynet::Expression i_err_att = pickneglogsoftmax(h_att_self[t], batch_local.align.at(t));
                errs_att.push_back(i_err_att);
                // calculate output loss
                dynet::Expression i_out_t = encdec->decoder_output(cg, t, batch_local.sent_label[t-1]);
                dynet::Expression i_err_out = pickneglogsoftmax(i_out_t, batch_local.sent_label[t]);
                errs_out.push_back(i_err_out);
            }
            dynet::Expression i_nerr_out = sum_batches(sum(errs_out)) / (float)(batch_local.batch_size());
            float loss_out = as_scalar(cg.forward(i_nerr_out));
            dynet::Expression i_nerr_att = sum_batches(sum(errs_att)) / (float)(batch_local.batch_size());
            float loss_att = as_scalar(cg.incremental_forward(i_nerr_att));
            dynet::Expression i_nerr_all = i_nerr_out + opts.guided_alignment_weight * i_nerr_att;
            float loss_all = as_scalar(cg.incremental_forward(i_nerr_all));
            // checking values
            {
                float inf = std::numeric_limits<float>::infinity();
                float nan = std::numeric_limits<float>::quiet_NaN();
                assert(-inf < loss_out && loss_out < inf);
                assert(-inf < loss_att && loss_att < inf);
                assert(-inf < loss_all && loss_all < inf);
                assert(loss_out != nan);
                assert(loss_att != nan);
                assert(loss_all != nan);
            }
            cg.backward(i_nerr_all);
            //cg.print_graphviz();
            trainer->update();
            std::cerr << "batch id: " << batch_cnt << "\tloss (all): " << loss_all << "\tloss (out): " << loss_out << "\tloss (att): " << loss_att << std::endl;
            batch_cnt++;
        }
        assert(batch_cnt > 0);
        // greedy decoding
        batch_cnt = 0;
        para_corp.reset_index();
        para_corp.set_para_batch_order(opts.max_batch_train, opts.src_tok_lim_train, opts.trg_tok_lim_train, opts.batch_type_train);
        while(para_corp.next_batch_para(batch_local)){
            std::vector<hypothesis<encoder_decoder_slahan>> hyp_stack(1);
            s2s::greedy_decode_ensemble(batch_local, hyp_stack, encdec, dicts, opts);
            std::cerr << "batch id: " << batch_cnt << std::endl;
            assert(hyp_stack.size() == 1);
            batch_cnt++;
        }
        assert(batch_cnt > 0);
        // beam decoding
        batch_cnt = 0;
        opts.max_batch_train = 1;
        para_corp.reset_index();
        para_corp.set_para_batch_order(opts.max_batch_train, opts.src_tok_lim_train, opts.trg_tok_lim_train, opts.batch_type_train);
        opts.beam_size = 10;
        while(para_corp.next_batch_para(batch_local)){
            std::vector<hypothesis<encoder_decoder_slahan>> hyp_stack(1);
            s2s::beam_decode_ensemble(batch_local, hyp_stack, encdec, dicts, opts);
            std::cerr << "batch id: " << batch_cnt << std::endl;
            if(batch_local.sent_word.size() >= 4 && hyp_stack.size() < opts.beam_size){
                assert(false);
            }
            if(hyp_stack.size() > opts.beam_size){
                assert(false);
            }
            batch_cnt++;
        }
        assert(batch_cnt > 0);
    }
    
    void test_encdec(s2s_options &opts, dicts &dicts, parallel_corpus &para_corp, std::unique_ptr<encoder_decoder_lstm>& encdec){
        dynet::ParameterCollection model;
        dynet::Trainer* trainer;
        if(opts.optim == "sgd"){
            trainer = new dynet::SimpleSGDTrainer(model);
        }else if(opts.optim == "momentum_sgd"){
            trainer = new dynet::MomentumSGDTrainer(model);
        }else if(opts.optim == "adagrad"){
            trainer = new dynet::AdagradTrainer(model);
        }else if(opts.optim == "adadelta"){
            trainer = new dynet::AdadeltaTrainer(model);
        }else if(opts.optim == "adam"){
            trainer = new dynet::AdamTrainer(model);
        }else{
            std::cerr << "Trainer does not exist !"<< std::endl;
            assert(false);
        }
        float learning_rate = opts.learning_rate;
        trainer->learning_rate = learning_rate;
        trainer->clipping_enabled = opts.clipping_enabled;
        trainer->clip_threshold = opts.clip_threshold;
        unsigned int batch_cnt = 0;
        batch batch_local(opts, dicts);
        para_corp.reset_index();
        para_corp.set_para_batch_order(opts.max_batch_train, opts.src_tok_lim_train, opts.trg_tok_lim_train, opts.batch_type_train);
        while(para_corp.next_batch_para(batch_local)){
            // network construction
            dynet::ComputationGraph cg;
            std::vector<dynet::Expression> errs_att;
            std::vector<dynet::Expression> errs_out;
            encdec->encoder(batch_local, cg);
            // checking distributions of the higher order attention module
            for (unsigned int t = 1; t < batch_local.sent_label.size(); ++t) {
                std::vector<dynet::real> bit_features = encdec->bit_features(t, batch_local.sent_label, batch_local.align[t]);
                dynet::Expression i_out_t = encdec->decoder_output(cg, t, batch_local.align[t], bit_features);
                dynet::Expression i_err = pickneglogsoftmax(i_out_t, batch_local.sent_label[t]);
                errs_out.push_back(i_err);
            }
            dynet::Expression i_nerr_out = sum_batches(sum(errs_out)) / (float)(batch_local.batch_size());
            float loss_out = as_scalar(cg.forward(i_nerr_out));
            // checking values
            {
                float inf = std::numeric_limits<float>::infinity();
                float nan = std::numeric_limits<float>::quiet_NaN();
                assert(-inf < loss_out && loss_out < inf);
                assert(loss_out != nan);
            }
            cg.backward(i_nerr_out);
            //cg.print_graphviz();
            trainer->update();
            std::cerr << "batch id: " << batch_cnt << "\tloss (out): " << loss_out << std::endl;
            batch_cnt++;
        }
        assert(batch_cnt > 0);
        // greedy decoding
        batch_cnt = 0;
        para_corp.reset_index();
        para_corp.set_para_batch_order(opts.max_batch_train, opts.src_tok_lim_train, opts.trg_tok_lim_train, opts.batch_type_train);
        while(para_corp.next_batch_para(batch_local)){
            std::vector<hypothesis<encoder_decoder_lstm>> hyp_stack(1);
            s2s::greedy_decode_ensemble(batch_local, hyp_stack, encdec, dicts, opts);
            std::cerr << "batch id: " << batch_cnt << std::endl;
            assert(hyp_stack.size() == 1);
            batch_cnt++;
        }
        assert(batch_cnt > 0);
        // beam decoding
        batch_cnt = 0;
        opts.max_batch_train = 1;
        para_corp.reset_index();
        para_corp.set_para_batch_order(opts.max_batch_train, opts.src_tok_lim_train, opts.trg_tok_lim_train, opts.batch_type_train);
        opts.beam_size = 10;
        while(para_corp.next_batch_para(batch_local)){
            std::vector<hypothesis<encoder_decoder_lstm>> hyp_stack(1);
            s2s::beam_decode_ensemble(batch_local, hyp_stack, encdec, dicts, opts);
            std::cerr << "batch id: " << batch_cnt << std::endl;
            if(batch_local.sent_word.size() >= 4 && hyp_stack.size() < opts.beam_size){
                assert(false);
            }
            if(hyp_stack.size() > opts.beam_size){
                assert(false);
            }
            batch_cnt++;
        }
        assert(batch_cnt > 0);
    }
    
    void test_encdec(s2s_options &opts, dicts &dicts, parallel_corpus &para_corp, std::unique_ptr<tagger>& encdec){
        dynet::ParameterCollection model;
        dynet::Trainer* trainer;
        if(opts.optim == "sgd"){
            trainer = new dynet::SimpleSGDTrainer(model);
        }else if(opts.optim == "momentum_sgd"){
            trainer = new dynet::MomentumSGDTrainer(model);
        }else if(opts.optim == "adagrad"){
            trainer = new dynet::AdagradTrainer(model);
        }else if(opts.optim == "adadelta"){
            trainer = new dynet::AdadeltaTrainer(model);
        }else if(opts.optim == "adam"){
            trainer = new dynet::AdamTrainer(model);
        }else{
            std::cerr << "Trainer does not exist !"<< std::endl;
            assert(false);
        }
        float learning_rate = opts.learning_rate;
        trainer->learning_rate = learning_rate;
        trainer->clipping_enabled = opts.clipping_enabled;
        trainer->clip_threshold = opts.clip_threshold;
        unsigned int batch_cnt = 0;
        batch batch_local(opts, dicts);
        para_corp.reset_index();
        para_corp.set_para_batch_order(opts.max_batch_train, opts.src_tok_lim_train, opts.trg_tok_lim_train, opts.batch_type_train);
        while(para_corp.next_batch_para(batch_local)){
            // network construction
            dynet::ComputationGraph cg;
            std::vector<dynet::Expression> errs_att;
            std::vector<dynet::Expression> errs_out;
            encdec->encoder(batch_local, cg);
            // checking distributions of the higher order attention module
            for (unsigned int t = 0; t < batch_local.sent_label.size(); ++t) {
                dynet::Expression i_out_t = encdec->output(cg, t);
                dynet::Expression i_err = pickneglogsoftmax(i_out_t, batch_local.sent_label[t]);
                errs_out.push_back(i_err);
            }
            dynet::Expression i_nerr_out = sum_batches(sum(errs_out)) / (float)(batch_local.batch_size());
            float loss_out = as_scalar(cg.forward(i_nerr_out));
            // checking values
            {
                float inf = std::numeric_limits<float>::infinity();
                float nan = std::numeric_limits<float>::quiet_NaN();
                assert(-inf < loss_out && loss_out < inf);
                assert(loss_out != nan);
            }
            cg.backward(i_nerr_out);
            //cg.print_graphviz();
            trainer->update();
            std::cerr << "batch id: " << batch_cnt << "\tloss (out): " << loss_out << std::endl;
            batch_cnt++;
        }
        assert(batch_cnt > 0);
        // greedy decoding
        batch_cnt = 0;
        para_corp.reset_index();
        para_corp.set_para_batch_order(opts.max_batch_train, opts.src_tok_lim_train, opts.trg_tok_lim_train, opts.batch_type_train);
        while(para_corp.next_batch_para(batch_local)){
            std::vector<hypothesis<tagger>> hyp_stack(1);
            s2s::greedy_decode_ensemble(batch_local, hyp_stack, encdec, dicts, opts);
            std::cerr << "batch id: " << batch_cnt << std::endl;
            assert(hyp_stack.size() == 1);
            batch_cnt++;
        }
        assert(batch_cnt > 0);
    }

};

#endif

