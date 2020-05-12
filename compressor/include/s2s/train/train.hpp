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

#include <boost/serialization/vector.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options.hpp>

#include "s2s/dynet/dict.h"
#include "s2s/nn/attn.hpp"
#include "s2s/nn/base.hpp"
#include "s2s/nn/lstm.hpp"
#include "s2s/nn/tagger.hpp"
#include "s2s/learner/attn.hpp"
#include "s2s/learner/base.hpp"
#include "s2s/learner/slahan.hpp"
#include "s2s/learner/lstm.hpp"
#include "s2s/learner/tagger.hpp"
#include "s2s/decode/greedy.hpp"
#include "s2s/decode/beam.hpp"
#include "s2s/decode/print.hpp"
#include "s2s/decode/hypothesis.hpp"
#include "s2s/corpus/corpora.hpp"
#include "s2s/corpus/comp.hpp"
#include "s2s/corpus/options.hpp"
#include "s2s/corpus/batch.hpp"

#include "word2vec/readbin.hpp"

#ifndef INCLUDE_GUARD_S2S_TRAIN_HPP
#define INCLUDE_GUARD_S2S_TRAIN_HPP

namespace s2s {
    

    template<class T_EncDec, class T_Learner>
    void train(const s2s_options &opts){
        s2s::dicts dicts;
        s2s::parallel_corpus para_corp_train(opts, dicts);
        s2s::parallel_corpus para_corp_dev(opts, dicts);
        dicts.set(opts);
        para_corp_train.load_src(opts.srcfile);
        para_corp_train.load_trg(opts.trgfile);
        para_corp_train.load_check();
        para_corp_dev.load_src(opts.srcvalfile);
        para_corp_dev.load_trg(opts.trgvalfile);
        para_corp_dev.load_check();
        if(opts.guided_alignment == true){
            para_corp_train.load_align(opts.alignfile);
            para_corp_train.load_check_with_align();
            para_corp_dev.load_align(opts.alignvalfile);
            para_corp_dev.load_check_with_align();
        }
        dicts.save(opts);
        dynet::ParameterCollection model;
        std::unique_ptr<T_EncDec> encdec(new T_EncDec(model, opts, dicts));
        std::unique_ptr<T_Learner> lr(new T_Learner(model, opts, encdec));
        // loading pre-trained vectors
        if(opts.src_pre_train_file != ""){
            // vocabularies are restricted to already existing words.
            encdec->read_pre_train_src(opts.src_pre_train_file, dicts.dict_set_src.d_word);
        }
        // train
        lr->prep_first_epoch();
        while(lr->epoch < opts.epochs){
            para_corp_train.sort_para_sent(opts.sort_sent_type_train, opts.max_batch_train, opts.src_tok_lim_train, opts.trg_tok_lim_train);
            para_corp_train.set_para_batch_order(opts.max_batch_train, opts.src_tok_lim_train, opts.trg_tok_lim_train, opts.batch_type_train);
            para_corp_train.shuffle_batch(opts.shuffle_batch_type_train);
            batch batch_train(opts, dicts, opts.elmo_hdf5_files);
            batch batch_dev(opts, dicts, opts.elmo_hdf5_dev_files);
            while(para_corp_train.next_batch_para(batch_train)){
                batch_train.drop_word();
                // train one batch
                lr->train_batch_local(batch_train);
            }
            std::cerr << std::endl;
            // dev validation
            std::cerr << "dev" << std::endl;
            para_corp_dev.sort_para_sent(opts.sort_sent_type_pred, opts.max_batch_pred, opts.src_tok_lim_pred, opts.trg_tok_lim_pred);
            para_corp_dev.set_para_batch_order(opts.max_batch_pred, opts.src_tok_lim_pred, opts.trg_tok_lim_pred, opts.batch_type_pred);
            lr->current_loss = 0.0;
            while(para_corp_dev.next_batch_para(batch_dev)){
                // validate one batch
                lr->current_loss += lr->dev_batch_local(batch_dev) * (float)(batch_dev.batch_size());
            }
            std::cerr << "current loss: " << lr->current_loss << ", previous_loss: " << lr->prev_loss << std::endl;
            std::cerr << "dev_decode" << std::endl;
            para_corp_train.reset_index();
            para_corp_dev.reset_index();
            // dev prediction
            std::vector<std::vector<unsigned int> > str_sents(para_corp_dev.src.size());
            while(para_corp_dev.next_batch_para(batch_dev)){
                std::vector<hypothesis<T_EncDec>> hyp_stack(1);
                if(opts.decoder_type == "greedy"){
                    s2s::greedy_decode_ensemble(batch_dev, hyp_stack, encdec, dicts, opts);
                }else if(opts.decoder_type == "beam"){
                    s2s::beam_decode_ensemble(batch_dev, hyp_stack, encdec, dicts, opts);
                }else{
                    std::cerr << "Decoder does not exist !"<< std::endl;
                    assert(false);
                }
                for(unsigned int batch_sent_id = 0; batch_sent_id < batch_dev.batch_size(); batch_sent_id++){
                    str_sents[batch_dev.sent_id.at(batch_sent_id)] = hyp_stack[0].output[batch_sent_id];
                }
            }
            // dev output
            std::string dev_out_path = opts.rootdir + "/dev_" + std::to_string(lr->epoch + 1) + ".txt";
            print_sents(dev_out_path, str_sents, dicts);
            // reset for next epoch
            para_corp_dev.reset_index();
            //encdec->enable_dropout();
            // preparation for next epoch
            lr->prep_next_epoch();
        }
    }

};

#endif
