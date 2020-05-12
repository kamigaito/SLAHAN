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
#include "s2s/nn/slahan.hpp"
#include "s2s/nn/lstm.hpp"
#include "s2s/nn/tagger.hpp"
#include "s2s/decode/greedy.hpp"
#include "s2s/decode/beam.hpp"
#include "s2s/decode/print.hpp"
#include "s2s/decode/hypothesis.hpp"
#include "s2s/corpus/corpora.hpp"
#include "s2s/corpus/comp.hpp"
#include "s2s/corpus/options.hpp"
#include "s2s/corpus/batch.hpp"

#include "word2vec/readbin.hpp"

#ifndef INCLUDE_GUARD_S2S_PREDICT_HPP
#define INCLUDE_GUARD_S2S_PREDICT_HPP

namespace s2s {

    template<class T_EncDec>
    void predict(const s2s_options &opts){
        std::vector<std::string> vec_rootdir;
        boost::algorithm::split_regex(vec_rootdir, opts.rootdir, boost::regex(","));
        std::vector<std::string> vec_modelfile;
        boost::algorithm::split_regex(vec_modelfile, opts.modelfile, boost::regex(","));
        if(vec_rootdir.size() != vec_modelfile.size()){
            assert(false);
        }
        std::vector<s2s_options> vec_opts(vec_rootdir.size(),opts);
        for(unsigned int i=0; i < vec_opts.size(); i++){
            std::cerr << "Loading from " << vec_rootdir[i] << "/options.txt ..." << std::endl;
            std::ifstream in(vec_rootdir[i] + "/options.txt");
            boost::archive::text_iarchive ia(in);
            ia >> vec_opts[i];
            in.close();
            vec_opts[i].rootdir = vec_rootdir[i];
            vec_opts[i].modelfile = vec_modelfile[i];
            boost::algorithm::split_regex(vec_opts[i].elmo_hdf5_files, opts.elmo_hdf5_files_str, boost::regex(","));
        }
        s2s::dicts dicts;
        dicts.load(vec_opts[0]);
        std::vector<dynet::ParameterCollection> model(vec_opts.size());
        std::vector<std::unique_ptr<T_EncDec> > encdec;
        for(unsigned int i=0; i < vec_opts.size(); i++){
            // load model
            encdec.push_back(std::unique_ptr<T_EncDec>(new T_EncDec(model[i], vec_opts[i], dicts)));
            std::cerr << "Loading from " << vec_opts[i].modelfile << std::endl;
            dynet::TextFileLoader l(vec_opts[i].modelfile);
            l.populate(model[i]);
            encdec[i]->disable_dropout();
        }
        // predict
        s2s::monoling_corpus mono_corp(vec_opts[0], dicts);
        mono_corp.load_src(opts.srcfile);
        if(opts.guided_alignment == true){
            mono_corp.load_align(opts.alignfile);
            mono_corp.load_check_with_align();
        }
        batch batch_pred(vec_opts[0], dicts, vec_opts[0].elmo_hdf5_files);
        mono_corp.sort_mono_sent(opts.sort_sent_type_pred);
        mono_corp.set_mono_batch_order(opts.max_batch_pred, opts.src_tok_lim_pred, opts.batch_type_pred);
        //
        std::vector<float> str_probs(mono_corp.src.size());
        std::vector<std::vector<unsigned int> > str_sents(mono_corp.src.size());
        std::vector<std::vector<std::vector<std::vector<float> > > > str_dists_out(encdec.size(), std::vector<std::vector<std::vector<float> > >(mono_corp.src.size()));
        std::vector<std::vector<std::vector<std::vector<float> > > > str_dists_att(encdec.size(), std::vector<std::vector<std::vector<float> > >(mono_corp.src.size()));
        //
        while(mono_corp.next_mono_batch(batch_pred)){
            std::vector<hypothesis<T_EncDec>> hyp_stack(1);
            if(opts.decoder_type == "greedy"){
                s2s::greedy_decode_ensemble(batch_pred, hyp_stack, encdec, dicts, vec_opts[0]);
            }else if(opts.decoder_type == "beam"){
                s2s::beam_decode_ensemble(batch_pred, hyp_stack, encdec, dicts, vec_opts[0]);
            }else{
                std::cerr << "Decoder does not exist !"<< std::endl;
                assert(false);
            }
            // store state
            for(unsigned int batch_sent_id = 0; batch_sent_id < batch_pred.batch_size(); batch_sent_id++){
                str_probs[batch_pred.sent_id.at(batch_sent_id)] = hyp_stack[0].prob[batch_sent_id];
                str_sents[batch_pred.sent_id.at(batch_sent_id)] = hyp_stack[0].output[batch_sent_id];
            }
            for(unsigned int model_id = 0; model_id < encdec.size(); model_id++){
                for(unsigned int batch_sent_id = 0; batch_sent_id < batch_pred.batch_size(); batch_sent_id++){
                    str_dists_out[model_id][batch_pred.sent_id.at(batch_sent_id)] = hyp_stack[0].dist_out[model_id][batch_sent_id];
                    str_dists_att[model_id][batch_pred.sent_id.at(batch_sent_id)] = hyp_stack[0].dist_att[model_id][batch_sent_id];
                }
            }
            for(unsigned int batch_sent_id = 0; batch_sent_id < batch_pred.batch_size(); batch_sent_id++){
                // debug
                std::cerr << batch_pred.sent_id.at(batch_sent_id) << std::endl;
            }
        }
        // dev output
        std::string path_sents = opts.trgfile + ".sents";
        print_sents(path_sents, str_sents, dicts);
        std::string path_probs = opts.trgfile + ".probs";
        print_probs(path_probs, str_probs);
        for(unsigned int model_id = 0; model_id < encdec.size(); model_id++){
            std::string path_dists_att = opts.trgfile + ".dists.att." + std::to_string(model_id);
            print_dists_att(path_dists_att, model_id, str_dists_att);
            std::string path_dists_out = opts.trgfile + ".dists.out." + std::to_string(model_id);
            print_dists_out(path_dists_out, model_id, str_dists_out);
        }
    }

};

#endif

