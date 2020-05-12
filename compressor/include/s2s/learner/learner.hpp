#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/rnn.h"
#include "dynet/gru.h"
#include "dynet/lstm.h"
#include "dynet/fast-lstm.h"
#include "dynet/expr.h"
#include "dynet/io.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <type_traits>
#include <time.h>

#include <boost/serialization/vector.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options.hpp>

#include "s2s/dynet/dict.h"
#include "s2s/corpus/corpora.hpp"
#include "s2s/corpus/comp.hpp"
#include "s2s/corpus/options.hpp"
#include "s2s/corpus/batch.hpp"

#include "word2vec/readbin.hpp"

#ifndef INCLUDE_GUARD_LEARNER_UTIL_HPP
#define INCLUDE_GUARD_LEARNER_UTIL_HPP

namespace s2s {

template<class T_EncDec>
class learner {
        
public:
        
    dynet::ParameterCollection& model;
    std::unique_ptr<T_EncDec>& encdec;
    dynet::Trainer* trainer;
    unsigned int epoch;
    float align_w;
    float prev_loss;
    float current_loss;
    const s2s_options& opts;

    learner(dynet::ParameterCollection& model, const s2s_options& opts, std::unique_ptr<T_EncDec>& encdec) : model(model), encdec(encdec), opts(opts){
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
        trainer->learning_rate = opts.learning_rate;
        trainer->clipping_enabled = opts.clipping_enabled;
        trainer->clip_threshold = opts.clip_threshold;
        epoch = 0;
        align_w = opts.guided_alignment_weight;
        prev_loss = FLT_MAX;
        current_loss = 0.0;
    }
   
    void save_model(){
        // save Model
        std::ofstream model_out(opts.rootdir + "/" + opts.save_file + "_" + std::to_string(epoch) + ".model");
        boost::archive::text_oarchive model_oa(model_out);
        model_oa << model << *(encdec.get());
        model_out.close();
    }

    std::string check_point_name(){
        return opts.rootdir + "/check_point";
    }

    std::string check_point_suffix(){
        std::string suffix = "";
        suffix = "epoch_" + std::to_string(epoch);
        return suffix;
    }

    void compress(const std::string file_path){
        assert(std::system(("gzip " + file_path).c_str()) == 0); 
    }

    void deflate(const std::string file_path){
        assert(std::system(("gzip -d " + file_path + ".gz").c_str()) == 0); 
    }

    void save_model(const std::string suffix){
        std::cerr << "Saving models..."<< std::endl;
        std::string model_path = opts.rootdir + "/" + opts.save_file + "_" + suffix + ".model";
        dynet::TextFileSaver s(model_path);
        s.save(model);
        compress(model_path);
    }

    void save_trainer(const std::string suffix){
        std::cerr << "Saving optimizer..."<< std::endl;
        std::string optimizer_path = opts.rootdir + "/" + opts.save_file + "_" + suffix + ".optimizer";
        std::ofstream s(optimizer_path);
        (*trainer).save(s);
        compress(optimizer_path);
    }

    void save_config() {
        std::ofstream os(check_point_name());
        os << epoch << "," << align_w << "," << prev_loss << std::endl;
        os.close();
        std::cerr << "Finished." << std::endl;
    }

    void check_point_save(){
        std::cerr << "Creating check point..." << std::endl;
        std::string suffix = check_point_suffix();
        // save Model
        save_model(suffix);
        // save Trainer
        save_trainer(suffix);
        // save states
        save_config();
    }

    // set states from a checkpoint file.
    void load_config(std::string last_line) {
        std::vector<std::string> str_cols;
        boost::algorithm::split_regex(str_cols, last_line, boost::regex(","));
        epoch = std::stoi(str_cols[0]);
        align_w = std::stof(str_cols[1]);
        prev_loss = std::stof(str_cols[2]);
    }

    void load_model(const std::string suffix){
        std::string model_path = opts.rootdir + "/" + opts.save_file + "_" + suffix + ".model";
        std::cerr << "Loading model from " << model_path << std::endl;
        deflate(model_path);
        dynet::TextFileLoader l(model_path);
        l.populate(model);
        compress(model_path);
    }

    void load_trainer(const std::string suffix){
        std::string optimizer_path = opts.rootdir + "/" + opts.save_file + "_" + suffix + ".optimizer";
        std::cerr << "Loading optimizer from " << optimizer_path << std::endl;
        deflate(optimizer_path);
        std::ifstream is(optimizer_path);
        (*trainer).populate(is);
        is.close();
        compress(optimizer_path);
    }

    void check_point_load(){
        std::ifstream in(check_point_name());
        if(in.is_open()){
            std::string last_line = "";
            std::string line = "";
            while(getline(in, line)){
                last_line = line;
            }
            in.close();
            // load states
            load_config(last_line);
            std::string suffix = check_point_suffix();
            // load model
            load_model(suffix);
            // load optimizer
            load_trainer(suffix);
        }
    }

    void prep_first_epoch() {
        check_point_load();
    }

    // prepare for next epoch
    void prep_next_epoch() {          
        // optimizer
        trainer->status();
        epoch++;
        if(opts.lr_auto_decay == true){
            if(current_loss > prev_loss){
                trainer->learning_rate *= opts.lr_decay;
            }
        }else{
            if(epoch >= opts.sgd_start_epoch){
                if(epoch > opts.sgd_start_decay){
                    if((epoch - opts.sgd_start_decay) % opts.sgd_start_decay_for_each == 0){
                        trainer->learning_rate *= opts.sgd_start_lr_decay;
                    }
                }else if(epoch == opts.sgd_start_epoch){
                    delete(trainer);
                    trainer = new dynet::SimpleSGDTrainer(model);
                    trainer->learning_rate = opts.sgd_start_learning_rate;
                    trainer->clipping_enabled = opts.clipping_enabled;
                    trainer->clip_threshold = opts.clip_threshold;
                }
            }else{
                if(epoch >= opts.start_epoch){
                    if(epoch > opts.start_epoch){
                        if((epoch - opts.start_epoch) % opts.decay_for_each == 0){
                            trainer->learning_rate *= opts.lr_decay;
                        }
                    }else if(epoch == opts.start_epoch){
                        trainer->learning_rate *= opts.lr_decay;
                    }
                }
            }
        }
        if(opts.guided_alignment == true){
            if(epoch > opts.guided_alignment_start_epoch){
                if((epoch - opts.guided_alignment_start_epoch) % opts.guided_alignment_decay_for_each == 0){
                    align_w *= opts.guided_alignment_decay;
                }
            }else if(epoch == opts.guided_alignment_start_epoch){
                align_w *= opts.guided_alignment_decay;
            }
        }
        prev_loss = current_loss;
        check_point_save();
    }
};

};

#endif

