#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options.hpp>
#include <boost/regex.hpp>
#include <boost/algorithm/string/regex.hpp>

#ifndef INCLUDE_GUARD_S2S_OPTIONS_HPP
#define INCLUDE_GUARD_S2S_OPTIONS_HPP

namespace s2s {

    struct s2s_options {

    public:

        std::string mode;
        std::string rootdir;
        std::string srcfile;
        std::string trgfile;
        std::string srcvalfile;
        std::string trgvalfile;
        std::string alignfile;
        std::string alignvalfile;
        std::string save_file;
        std::string modelfile;
        std::string dict_prefix;
        std::string src_pre_train_file;
        std::string trg_pre_train_file;
        std::string elmo_hdf5_files_str;
        std::string elmo_hdf5_dev_files_str;
        std::string elmo_hdf5_layers_str;
        std::string elmo_hdf5_dims_str;

        unsigned int num_layers;
        unsigned int rnn_size;
        unsigned int char_num_layers;
        unsigned int char_rnn_size;
        unsigned int att_size;

        bool additional_output_layer;
        bool additional_connect_layer;
        bool clipping_enabled;
        bool flag_par_embed;
        bool flag_bit_feature;
        bool flag_bi_enc;
        bool selective_gate;

        std::string lookup_type;
        std::string context_type;
        std::string self_attn_type;
        std::string rec_attn_dep_str;
        std::vector<unsigned int> rec_attn_dep;

        unsigned int enc_word_vec_size;
        int enc_word_vocab_size;
        unsigned int enc_char_vec_size;
        int enc_char_vocab_size;
        std::string enc_feature_vec_size_str;
        std::string enc_feature_vocab_size_str;
        std::vector<unsigned int> enc_feature_vec_size;
        std::vector<int> enc_feature_vocab_size;
        std::vector<std::string> elmo_hdf5_files;
        std::vector<std::string> elmo_hdf5_dev_files;
        std::vector<unsigned int> elmo_hdf5_layers;
        std::vector<unsigned int> elmo_hdf5_dims;

        unsigned int dec_word_vec_size;
        int dec_word_vocab_size;
        unsigned int dec_char_vec_size;
        int dec_char_vocab_size;
        std::string dec_feature_vec_size_str;
        std::string dec_feature_vocab_size_str;
        std::vector<unsigned int> dec_feature_vec_size;
        std::vector<int> dec_feature_vocab_size;

        bool guided_alignment;
        float guided_alignment_weight;
        float guided_alignment_decay;
        float guided_output_weight;
        float guided_output_decay;
        unsigned int guided_alignment_start_epoch;
        unsigned int guided_alignment_decay_for_each;

        unsigned int epochs;
        unsigned int start_epoch;
        unsigned int decay_for_each;
        unsigned int sgd_start_epoch;
        unsigned int sgd_start_decay;
        unsigned int sgd_start_decay_for_each;

        std::string optim;
        bool lr_auto_decay;
        float learning_rate;
        float lr_decay;
        float sgd_start_learning_rate;
        float sgd_start_lr_decay;

        float dropout_rate_lstm_char;
        float dropout_rate_lstm_word;

        float clip_threshold;

        float drop_word_alpha;

        float dummy_flt_min;
        float bit_size_flt;
        unsigned int max_batch_train;
        unsigned int max_batch_pred;
        unsigned int max_length;
        unsigned int src_tok_lim_train;
        unsigned int trg_tok_lim_train;
        unsigned int src_tok_lim_pred;
        unsigned int trg_tok_lim_pred;
        unsigned int beam_size;
        std::string start_symbol;
        std::string keep_symbol;
        std::string delete_symbol;
        std::string end_symbol;
        std::string unk_symbol;
        std::string pad_symbol;
        std::string sort_sent_type_train;
        std::string batch_type_train;
        std::string shuffle_batch_type_train;
        std::string sort_sent_type_pred;
        std::string batch_type_pred;
        std::string shuffle_batch_type_pred;
        std::string decoder_type;
        unsigned int save_every;
        unsigned int print_every;
        unsigned int seed;

        s2s_options(){

            mode = "";
            rootdir = "";
            srcfile = "";
            trgfile = "";
            srcvalfile = "";
            trgvalfile = "";
            alignfile = "";
            alignvalfile = "";
            src_pre_train_file = "";
            trg_pre_train_file = "";
            elmo_hdf5_files_str = "";
            elmo_hdf5_dev_files_str = "";
            save_file = "save_";
            dict_prefix = "dict_";
            num_layers = 3;
            rnn_size = 200;
            char_num_layers = 2;
            char_rnn_size = 50;
            att_size = 200;
            char_num_layers = 1;
            char_rnn_size = 25;
            clipping_enabled = false;
            flag_par_embed = false;
            flag_bit_feature = false;
            flag_bi_enc = false;
            selective_gate = true;

            enc_word_vec_size = 150;
            enc_word_vocab_size = 20000;
            enc_char_vec_size = 25;
            enc_char_vocab_size = 20000;
            enc_feature_vec_size_str = "";
            enc_feature_vocab_size_str = "";
            
            dec_word_vec_size = 150;
            dec_word_vocab_size = 20000;
            dec_char_vec_size = 25;
            dec_char_vocab_size = 20000;
            dec_feature_vec_size_str = "";
            dec_feature_vocab_size_str = "";
            
            guided_alignment = false;
            guided_alignment_weight = 0.5;
            guided_alignment_decay = 0.9;
            guided_alignment_start_epoch = 5;
            guided_alignment_decay_for_each = 5;
            guided_output_weight = 1.0;
            guided_output_decay = 1.0;
            additional_output_layer = false;
            additional_connect_layer = false;
            lookup_type = "word";
            self_attn_type = "self";
            rec_attn_dep_str = "1";
            context_type = "both";
            epochs = 20;
            start_epoch = 5;
            decay_for_each = 5;
            sgd_start_epoch = UINT_MAX;
            sgd_start_decay = UINT_MAX;
            sgd_start_decay_for_each = UINT_MAX;
            lr_auto_decay = false;
            learning_rate = 1.0;
            lr_decay = 1.0;
            sgd_start_learning_rate = 0.1;
            sgd_start_lr_decay = 1.0;
            optim = "sgd";
            dropout_rate_lstm_char = 0.3;
            dropout_rate_lstm_word = 0.3;
            dummy_flt_min = -100000.0;
            bit_size_flt = 0.01;
            clip_threshold = 5.0;
            drop_word_alpha = 0.0;
            max_batch_train = 32;
            max_batch_pred = 32;
            max_length = 300;
            src_tok_lim_train = 500;
            trg_tok_lim_train = 500;
            src_tok_lim_pred = 500;
            trg_tok_lim_pred = 500;
            beam_size = 10;
            sort_sent_type_train = "same_length";
            batch_type_train = "default";
            shuffle_batch_type_train = "default";
            sort_sent_type_pred = "same_length";
            batch_type_pred = "default";
            shuffle_batch_type_pred = "default";
            decoder_type = "greedy";
            start_symbol = "<s>";
            keep_symbol = "1";
            delete_symbol = "0";
            end_symbol = "</s>";
            unk_symbol = "<unk>";
            pad_symbol = "<pad>";
            save_every = 1;
            print_every = 1;
            seed = 1;
        }

        private:
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version) {
            ar & save_file;
            ar & dict_prefix;
            ar & num_layers;
            ar & rnn_size;
            ar & char_num_layers;
            ar & char_rnn_size;
            ar & att_size;
            ar & additional_output_layer;
            ar & additional_connect_layer;
            ar & lookup_type;
            ar & context_type;
            ar & flag_par_embed;
            ar & flag_bit_feature;
            ar & flag_bi_enc;
            ar & selective_gate;
            ar & self_attn_type;
            ar & rec_attn_dep;
            ar & rec_attn_dep_str;
            ar & enc_word_vec_size;
            ar & enc_word_vocab_size;
            ar & enc_char_vec_size;
            ar & enc_char_vocab_size;
            ar & enc_feature_vec_size;
            ar & enc_feature_vocab_size;
            ar & enc_feature_vec_size_str;
            ar & enc_feature_vocab_size_str;
            ar & dec_word_vec_size;
            ar & dec_word_vocab_size;
            ar & dec_char_vec_size;
            ar & dec_char_vocab_size;
            ar & dec_feature_vec_size;
            ar & dec_feature_vocab_size;
            ar & dec_feature_vec_size_str; 
            ar & dec_feature_vocab_size_str;
            ar & elmo_hdf5_layers;
            ar & elmo_hdf5_dims;
            ar & elmo_hdf5_layers_str;
            ar & elmo_hdf5_dims_str;
            ar & guided_alignment;
            ar & dummy_flt_min;
            ar & bit_size_flt;
            ar & start_symbol;
            ar & keep_symbol;
            ar & delete_symbol;
            ar & end_symbol;
            ar & unk_symbol;
            ar & pad_symbol;
            ar & seed;
        }

    };

    void set_s2s_options(boost::program_options::options_description *bpo, s2s_options *opts) {
        namespace po = boost::program_options;
        bpo->add_options()
        ("dynet-mem", po::value<std::string>()->default_value("2000,2000,2000"), "dynet memory limit")
        ("dynet-weight-decay", po::value<float>()->default_value(0.0), "l2 decay")
        ("dynet-devices", po::value<std::string>()->default_value(""), "Specify the CPU/GPU devices that you want to use.")
        ("dynet-autobatch", po::value<int>()->default_value(0), "Turns on DyNet’s automatic operation batching capability.")
        ("mode", po::value<std::string>(&(opts->mode))->required(), "select from 'train', 'predict' or 'test'")
        ("rootdir", po::value<std::string>(&(opts->rootdir))->required(), "root directory path")
        ("srcfile", po::value<std::string>(&(opts->srcfile))->required(), "path to source side training file")
        ("trgfile", po::value<std::string>(&(opts->trgfile))->required(), "path to target side training file")
        ("srcvalfile", po::value<std::string>(&(opts->srcvalfile)), "path to source side development file")
        ("trgvalfile", po::value<std::string>(&(opts->trgvalfile)), "path to target side development file")
        ("alignfile", po::value<std::string>(&(opts->alignfile)), "path to alignment training file")
        ("alignvalfile", po::value<std::string>(&(opts->alignvalfile)), "path to alignment development file")
        ("src_pre_train_file", po::value<std::string>(&(opts->src_pre_train_file))->default_value(""), "path to pre-trained source side word vector file")
        ("trg_pre_train_file", po::value<std::string>(&(opts->trg_pre_train_file))->default_value(""), "path to pre-trained target side word vector file")
        ("modelfile", po::value<std::string>(&(opts->modelfile)), "path to trained model file")
        ("save_file_prefix", po::value<std::string>(&(opts->save_file))->default_value("save"), "prefix of model files")
        ("dict_prefix", po::value<std::string>(&(opts->dict_prefix))->default_value("dict_"), "prefix of dictionary files")
        ("num_layers", po::value<unsigned int>(&(opts->num_layers))->default_value(3), "number of LSTM layers")
        ("rnn_size", po::value<unsigned int>(&(opts->rnn_size))->default_value(256), "number of hidden dimensions")
        ("char_num_layers", po::value<unsigned int>(&(opts->char_num_layers))->default_value(2), "number of character level LSTM layers")
        ("char_rnn_size", po::value<unsigned int>(&(opts->char_rnn_size))->default_value(50), "number of hidden dimensions in character level LSTM")
        ("att_size", po::value<unsigned int>(&(opts->att_size))->default_value(256), "attention vector size")
        ("additional_output_layer", po::value<bool>(&(opts->additional_output_layer))->default_value(true), "Add an additional layer before the output layer. 1:Yes, 2:No")
        ("additional_connect_layer", po::value<bool>(&(opts->additional_connect_layer))->default_value(false), "Add an additional layer to the connection of the encoder and decoder.")
        ("lookup_type", po::value<std::string>(&(opts->lookup_type))->default_value("word"), "choose an embedding type. word : word embedding, char : character embedding, word+char : word and character embeddings, none: no embeddings")
        ("context_type", po::value<std::string>(&(opts->context_type))->default_value("both"), "choose a context type from 'both', 'parent' or 'child'.")
        ("flag_par_embed", po::value<bool>(&(opts->flag_par_embed))->default_value(false), "on/off the feeding of parent word embeddings")
        ("flag_bit_feature", po::value<bool>(&(opts->flag_bit_feature))->default_value(false), "on/off feeding of the bit features")
        ("flag_bi_enc", po::value<bool>(&(opts->flag_bi_enc))->default_value(true), "on/off the bidirectional encoding 1:on, 2:off")
        ("selective_gate", po::value<bool>(&(opts->selective_gate))->default_value(true), "on/off the selective gate 1:on, 2:off")
        ("self_attn_type", po::value<std::string>(&(opts->self_attn_type))->default_value("self"), "choose supervised attention method. given: dependency results are directory used as attention dicstibutions, self: attention distributions are learned by supervised manner, auto: attention distributions are automatically learned.")
        ("rec_attn_dep", po::value<std::string>(&(opts->rec_attn_dep_str))->default_value("1"), "depth of recursive attention e.g. 1,2,4")
        ("clipping_enabled", po::value<bool>(&(opts->clipping_enabled))->default_value(true), "on/off the gradient clipping. 1:on, 2:off")
        ("enc_word_vec_size", po::value<unsigned int>(&(opts->enc_word_vec_size))->default_value(256), "source side word vector size")
        ("enc_word_vocab_size", po::value<int>(&(opts->enc_word_vocab_size))->default_value(20000), "source side word vocabulary size -1: no limit")
        ("enc_char_vec_size", po::value<unsigned int>(&(opts->enc_char_vec_size))->default_value(256), "source side character vector size")
        ("enc_char_vocab_size", po::value<int>(&(opts->enc_char_vocab_size))->default_value(20000), "soruce side character vocabulary size -1: no limit")
        ("enc_feature_vec_size", po::value<std::string>(&(opts->enc_feature_vec_size_str))->default_value(""), "source side feature vector size. e.g. 10,10,10")
        ("enc_feature_vocab_size", po::value<std::string>(&(opts->enc_feature_vocab_size_str))->default_value(""), "source side feature vocabulary size. e.g. 50,50,50")
        ("elmo_hdf5_files", po::value<std::string>(&(opts->elmo_hdf5_files_str))->default_value(""), "ELMo style HDF5 file paths")
        ("elmo_hdf5_dev_files", po::value<std::string>(&(opts->elmo_hdf5_dev_files_str))->default_value(""), "ELMo style HDF5 file paths")
        ("elmo_hdf5_layers", po::value<std::string>(&(opts->elmo_hdf5_layers_str))->default_value(""), "layer sizes of ELMo style HDF5 files")
        ("elmo_hdf5_dims", po::value<std::string>(&(opts->elmo_hdf5_dims_str))->default_value(""), "dimension sizes of ELMo style HDF5 files")
        ("dec_word_vec_size", po::value<unsigned int>(&(opts->dec_word_vec_size)), "target side word vector size")
        ("dec_word_vocab_size", po::value<int>(&(opts->dec_word_vocab_size)), "target side word vocabulary size")
        ("dec_char_vec_size", po::value<unsigned int>(&(opts->dec_char_vec_size)), "target side character vector size")
        ("dec_char_vocab_size", po::value<int>(&(opts->dec_char_vocab_size)), "target side character vocabulary size")
        ("dec_feature_vec_size", po::value<std::string>(&(opts->dec_feature_vec_size_str))->default_value(""), "target side feature vector size. e.g. 10,10,10")
        ("dec_feature_vocab_size", po::value<std::string>(&(opts->dec_feature_vocab_size_str))->default_value(""), "target side feature vocabulary size. e.g. 50,50,50")
        ("guided_alignment", po::value<bool>(&(opts->guided_alignment))->default_value(false), "on/off of supervised attention")
        ("guided_alignment_weight", po::value<float>(&(opts->guided_alignment_weight))->default_value(0.3), "set lambda for the supervised attention")
        ("guided_alignment_decay", po::value<float>(&(opts->guided_alignment_decay))->default_value(0.5), "decay rate of lambda for the supervised attention")
        ("guided_alignment_start_epoch", po::value<unsigned int>(&(opts->guided_alignment_start_epoch))->default_value(5), "start epoch of lambda decaying in the supervised attention")
        ("guided_alignment_decay_for_each", po::value<unsigned int>(&(opts->guided_alignment_decay_for_each))->default_value(5), "decay interval in the supervised attention")
        ("guided_output_weight", po::value<float>(&(opts->guided_output_weight))->default_value(1.0), "output weight")
        ("guided_output_decay", po::value<float>(&(opts->guided_output_decay))->default_value(1.0), "decay")
        ("epochs", po::value<unsigned int>(&(opts->epochs))->default_value(20), "number of epochs")
        ("start_epoch", po::value<unsigned int>(&(opts->start_epoch))->default_value(5), "learning rate decaying is started from this epoch")
        ("decay_for_each", po::value<unsigned int>(&(opts->decay_for_each))->default_value(5), "learning rate is decayed for each these epochs")
        ("sgd_start_epoch", po::value<unsigned int>(&(opts->sgd_start_epoch))->default_value(UINT_MAX), "switching to sgd is started from this epoch")
        ("sgd_start_decay", po::value<unsigned int>(&(opts->sgd_start_decay))->default_value(UINT_MAX), "learning rate decaying of the switched sgd is started from this epoch")
        ("sgd_start_decay_for_each", po::value<unsigned int>(&(opts->sgd_start_decay_for_each))->default_value(UINT_MAX), "learning rate decaying of switched sgd is started from this epoch")
        ("optim", po::value<std::string>(&(opts->optim))->default_value("sgd"), "You can select the optimizer for training: sgd, momentum_sgd, adam, adagrad, adadelta")
        ("sgd_start_learning_rate", po::value<float>(&(opts->sgd_start_learning_rate))->default_value(0.1), "first learning rate of the switched sgd")
        ("sgd_start_lr_decay", po::value<float>(&(opts->sgd_start_lr_decay))->default_value(1.0), "decaying rate of the switched sgd")
        ("lr_auto_decay", po::value<bool>(&(opts->lr_auto_decay))->default_value(false), "on/off the automatic learning rate decaying")
        ("learning_rate", po::value<float>(&(opts->learning_rate))->default_value(0.1), "first learning rate")
        ("lr_decay", po::value<float>(&(opts->lr_decay))->default_value(1.0), "decaying rate for the learning rate")
        ("dropout_rate_lstm_word", po::value<float>(&(opts->dropout_rate_lstm_word))->default_value(0.3), "dropout rate of the word level lstm")
        ("dropout_rate_lstm_char", po::value<float>(&(opts->dropout_rate_lstm_char))->default_value(0.3), "dropout rate of the character level lstm")
        ("clip_threshold", po::value<float>(&(opts->clip_threshold))->default_value(5.0), "clipping threshold")
        ("drop_word_alpha", po::value<float>(&(opts->drop_word_alpha))->default_value(0.0), "word dropout ratio")
        ("dummy_flt_min", po::value<float>(&(opts->dummy_flt_min))->default_value(-FLT_MAX), "minimum value of the float")
        ("bit_size_flt", po::value<float>(&(opts->bit_size_flt))->default_value(0.01), "a value representing a bit which is true")
        ("max_batch_train", po::value<unsigned int>(&(opts->max_batch_train))->default_value(32), "maximum batch size in the training step")
        ("max_batch_pred", po::value<unsigned int>(&(opts->max_batch_pred))->default_value(1), "maximum batch size in the prediction step")
        ("max_length", po::value<unsigned int>(&(opts->max_length))->default_value(300), "maximum length in the prediction. # currently not implemented")
        ("start_symbol", po::value<std::string>(&(opts->start_symbol))->default_value("<s>"), "symbol of the start of sentences. default is <s>")
        ("keep_symbol", po::value<std::string>(&(opts->keep_symbol))->default_value("1"), "symbol of the kept tokens. default is 1")
        ("delete_symbol", po::value<std::string>(&(opts->delete_symbol))->default_value("0"), "symbol of the deleted tokens. default is 0")
        ("end_symbol", po::value<std::string>(&(opts->end_symbol))->default_value("</s>"), "symbol of the end of sentences. default is </s>")
        ("unk_symbol", po::value<std::string>(&(opts->unk_symbol))->default_value("<unk>"), "symbol of unknown words. default is <unk>")
        ("pad_symbol", po::value<std::string>(&(opts->pad_symbol))->default_value("<pad>"), "symbol of padding words. default is <pad>")
        ("src_tok_lim_train", po::value<unsigned int>(&(opts->src_tok_lim_train)), "source side token limit in the training step for each batch")
        ("trg_tok_lim_train", po::value<unsigned int>(&(opts->trg_tok_lim_train)), "token side token limit in the training step for each batch")
        ("src_tok_lim_pred", po::value<unsigned int>(&(opts->src_tok_lim_pred)), "source side token limit in the prediction step for each batch")
        ("trg_tok_lim_pred", po::value<unsigned int>(&(opts->trg_tok_lim_pred)), "target side token limit in the prediction step for each batch")
        ("beam_size", po::value<unsigned int>(&(opts->beam_size)), "beam size")
        ("sort_sent_type_train", po::value<std::string>(&(opts->sort_sent_type_train))->default_value("default"), "You can change the sentence order in the training step: default, random, sort_default, sort_random")
        ("batch_type_train", po::value<std::string>(&(opts->batch_type_train))->default_value("sent_length"), "You can set the batch type in the training step: default, sent_length")
        ("shuffle_batch_type_train", po::value<std::string>(&(opts->shuffle_batch_type_train))->default_value("default"), "You can change the batch order in the training step: default, random")
        ("sort_sent_type_pred", po::value<std::string>(&(opts->sort_sent_type_pred))->default_value("default"), "You can change the sentence order in the prediction step: default, random, sort_default, sort_random")
        ("batch_type_pred", po::value<std::string>(&(opts->batch_type_pred))->default_value("sent_length"), "You can set the batch type in the prediction step: default, sent_length")
        ("shuffle_batch_type_pred", po::value<std::string>(&(opts->shuffle_batch_type_pred))->default_value("default"), "You can change the batch order in the training step: default, random")
        ("decoder_type", po::value<std::string>(&(opts->decoder_type))->default_value("greedy"), "You can select the decoder types: greedy, beam")
        ("save_every", po::value<unsigned int>(&(opts->save_every))->default_value(1), "# currently not implemented")
        ("print_every", po::value<unsigned int>(&(opts->print_every))->default_value(1), "# currently not implemented")
        ("seed", po::value<unsigned int>(&(opts->seed))->default_value(0), "seed for random values. # currently not implemented.");
    }

    void add_s2s_options_train(const boost::program_options::variables_map &vm, s2s_options *opts){
        // ELMo files
        if(opts->elmo_hdf5_files_str != ""){
            // filename
            boost::algorithm::split_regex(opts->elmo_hdf5_files, opts->elmo_hdf5_files_str, boost::regex(","));
            // layer size
            std::vector<std::string> vec_elmo_hdf5_layers_str;
            boost::algorithm::split_regex(vec_elmo_hdf5_layers_str, opts->elmo_hdf5_layers_str, boost::regex(","));
            for(auto layer_size : vec_elmo_hdf5_layers_str){
                opts->elmo_hdf5_layers.push_back(std::stoi(layer_size));
            }
            // dimension size
            std::vector<std::string> vec_elmo_hdf5_dims_str;
            boost::algorithm::split_regex(vec_elmo_hdf5_dims_str, opts->elmo_hdf5_dims_str, boost::regex(","));
            for(auto dim_size : vec_elmo_hdf5_dims_str){
                opts->elmo_hdf5_dims.push_back(std::stoi(dim_size));
            }
            assert(opts->elmo_hdf5_files.size() == opts->elmo_hdf5_layers.size());
            assert(opts->elmo_hdf5_files.size() == opts->elmo_hdf5_dims.size());
        }
        // ELMo dev files
        if(opts->elmo_hdf5_dev_files_str != ""){
            // filename
            boost::algorithm::split_regex(opts->elmo_hdf5_dev_files, opts->elmo_hdf5_dev_files_str, boost::regex(","));
            assert(opts->elmo_hdf5_dev_files.size() == opts->elmo_hdf5_layers.size());
            assert(opts->elmo_hdf5_dev_files.size() == opts->elmo_hdf5_dims.size());
        }
        // features in encoder
        if(opts->enc_feature_vec_size_str != ""){
            std::vector<std::string> vec_str_enc_feature_vec_size;
            boost::algorithm::split_regex(vec_str_enc_feature_vec_size, opts->enc_feature_vec_size_str, boost::regex(","));
            for(auto feature_vec_size : vec_str_enc_feature_vec_size){
                opts->enc_feature_vec_size.push_back(std::stoi(feature_vec_size));
            }
        }
        if(opts->enc_feature_vocab_size_str != ""){
            std::vector<std::string> vec_str_enc_feature_vocab_size;
            boost::algorithm::split_regex(vec_str_enc_feature_vocab_size, opts->enc_feature_vocab_size_str, boost::regex(","));
            for(auto feature_vocab_size : vec_str_enc_feature_vocab_size){
                opts->enc_feature_vocab_size.push_back(std::stoi(feature_vocab_size));
            }
        }
        assert(opts->enc_feature_vocab_size.size() == opts->enc_feature_vec_size.size());
        // features in decoder
        if(opts->dec_feature_vec_size_str != ""){
            std::vector<std::string> vec_str_dec_feature_vec_size;
            boost::algorithm::split_regex(vec_str_dec_feature_vec_size, opts->dec_feature_vec_size_str, boost::regex(","));
            for(auto feature_vec_size : vec_str_dec_feature_vec_size){
                opts->dec_feature_vec_size.push_back(std::stoi(feature_vec_size));
            }
        }
        if(opts->dec_feature_vocab_size_str != ""){
            std::vector<std::string> vec_str_dec_feature_vocab_size;
            boost::algorithm::split_regex(vec_str_dec_feature_vocab_size, opts->dec_feature_vocab_size_str, boost::regex(","));
            for(auto feature_vocab_size : vec_str_dec_feature_vocab_size){
                opts->dec_feature_vocab_size.push_back(std::stoi(feature_vocab_size));
            }
        }
        assert(opts->dec_feature_vocab_size.size() == opts->dec_feature_vec_size.size());
        // set of orders
        assert(opts->rec_attn_dep_str != "");
        std::vector<std::string> vec_str_attn_dep;
        boost::algorithm::split_regex(vec_str_attn_dep, opts->rec_attn_dep_str, boost::regex(","));
        for(std::string rec_dep : vec_str_attn_dep){
            opts->rec_attn_dep.push_back(std::stoi(rec_dep));
        }
        std::sort(opts->rec_attn_dep.begin(), opts->rec_attn_dep.end());

    }

    void check_s2s_options_train(const boost::program_options::variables_map &vm, const s2s_options &opts){

    }

    void check_s2s_options_predict(const boost::program_options::variables_map &vm, const s2s_options &opts){

    }
};

#endif
