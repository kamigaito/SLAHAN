#include <iostream>
#include <fstream>
#include <sstream>
#include <type_traits>
#include <string>

#ifndef INCLUDE_GUARD_PRINT_HPP
#define INCLUDE_GUARD_PRINT_HPP

namespace s2s {

    void print_probs(const std::string file_path, const std::vector<float>& str_probs) {
        std::string delim = "\n";
        std::ofstream out_probs(file_path);
        for(float prob : str_probs){
            std::string line = std::to_string(prob);
            line += delim;
            out_probs << line;
        }
        out_probs.close();
    }

    void print_sents(const std::string file_path, const std::vector<std::vector<unsigned int> >& str_sents, dicts& d) {
        std::string delim = "\n";
        std::string sep = " ";
        std::ofstream out_sents(file_path);
        for(auto sent : str_sents){
            std::string line = "";
            for(auto wid : sent){
                std::string word = d.dict_set_trg.d_word.convert(wid);
                line += word;
                if(wid == d.dict_set_trg.end_id_word){
                    break;
                }
                line += sep;
            }
            line += delim;
            out_sents << line;
        }
        out_sents.close();
    }

    void print_dists_att(const std::string file_path, const unsigned int model_id, std::vector<std::vector<std::vector<std::vector<float> > > >& str_dists_att) {
        std::string delim = "\n";
        std::string sep1 = ":";
        std::string sep2 = " ";
        std::ofstream out_dists_att(file_path);
        for(auto sent : str_dists_att[model_id]){
            std::string line_sent = "";
            for(auto trg : sent){
                std::string line_trg = "";
                for(auto prob : trg){
                    if(line_trg != ""){
                        line_trg += sep1;
                    }
                    line_trg += std::to_string(prob);
                }
                if(line_sent != ""){
                    line_sent += sep2;
                }
                line_sent += line_trg;
            }
            line_sent += delim;
            out_dists_att << line_sent;
        }
        out_dists_att.close();
    }

    void print_dists_out(const std::string file_path, const unsigned int model_id, std::vector<std::vector<std::vector<std::vector<float> > > >& str_dists_out) {
        std::string delim = "\n";
        std::string sep1 = ":";
        std::string sep2 = " ";
        std::ofstream out_dists_out(file_path);
        for(auto sent : str_dists_out[model_id]){
            std::string line_sent = "";
            for(auto trg : sent){
                std::string line_trg = "";
                for(auto prob : trg){
                    if(line_trg != ""){
                        line_trg += sep1;
                    }
                    line_trg += std::to_string(prob);
                }
                if(line_sent != ""){
                    line_sent += sep2;
                }
                line_sent += line_trg;
            }
            line_sent += delim;
            out_dists_out << line_sent;
        }
        out_dists_out.close();
    }

};

#endif // INCLUDE_GUARD_PRINT_HPP
