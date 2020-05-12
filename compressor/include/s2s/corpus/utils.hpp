#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options.hpp>
#include <boost/regex.hpp>
#include <boost/algorithm/string/regex.hpp>

#ifndef INCLUDE_GUARD_S2S_UTILS_HPP
#define INCLUDE_GUARD_S2S_UTILS_HPP

namespace s2s {
  /*
    Note that this code can only parse sentences encoded by UTF8.
  */
  std::vector<std::string> split_to_char(std::string& str_word){
    std::vector<std::string> chars;
    unsigned int char_size = 0;
    for(unsigned int char_pos = 0; char_pos < str_word.length(); char_pos += char_size){
      unsigned char head = str_word[char_pos];
      if(head < 0x80){
        char_size = 1;
      }else if(head < 0xE0){
        char_size = 2;
      }else if(head < 0xF0){
        char_size = 3;
      }else if(head < 0xF8){
        char_size = 4;
      }else if(head < 0xFC){
        char_size = 5;
      }else{
        char_size = 6;
      }
      chars.emplace_back(str_word.substr(char_pos, char_size));
    }
    return chars;
  }

};

#endif
