#include <string>

#ifndef INCLUDE_GUARD_COMP_HPP
#define INCLUDE_GUARD_COMP_HPP

namespace s2s {

    // Sort in descending order of length
    struct CompareLength {
        bool operator()(const std::pair<unsigned int, std::pair<unsigned int, unsigned int> >& first, const std::pair<unsigned int, std::pair<unsigned int, unsigned int> >& second) {
            if(
                (first.second.first < second.second.first) ||
                (first.second.first == second.second.first && first.second.second < second.second.second)
            ){
                return true;
            }
            return false;
        }
        bool operator()(const std::pair<unsigned int, unsigned int>& first, const std::pair<unsigned int, unsigned int>& second) {
            if(first.second < second.second){
                return true;
            }
            return false;
        }
    };

    struct CompareString {
        bool operator()(const std::pair<std::vector<unsigned int>, std::vector<unsigned int>>& first, const std::pair<std::vector<unsigned int>, std::vector<unsigned int>>& second) {
            if(
                (first.first.size() > second.first.size()) ||
                (first.first.size() == second.first.size() && first.second.size() > second.second.size())
            ){
                return true;
            }
            return false;
        }
        bool operator()(const std::vector<unsigned int>& first, const std::vector<unsigned int>& second) {
            if(first.size() > second.size()){
                return true;
            }
            return false;
        }
        bool operator()(const std::pair<std::string, unsigned int>& first, const std::pair<std::string, unsigned int>& second) {
            if(first.second > second.second){
                return true;
            }
            return false;
        }
    };

};

#endif // INCLUDE_GUARD_COMP_HPP
