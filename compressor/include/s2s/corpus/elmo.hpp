#include <iostream>
#include <vector>
#include <cassert>
#include <iomanip>
#include "H5Cpp.h"

using namespace std;
using namespace H5;

#ifndef INCLUDE_GUARD_S2S_ELMO_HPP
#define INCLUDE_GUARD_S2S_ELMO_HPP

class ELMo {

private:

    H5File file;

public:

    explicit ELMo(std::string file_name) : file(file_name, H5F_ACC_RDONLY) {
    }

    ~ELMo (){
        file.close();
    }

    void get(unsigned int sent_id, std::vector<std::vector<std::vector<float> > >& elmo_vector) {
        const std::string get_id = std::to_string(sent_id);
        DataSet dataset = file.openDataSet(get_id);
        DataType datatype = dataset.getDataType();
        DataSpace dataspace = dataset.getSpace();
        int rank = dataspace.getSimpleExtentNdims();
        hsize_t dims[rank];
        dataspace.getSimpleExtentDims(dims);
        FloatType ftype = dataset.getFloatType();
        std::string order_string;
        H5T_order_t order = ftype.getOrder( order_string);
        size_t size = ftype.getSize();
        const size_t layer_size = dims[0];
        const size_t token_size = dims[1];
        const size_t vector_size = dims[2];
        auto data = new float[layer_size*token_size*vector_size];
        if(order==0 && size == 4){
            dataset.read(data, PredType::IEEE_F32LE);
        }else if(order == 0 && size == 8){ 
            dataset.read(data, PredType::IEEE_F64LE);
        }else if(order == 1  && size == 4){
            dataset.read(data, PredType::IEEE_F32BE);
        }else if(order ==1 && size == 8){
            dataset.read(data, PredType::IEEE_F64BE);
        }else{ 
            std::cout << "Unknown data format" << std::endl;
            assert(false);
        }
        elmo_vector.resize(layer_size);
        for(int lid=0; lid < layer_size; lid++){
            elmo_vector[lid].resize(token_size);
            for(int tid=0; tid < token_size; tid++){
                elmo_vector[lid][tid].resize(vector_size);
            }
        }
        // Assign 3D vector
        for ( size_t lid = 0; lid < layer_size; lid++ ){
            for ( size_t tid = 0; tid < token_size; tid++ ){
                for ( size_t vid = 0; vid < vector_size; vid++ ){
                    elmo_vector[lid][tid][vid] = data[token_size*vector_size*lid + vector_size*tid + vid];
                }
            }
        }
        delete data;
        dataspace.close();
        datatype.close();
        dataset.close();
    }
};

#endif
