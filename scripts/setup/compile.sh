#!/bin/bash -eu

ROOTDIR=${1}

# Build Eigen

if [ -d ${ROOTDIR}/build_eigen ]; then
    rm -rf ${ROOTDIR}/build_eigen
fi

mkdir ${ROOTDIR}/build_eigen

cd ${ROOTDIR}/build_eigen

cmake ../eigen -DCMAKE_INSTALL_PREFIX=`pwd` -DBOOST_ROOT=${BOOST_ROOT}
make -j4
make install

# Build Dynet

if [ -d ${ROOTDIR}/build_dynet ]; then
    rm -rf ${ROOTDIR}/build_dynet
fi

mkdir ${ROOTDIR}/build_dynet

cd ${ROOTDIR}/build_dynet

cmake ../dynet -DEIGEN3_INCLUDE_DIR=`pwd`/../build_eigen/include/eigen3 -DBOOST_ROOT:PATHNAME=${BOOST_ROOT} -DBoost_LIBRARY_DIRS:FILEPATH=${BOOST_ROOT}/lib -DBACKEND=cuda
make -j4

# Build sentence compressor

if [ -d ${ROOTDIR}/build_compressor ]; then
    rm -rf ${ROOTDIR}/build_compressor
fi

mkdir ${ROOTDIR}/build_compressor

cd ${ROOTDIR}/build_compressor

cmake ../compressor -DEigen3_INCLUDE_DIRS=`pwd`/../build_eigen/include/eigen3 -DDynet_INCLUDE_DIRS=`pwd`/../dynet -DDynet_LIBRARIES=`pwd`/../build_dynet/dynet -DDynet_LINK=dynet -DBoost_INCLUDE_DIRS=${BOOST_ROOT}/include -DBoost_LIBRARIES:FILEPATH=${BOOST_ROOT}/lib 
make -j4
