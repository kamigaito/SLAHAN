#!/bin/bash -eu

ROOT_DIR=${1}
IN_DIR="${ROOT_DIR}/dataset2"

if [ ! -e ${IN_DIR} ]
then
    mkdir -p ${IN_DIR}
fi

# Download preprocessed ziff davis and broadcast news comment dataset
cd ${IN_DIR}
wget --no-check-certificate https://www.jamesclarke.net/media/data/broadcastnews-compressions.tar.gz
tar xvzf broadcastnews-compressions.tar.gz

cd ${ROOT_DIR}
