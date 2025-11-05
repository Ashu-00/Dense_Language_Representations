#!/bin/bash

ENG_URL="https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz"
ENG_OUTPUT_FILE="cc.en.300.vec.gz"

echo "Downloading english embeddings"
curl -o $ENG_OUTPUT_FILE $ENG_URL

HI_URL="https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.hi.300.vec.gz"
HI_OUTPUT_FILE="cc.hi.300.vec.gz"

echo "Downloading hindi embeddings"
curl -o $HI_OUTPUT_FILE $HI_URL

gunzip $HI_OUTPUT_FILE
gunzip $ENG_OUTPUT_FILE