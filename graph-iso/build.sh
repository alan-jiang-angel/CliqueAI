#!/bin/bash

docker build -f Dockerfile.build -t bliss_builder .

docker run --rm \
  -v $(pwd):/build \
  bliss_builder \
  g++ indexer.cpp -o indexer \
    -O3 \
    -static \
    -lbliss \
    -lgmp \
    -lhiredis \
    -lssl \
    -lcrypto

echo "Build complete: ./indexer"

