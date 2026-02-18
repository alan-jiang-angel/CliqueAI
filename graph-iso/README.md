Dockerfile.build 
    - install g++ compiler

build.sh
    - build

indexer.cpp
    - input 0 index based adjacency list (array of array)
    - output SHA256 hash for fingerprint
    - output canonical permutation for index mapping

ingest.py
    - loop through all input json files,
    - calculate hash and store it on redis with Canonical Permutation and FileName (Graph)
    - for isomorphically identical graph, store file names

Analyzer
    - Check redis to see how much graphs are re used



OVERALL: Not that helpful