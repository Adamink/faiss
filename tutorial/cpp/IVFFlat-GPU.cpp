/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <cstdlib>
#include <random>
#include <iostream>
#include <boost/program_options.hpp>
// #include <faiss/IndexFlat.h>
// #include <faiss/IndexIVFPQ.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>

using idx_t = faiss::idx_t;

int main(int argc, char** argv) {
    int d = 1024;    // dimension
    int nb = 100000; // database size
    int nq = 10000;  // number of queries
    int nlist = 100; // number of lists for IVF indexing
    int nprobe = 10; // number of lists to search for IVF indexing
    int k = 100;     // k nearest neighbors to be found
    int m = 8;       // number of subquantizers
    int bitsPerCode = 8; 
    bool usePrecomputedTables = false;
    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
    ("help", "produce help message")
    ("dimension,d", po::value<int>(&d)->default_value(1024),  "dimension")
    ("nb", po::value<int>(&nb)->default_value(100000),  "dimension")
    ("nq", po::value<int>(&nq)->default_value(10000), "number of queries")
    ("nlist", po::value<int>(&nlist)->default_value(100), "number of lists for IVF indexing")
    ("nprobe", po::value<int>(&nprobe)->default_value(10), "number of lists to search for IVF indexing")
    ("k", po::value<int>(&k)->default_value(100),  "k nearest neighbors to be found")
    ("m", po::value<int>(&m)->default_value(8), "number of subquantizers")
    ("bits", po::value<int>(&bitsPerCode)->default_value(8), "")
    ("usePrecomputedTables, u", po::value<bool>(&usePrecomputedTables)->default_value(false), "")
  ;

    // Parse command line arguments
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
    po::notify(vm);
    std::cout << "IVFFlat-GPU" << std::endl;
    std::cout << "d" << d << "_nb" << nb << "_nq" << nq << "_nlist" << nlist << "_nprobe" << nprobe << "_k" << k << "_m" << 
     m << "_bits" << bitsPerCode << "_u" << usePrecomputedTables << std::endl;

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    float* xb = new float[d * nb];
    float* xq = new float[d * nq];

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++)
            xb[d * i + j] = distrib(rng);
        xb[d * i] += i / 1000.;
    }

    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < d; j++)
            xq[d * i + j] = distrib(rng);
        xq[d * i] += i / 1000.;
    }

    faiss::gpu::StandardGpuResources res;

    faiss::gpu::GpuIndexIVFFlatConfig config;

    faiss::gpu::GpuIndexFlatL2 quantizer(&res, d); 
    faiss::gpu::GpuIndexIVFFlat index(&res, &quantizer, d, nlist, faiss::METRIC_L2, config);

    index.train(nb, xb);
    index.add(nb, xb);

    std::cout << "*************************" << std::endl;
    { // search xq
        idx_t* I = new idx_t[k * nq];
        float* D = new float[k * nq];

        index.nprobe = nprobe;
        index.search(nq, xq, k, D, I);

        delete[] I;
        delete[] D;
    }

    delete[] xb;
    delete[] xq;

    return 0;
}