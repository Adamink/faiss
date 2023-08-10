/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <ctime>
#include <iostream>

#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <boost/program_options.hpp>

int main(int argc, char** argv) {
    int d = 1024;    // dimension
    int nb = 100000; // database size
    int nq = 10000;  // number of queries
    int nlist = 100; // number of lists for IVF indexing
    int nprobe = 10; // number of lists to search for IVF indexing
    int k = 100;     // k nearest neighbors to be found
    int m = 8;       // number of subquantizers
    int bitsPerCode = 8; // ksub = 2^bitsPerCode, determines number of centroids for each subquantizer
    bool usePrecomputedTables = false;
    bool sift = false;
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
    ("usePrecomputedTables,u", po::value<bool>(&usePrecomputedTables)->default_value(false), "")
    ("sift", po::value<bool>(&sift)->default_value(false),"")
  ;

    // Parse command line arguments
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
    po::notify(vm);

    // std::cout << "Flat-GPU" << std::endl;
    // std::cout << "d" << d << "_nb" << nb << "_nq" << nq << "_nlist" << nlist << "_nprobe" << nprobe << "_k" << k << "_m" << 
    //  m << "_bits" << bitsPerCode << "_u" << usePrecomputedTables << std::endl;

    nb = nlist;
    k = nprobe;

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

    // Using a flat index

    faiss::gpu::GpuIndexFlatL2 index_flat(&res, d);
    
    index_flat.add(nb, xb); // add vectors to the index

    { // search xq
        long* I = new long[k * nq];
        float* D = new float[k * nq];
        auto startTime = clock();
        index_flat.search(nq, xq, k, D, I);
        std::cout << "TotalTime:" << double(clock() - startTime) / CLOCKS_PER_SEC * 1000. << std::endl;

        delete[] I;
        delete[] D;
    }


/*
    // Using an IVF index

    int nlist = 100;
    faiss::gpu::GpuIndexIVFFlat index_ivf(&res, d, nlist, faiss::METRIC_L2);

    assert(!index_ivf.is_trained);
    index_ivf.train(nb, xb);
    assert(index_ivf.is_trained);
    index_ivf.add(nb, xb); // add vectors to the index

    printf("is_trained = %s\n", index_ivf.is_trained ? "true" : "false");
    printf("ntotal = %ld\n", index_ivf.ntotal);

    { // search xq
        long* I = new long[k * nq];
        float* D = new float[k * nq];

        index_ivf.search(nq, xq, k, D, I);

        // print results
        printf("I (5 first results)=\n");
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        printf("I (5 last results)=\n");
        for (int i = nq - 5; i < nq; i++) {
            for (int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        delete[] I;
        delete[] D;
    }

    delete[] xb;
    delete[] xq;
*/
    return 0;
}
