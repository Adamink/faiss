/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <random>
#include <iostream>
#include <boost/program_options.hpp>
// #include <faiss/IndexFlat.h>
// #include <faiss/IndexIVFPQ.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/StandardGpuResources.h>

#include <cassert>
#include <sys/stat.h>

using idx_t = faiss::idx_t;


float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}

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

    std::cout << "IVFPQ-GPU" << std::endl;
    std::cout << "d" << d << "_nb" << nb << "_nq" << nq << "_nlist" << nlist << "_nprobe" << nprobe << "_k" << k << "_m" << 
     m << "_bits" << bitsPerCode << "_u" << usePrecomputedTables << std::endl;



    float* xb;
    float* xq;

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    if(!sift){
        xb = new float[d * nb];
        xq = new float[d * nq];

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
    }
    else{
        size_t d1, d2, nout, nqout;
        xb = fvecs_read("/home/xiao/codes/sift/sift_base.fvecs", &d1, &nout);
        xq = fvecs_read("/home/xiao/codes/sift/sift_query.fvecs", &d2, &nqout);
        // dim = 128, nq = 1000000
        // std::cout << d1 << " " << nout << " " << d2 << " " << nqout << std::endl;
        assert(d1 == d2 || !"query does not have same dimension as train set");
        assert(d == d1);
        assert(nout == nb);
        assert(nqout == nq);
    }

    faiss::gpu::StandardGpuResources res;

    faiss::gpu::GpuIndexIVFPQConfig config;
    config.usePrecomputedTables = usePrecomputedTables; // default = false

    faiss::gpu::GpuIndexFlatL2 quantizer(&res, d); 
    faiss::gpu::GpuIndexIVFPQ index(&res, &quantizer, d, nlist, m, bitsPerCode, faiss::METRIC_L2, config);

    index.train(nb, xb);
    index.add(nb, xb);

    std::cout << "********training and adding finished*******" << std::endl;
    { // search xq
        idx_t* I = new idx_t[k * nq];
        float* D = new float[k * nq];

        index.nprobe = nprobe;
        auto startTime = clock();
        index.search(nq, xq, k, D, I);
        std::cout << "TotalTime:" << double(clock() - startTime) / CLOCKS_PER_SEC * 1000. << std::endl;
        delete[] I;
        delete[] D;
    }

    delete[] xb;
    delete[] xq;

    return 0;
}
