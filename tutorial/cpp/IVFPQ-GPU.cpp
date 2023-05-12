/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <cstdlib>
#include <random>

// #include <faiss/IndexFlat.h>
// #include <faiss/IndexIVFPQ.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/StandardGpuResources.h>

using idx_t = faiss::idx_t;

int main() {
    int d = 1024;      // dimension
    int nb = 100000; // database size
    int nq = 10000;  // number of queries
    int nlist = 100; // number of lists for IVF indexing
    int k = 100; // k nearest neighbors to be found
    int m = 8; // number of subquantizers

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

    GpuIndexIVFPQConfig config;
    faiss::gpu::config.usePrecomputedTables = true;
    faiss::gpu::GpuIndexFlatL2 quantizer(&res, d); // the other index
    faiss::gpu::GpuIndexIVFPQ index(&res, &quantizer, d, nlist, m, 8);

    index.train(nb, xb);
    index.add(nb, xb);

    { // search xq
        idx_t* I = new idx_t[k * nq];
        float* D = new float[k * nq];

        index.nprobe = 10;
        index.search(nq, xq, k, D, I);

        delete[] I;
        delete[] D;
    }

    delete[] xb;
    delete[] xq;

    return 0;
}
