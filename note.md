```cmake
if(FAISS_ENABLE_GPU)
  include_directories("/usr/local/cuda/include")
endif()
```
```sh
cmake -B build . -DCMAKE_CUDA_ARCHITECTURES="75" -DCUDAToolkit_ROOT="/usr/local/cuda" # used when openblaslib is installed into /usr/lib/
cmake -B build . -DCMAKE_CUDA_ARCHITECTURES="75" -DCUDAToolkit_ROOT="/usr/local/cuda" -DBLAS_LIBRARIES="/mnt/scratch/xiaowu/OpenBLASLib/lib" -DLAPACK_LIBRARIES="/home/xiaowu/.local/liblapack.a/"
sudo apt-get install -y libboost-program-options-dev
```
In c_cpp_properties.json
```json
  ...
  "compilerPath": "/usr/local/cuda/bin/nvcc",
  ...
```

```
sudo nsys profile --stats=true --trace=cuda --gpu-metrics-device=0 	--cuda-memory-usage=true --cuda-graph-trace=graph ./IVFPQ-GPU
```

```txt
nb, n: database size
nq: number of queries
d: dimensionality of the input vectors
k: k nearest neighbors
M, m, subQuantizers = 8 : number of subquantizers [1,2,3,4,8,12,16,20,24,28,32,40,48,56,64,96] (float16 for >=56)
nbits: number of bits per subvector index (per quantization index)
dsub = d / M: dimensionality of each subvector
ksub = 1 << nbits: number of centroids for each subquantizer
nprobe: number of probes at query time (IVF), must <= 2048
nlist = 100 : number of inverted lists, defined in IndexIVF.h
bitsPerCode = 8

interleavedLayout: false this is a feature under development, do not use!

Train_hot_start,     ///< the centroids are already initialized
Train_shared,        ///< share dictionary accross PQ segments
Train_hypercube,     ///< initialize centroids with nbits-D hypercube
Train_hypercube_pca, ///< initialize centroids with nbits-D hypercube

centroids: (M, ksub, dsub)

IndexPQ: 继承IndexFlatCodes，包含ProductQuantizer
ProductQuantizer::train
    xlice: 根据m，取x的某几层作为subvector
    clus.train(n, xslice, assign_index ? *assign_index : index);
        Clustering::train_encoded(n, xslice, nullptr, index, weights)
            subsample_training_set
```
In main func:
```cpp
    faiss::gpu::GpuIndexFlatL2 quantizer(&res, d); // the other index
    faiss::gpu::GpuIndexIVFPQ index(&res, &quantizer, d, nlist, m, 8);

    index.train(nb, xb);
    index.add(nb, xb);

    ...
    idx_t* I = new idx_t[k * 5];
    float* D = new float[k * 5];

    index.search(5, xb, k, D, I);
```

```cpp
class GpuIndexIVFPQ:
  // Trains the coarse and product quantizer based on the given vector data
  void train(idx_t n, const float* x) override;

  // = GpuIndex::add
  // Add n vectors of dimension d to the index
  // Vectors are implicitly assigned labels ntotal .. ntotal + n - 1
  void add(idx_t, const float* x) override;
    baseIndex_->addVectors

  // = GpuIndex::search
  void search(idx_t n, const float* x, idx_t k, float* distances, idx_t* labels, const SearchParameters* params) const

  // Called from GpuIndex for search
  void searchImpl(idx_t n,const float* x,int k,float* distances,idx_t* labels,const SearchParameters* params) const override;
```

```cpp
GpuIndexFlat::searchImpl_ // calls data_->query
  FlatIndex::query
    bfKnnOnDevice
      runL2Distance
        runDistance
          runL2Norm
          runMatrixMult
          runL2SelectMin
            l2selectMin1(kernel)

GpuIndexIVFPQ::search = GpuIndex::search // make sure searchImpl_ called with device-resident pointers
  // outDistances, outLabels = toDeviceTemporary...
  // Currently, we don't handle the case where the output data won't fit on the GPU
  GpuIndex::searchFromCpuPaged_ or searchNonPaged_ // batch-processing queries by using pinned memories
  // batchSize = nextHighestPowerOf2((kNonPinnedPageSize / (sizeof(float) * this->d)))
  // only if dataSize >= minPagedSize_(268435456)
    GpuIndexIVF::searchImpl_ // set nprobe, calls baseIndex_::search(inited in GpuIndexIVFPQ)
      IVFPQ::search
        IVFBase::searchCoarseQuantizer_ // Performs search in a CPU or GPU coarse quantizer for IVF cells, calls coarseQuantizer::search
        // always: residuals, gpu, but not centroids
          GpuIndex::search // ??? might recursive here
            GpuIndexFlat::searchImpl_ // called from GpuIndex::search, calls data_->query
              FlatIndex::query
                bfKnnOnDevice
                  runL2Distance
                    runDistance
                      runL2Norm // ||c||^2(might be precomputed), ||q||^2
                        l2NormRowMajor(kernel)
                      chooseTileSize // Distance.cu, both number of queries and number of centroids being at least 512
                      //For <= 8 GB GPUs, prefer 768 MB of usage
                      // preferredTileRows = 512(batch_size)
                      // tileCols = std::min(targetUsage / preferredTileRows, numCentroids);
                      runMatrixMult // (query id x dim) x (centroid id, dim)' = (query id, centroid id)
                        rawGemm
                          cublasGemmEx(cublas api)
                            volta_sgemm_128x64_tn(kernel) // 
                            volta_sgemm_128x32_tn(kernel) // whem dim = 100
                            gemmk1_kernel(kernel) // when dim = 1
                      runL2SelectMin
                      // For L2 distance, we use this fused kernel that performs both
                      // adding ||c||^2 to -2qc and k-selection, so we only need two
                      // passes (one write by the gemm, one read here) over the huge
                      // region of output memory
                        l2selectMin1(kernel)
                      runSumAlongRows // outDistance
                        sumAlongRows(kernel)
        IVFPQ::searchImpl_
          IVFPQ::runPQPrecomputedCodes_ // Performs matrix multiplication to calculate term 3: - 2 * (x|y_R) (Construct Table)
            runTransposeAny
              transposeOuter(kernel)
              transposeAny(kernel)
            runBatchMatrixMult
              rawBatchGemm
                cublasGemmStridedBatchedEx(cublas api) // cublas kernel
            IVFPQ::runPQScanMultiPassPrecomputed 
              runMultiPassTile // same as below
                pqScanPrecomputedMultiPass(kernel)
                runPass1SelectLists(kernel)
                runPass2SelectLists(kernel)
          IVFPQ::runPQNoPrecomputedCodes_
            runPQScanMultiPassNoPrecomputed
              runMultiPassTile
                runCalcListOffsets // Calculate offset lengths, so we know where to write out intermediate results
                runPQCodeDistances // Calculate residual code distances, since this is without precomputed codes
                  pqCodeDistances(kernel)
                pqScanNoPrecomputedMultiPass(kernel)
                runPass1SelectLists(kernel) // k-select the output in chunks, to increase parallelism
                runPass2SelectLists(kernel) // select final results

```

sgemm: Single Precision General Matrix Multiplication

grace hopper: CPU-GPU NVLink bandwidth 900GB/s
![](grace_hopper.png)

[Faiss之IVF详解](https://blog.csdn.net/lijinwen920523/article/details/113819843)
[Faiss专栏](https://blog.csdn.net/rangfei/category_10080429.html)


Questions:
What is precomputed codes? What is precomputed table?
  如果设置了usePrecomputedTables_，GpuIndexIVFPQ::setPrecomputedCodes会被调用
  IndexIVFPQ.cpp设置并描述了Precomputed tables, 指的是下面的term2

  ```cpp
  /** Precomputed tables for residuals
   *
   * During IVFPQ search with by_residual, we compute
   *
   *     d = || x - y_C - y_R ||^2
   *
   * where x is the query vector, y_C the coarse centroid, y_R the
   * refined PQ centroid. The expression can be decomposed as:
   *
   *    d = || x - y_C ||^2 + || y_R ||^2 + 2 * (y_C|y_R) - 2 * (x|y_R)
   *        ---------------   ---------------------------       -------
   *             term 1                 term 2                   term 3
   *
   * When using multiprobe, we use the following decomposition:
   * - term 1 is the distance to the coarse centroid, that is computed
   *   during the 1st stage search.
   * - term 2 can be precomputed, as it does not involve x. However,
   *   because of the PQ, it needs nlist * M * ksub storage. This is why
   *   use_precomputed_table is off by default
   * - term 3 is the classical non-residual distance table.
   *
   * Since y_R defined by a product quantizer, it is split across
   * subvectors and stored separately for each subvector. If the coarse
   * quantizer is a MultiIndexQuantizer then the table can be stored
   * more compactly.
   *
   * At search time, the tables for term 2 and term 3 are added up. This
   * is faster when the length of the lists is > ksub * M.
   */
  ```
train和add做了什么? 什么时候precompute的table（不重要）
```cpp
  IVFBase::addVectors // Classify and encode/add vectors to our IVF lists.
  GpuIndexIVFPQ::train // Trains the coarse quantizer based on the given vector data
```

[理解CUDA中的thread,block,grid和warp](https://zhuanlan.zhihu.com/p/123170285)

考证：
索引的训练不是越多越好，在faiss的源代码中已经默认设置了一个quantizer容纳的最多向量是256个，所以训练集最大为nlist *256，大于该值则会从训练集中随机取子集。待考证
IVFBase::maxListLength_? seems not
sgemm是在哪里被调用的
  cublasGemmEx
为什么l2selectmin只计算||c||^2 to -2qc？不计算||q||^2