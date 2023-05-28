analysis summary
```txt
RTX 2060
SM Count 30
L2 Cache Size 3M
Memory Bandwidth 312.97 GiB/s
Memory Size 5.77GB
Core Clock 1.68GHz
```

batch_size
IVF select list: nlist, nprobe, d, IVF-Flat (nb = nlist)
construct LUT isolate: k=1, nb = 1) 
pq-scan 
k-select

sudo nsys profile --stats=true --trace=cuda ./IVFPQ-GPU --nq 1000000 --nlist 100000 --nprobe 100 --k 100
8-d1024_nb100000_nq1000000_nlist100_nprobe10_k100_m8_bits8_u0
=> l2SelectMinK: 1953 instances, average 200000ns
=> tilesize:（512, 100）
=> instances: 1000000 / 512 = 1953
=> 1 pass IO over a submatrix of (query_id, centroid_id), 512 * 100 * 4Byte = 200KB, 
=> bandwidth = 200K / 2e5ns = 1GB/s 

Questions:
    kernel之间的overlap? 
    用了多少SM?
    grid_size 512x8

    dim = 1, k = 100
    separate scan(multiplication) / l2select (same k)
    t2 - t1 ~= scan

end-to-end CPU timer 

sudo nsys profile --stats=true --trace=cuda ./IVFPQ-GPU --nq 100000 --nlist 100000 --nprobe 100 --k 100 --d 256

sudo nsys profile --stats=true --trace=cuda ./IVFPQ-GPU --nq 1000000 --nlist 100000 --nprobe 100 --k 100
9-PQ-d1024_nb100000_nq1000000_nlist100000_nprobe100_k100_m8_bits8_u0

sudo nsys profile --stats=true --trace=cuda ./IVFFlat-GPU --nb 10000 --nq 100000 --nlist 10000 --nprobe 100 --k 100
terminate called after throwing an instance of 'faiss::FaissException' // ?

sudo nsys profile --stats=true --trace=cuda --gpu-metrics-device=0 	--cuda-memory-usage=true --cuda-graph-trace=graph ./IVFPQ-GPU ./IVFPQ-GPU --nq 1000000 --nlist 100000 --nprobe 100 --k 100
10-d1024_nb100000_nq1000000_nlist100000_nprobe100_k100_m8_bits8_u0