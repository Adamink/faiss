import math

cpu_ratio = 72
gpu_bandwidth_ratio = 4000 / 313
gpu_core_ratio = 18432 / 1920

class Parameter:
    def __init__(self): # set default values
        self.dim = 1024
        self.m = 8
        self.k = 100
        self.nb = 1000000
        self.nq = 1000000
        self.nlist = 1
        self.nprobe = 1
    def __str__(self):
        s = "d{}_nq{}_nb{}_nlist{}_nprobe{}_k{}_m{}".format(
            self.dim, self.nq, self.nb, self.nlist, self.nprobe, self.k, self.m)
        return s
    def important_params(self):
        s = "nb{}_nlist{}_nprobe{}_k{}_m{}".format(
            self.nb, self.nlist, self.nprobe, self.k, self.m)
        return s
    def parse(self, s):
        self.dim = int(s.split('_')[0][1:])
        self.nq = int(s.split('_')[1][2:])
        self.nb = int(s.split('_')[2][2:])
        self.nlist = int(s.split('_')[3][5:])
        self.nprobe = int(s.split('_')[4][6:])
        self.k = int(s.split('_')[5][1:])
        self.m = int(s.split('_')[6][1:])
        # self.dim, self.nq, self.nb, self.nlist, self.nprobe, self.k, self.m = [int(x) for x in s.split('_')]
    
def ivf_time(nlist = 128, dim = 1024, nq = 10000):
    # nlist * dim * nq
    # nq * nlist * log(nprobe)
    sample_value = 100 * 1024 * 10000
    sample_time = 12 # ms
    formula = nlist * dim * nq
    return sample_time / sample_value * formula / gpu_core_ratio

def pq_time_cpu(nq = 10000, nprobe = 1, nlist = 1, nb = 200000, m = 8):
    sample_time = 68.97
    sample_value = 10000 * 200000 * 8
    formula = nq * nprobe / nlist * nb * m
    return sample_time / sample_value * formula / cpu_ratio

def pq_time(nq = 10000, nprobe = 1, nlist = 1, nb = 200000, m = 8):
    # time for pq, limited by memory bandwidth
    # nq * (nprobe / nlist * nb) * m 
    sample_time = 0.0756
    sample_value = 10000 * 200000 * 8
    formula = nq * nprobe / nlist * nb * m
    return sample_time / sample_value * formula / gpu_bandwidth_ratio

def k_select_time_cpu(nq = 10000, nprobe = 1, nlist = 1, nb = 100000, k = 1000):
    # time for kselect, limited by CPU frequency
    sample_time = 2.885
    sample_value = 10000 * 100000 * math.log2(1000)
    formula = nq * nprobe / nlist * nb * math.log2(k)
    return sample_time / sample_value * formula / cpu_ratio

def k_select_time_gpu(nq = 10000, nprobe = 1, nlist = 1, nb = 100000, k = 1000):
    sample_time = 80.879
    sample_value = 10000 * 1000 * 1000 * math.log2(1000) * math.log2(1000)
    formula = nq * k * k * math.log2(k + 1) * math.log2(nprobe / nlist * nb)
    return sample_time / sample_value * formula / gpu_core_ratio

def estimate(p):
    ivf = ivf_time(p.nlist, p.dim, p.nq)
    pq = pq_time(p.nq, p.nprobe, p.nlist, p.nb, p.m)
    pq_cpu = pq_time_cpu(p.nq, p.nprobe, p.nlist, p.nb, p.m)

    k_select = k_select_time_gpu(p.nq, p.nprobe, p.nlist, p.nb, p.k)
    k_select_cpu = k_select_time_cpu(p.nq, p.nprobe, p.nlist, p.nb, p.k)

    pure_cpu = ivf + pq_cpu + k_select_cpu
    pure_gpu = ivf + pq + k_select
    combined = ivf + pq + k_select_cpu

    speedup_gpu = max(1.0, pure_gpu / combined)
    speedup_cpu = max(1.0, pure_cpu / combined)

    s1 = ivf / pure_gpu
    s2 = pq / pure_gpu
    s3 = k_select / pure_gpu
    if max(s1, s2, s3) == s1:
        dom = "IVFDist"
    elif max(s1, s2, s3) == s2:
        dom = "PQDist"
    else:
        dom = "SelK"
    # print("$nlist={},nprobe={},k={},m={}$ & {:.1f} & {:.1f} & {}({:.1f}\\%) & {:.2f} & {:.2f}\\\\".format(
    #     p.nlist, p.nprobe, p.k, p.m,
    #     pure_cpu, pure_gpu, dom, 100 * max(s1, s2, s3), speedup_cpu, speedup_gpu))
    print("{}-{}-{}-{} & {:.1f} & {:.1f} & {}({:.1f}\\%) & {:.1f} & {:.1f}x & {:.1f}x\\\\".format(
        p.nlist, p.nprobe, p.k, p.m,
        pure_cpu, pure_gpu, dom, 100 * max(s1, s2, s3), combined, speedup_cpu, speedup_gpu))
    
    # print("{:.3f} {:.3f} {:.3f}".format(ivf, pq, k_select))
    # print("{:.3f} {:.3f} {:.3f}".format(ivf / pure_gpu, pq / pure_gpu, k_select / pure_gpu))
    # print("{:.3f} {:.3f} {:.3f}".format(ivf / pure_cpu, pq_cpu / pure_cpu, k_select_cpu / pure_cpu))
    # print("{:.3f} {:.3f}".format(speedup_gpu, speedup_cpu))
    

# p = Parameter()
# p.k = 1000
# estimate(p)
# print(p)

exps = ["d1024_nq1000000_nb1000000_nlist1_nprobe1_k500_m16", 
        "d1024_nq1000000_nb1000000_nlist1_nprobe1_k1000_m16", 
        "d1024_nq1000000_nb1000000_nlist1000_nprobe100_k1_m16",
        "d1024_nq1000000_nb1000000_nlist1000_nprobe100_k500_m16", 
        "d1024_nq1000000_nb1000000_nlist1000_nprobe100_k1000_m16",
        "d1024_nq1000000_nb1000000_nlist1_nprobe1_k1_m64",
        "d1024_nq1000000_nb1000000_nlist1_nprobe1_k500_m64",
        "d1024_nq1000000_nb1000000_nlist1_nprobe1_k1000_m64",
        "d1024_nq1000000_nb1000000_nlist100_nprobe1_k100_m16",
        "d1024_nq1000000_nb1000000_nlist100_nprobe1_k100_m16",
        "d1024_nq1000000_nb1000000_nlist100_nprobe50_k100_m16",
        "d1024_nq1000000_nb1000000_nlist100_nprobe100_k100_m16"]

for e in exps:
    p = Parameter()
    p.parse(e)
    estimate(p)

# H100 features 80 Streaming Multiprocessors (SMs) and 18,432 CUDA cores
# Grace CPU cores (number) Up to 72 cores 
# CPU 2.3Ghz