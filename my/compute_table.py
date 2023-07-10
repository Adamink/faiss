import os
import subprocess

result_fd = "/home/xiao/codes/faiss/results/"
exe_fd = "/home/xiao/codes/faiss/build/tutorial/cpp/"

# bitsPerCode, dim, m 
# nq: 1e6
# db: 1e5

os.chdir(exe_fd)
bitsList = [4, 5, 6, 8]
mList = [1, 2, 4, 8, 16, 32]
dimList = list(range(256, 4097, 256))

for dim in dimList:
    file_name = "dim{}.txt".format(dim)
    file_pth = os.path.join(result_fd, file_name)
    cmd = "./IVFPQ-GPU --nq 1000000 --dim {} -u true > {}".format(dim, file_pth)
    os.system(cmd)
    with open(file_pth, "r") as f:
        lines = f.readlines()
        compute_table_time = 0.
        for line in lines:
            if "Term3Time:" in line:
                term3Time = float(line[len("Term3Time:"):])
                compute_table_time += term3Time
    print("{}:{:.3f}".format(dim, compute_table_time))

# for m in mList:
#     file_name = "m{}.txt".format(m)
#     file_pth = os.path.join(result_fd, file_name)
#     cmd = "./IVFPQ-GPU --nq 1000000 --m {} -u true > {}".format(m, file_pth)
#     os.system(cmd)
#     with open(file_pth, "r") as f:
#         lines = f.readlines()
#         compute_table_time = 0.
#         for line in lines:
#             if "Term3Time:" in line:
#                 term3Time = float(line[len("Term3Time:"):])
#                 compute_table_time += term3Time
#     print("{}:{:.3f}".format(m, compute_table_time))
        
