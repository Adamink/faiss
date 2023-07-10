import os
import subprocess

home_fd = "../"
program_fd = os.path.join(home_fd, "build/tutorial/cpp/")
nsys_fd = os.path.join(home_fd, "results/nsys/")
log_fd = os.path.join(home_fd, "results/logs/")

dim = 1024
m = 8
k = 1
nb = 1000000
nq = 1000000
nlist = 1
nprobe = 1
bits = 8

program_cmd = os.path.join(program_fd, 
    "IVFPQ-GPU --dim {} --nq {} --nb {} --nlist {} --nprobe {} -k {} --m {} --bits {} --u true".format(
    dim, nq, nb, nlist, nprobe, k, m, bits))

output_prefix = "d{}_nq{}_nb{}_nlist{}_nprobe{}_k{}_m{}".format(dim, nq, nb, nlist, nprobe, k, m)
nsys_file_pth = os.path.join(nsys_fd, output_prefix + ".nsys-rep")
nsys_cmd = "sudo nsys profile --stats=true --trace=cuda --gpu-metrics-device=0 -o {}".format(nsys_file_pth)

log_file_pth = os.path.join(log_fd, output_prefix + ".log")
log_cmd = "> {}".format(log_file_pth)

cmd = " ".join(nsys_cmd, program_cmd, log_cmd)

# os.chdir(program_fd)
os.system(cmd)

def parse_log(log_file_pth):
    with open(log_file_pth, "r") as f:
        lines = f.readlines()

        searchImpl = 0.
        runpq = 0.
        table = 0.
        calcoffsets = 0.
        pqscan = 0.
        pass1 = 0.
        pass2 = 0.
        for line in lines:
            if "calcoffsets" in line:
                calcoffsets += float(line.split(":")[-1])
            elif "pqscan" in line:
                pqscan += float(line.split(":")[-1])
            elif "pass1" in line:
                pass1 += float(line.split(":")[-1])
            elif "pass2" in line:
                pass2 += float(line.split(":")[-1])
            elif "PreComputeTable" in line:
                table += float(line.split(":")[-1])
            elif "runPQScanMultiPassPrecomputed"in line:
                runpq += float(line.split(":")[-1])
            elif "searchImpl_" in line:
                searchImpl += float(line.split(":")[-1])
        all = calcoffsets + pqscan + pass1 + pass2
        print("calcoffsets:{:.3f}, pqscan:{:.3f}, pass1:{:.3f}, pass2:{:.3f}".format(calcoffsets, pqscan, pass1, pass2))
        print("calcoffsets:{:.1f}%, pqscan:{:.1f}%, pass1:{:.1f}%, pass2:{:.1f}%".format( 
            100 * calcoffsets / all, 100 * pqscan / all, 100 * pass1 / all, 100 * pass2 / all))
        print("precomputetable:{:.3f}, runpq:{:.3f}, searchImpl:{:.3f}".format(table, runpq, all, searchImpl))


    