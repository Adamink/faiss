import os
import sys
import csv
import matplotlib.pyplot as plt

def log(s, log_file_pth):
    print(s)
    if log_file_pth is not None:
        with open(log_file_pth, "a+") as f:
            print(s, file = f)

class Parameters:
    def __init__(self): # set default values
        self.dim = 1024
        self.m = 8
        self.k = 1
        self.nb = 1000000
        self.nq = 1000000
        self.nlist = 1
        self.nprobe = 1
        self.bits = 8
        self.u = True

    def __str__(self):
        return "d{}_nq{}_nb{}_nlist{}_nprobe{}_k{}_m{}".format(
            self.dim, self.nq, self.nb, self.nlist, self.nprobe, self.k, self.m)
    
class Experiment:
    def __init__(self):
        self.name = "empty"
        self.kernel_name = "empty"
    def __iter__(self):
        self.index = 0
        return self
    def __next__(self):
        if self.index < len(self.paramList):
            ret = self.paramList[self.index]
            self.index += 1
            return ret
        else:
            raise StopIteration
    def set_param_from_value_list(self, param_name, value_list):
        self.paramList = []
        for value in value_list:
            p = Parameters()
            setattr(p, param_name, value)
            self.paramList.append(p)
    
class K_Experiment(Experiment):
    def __init__(self):
        self.name = "k"
        self.kernel_name = "pass1SelectLists"
        self.set_param_from_value_list("k", [int(2 ** i) for i in range(0, 9)])

class M_Scan_Experiment(Experiment):
    def __init__(self):
        self.name = "m_scan"
        self.kernel_name = "pqScanPrecomputedMultiPass"
        self.set_param_from_value_list("m", [1,2,4,8,16,32])

class Nb_Scan_Experiment(Experiment):
    def __init__(self):
        self.name = "nb_scan"
        self.kernel_name = "pqScanPrecomputedMultiPass"
        self.set_param_from_value_list("nb", [i * 100000 for i in range(1, 11)])

class ExperimentRunner:
    def __init__(self):
        self.home_fd = "../"
        self.results_fd = os.path.join(self.home_fd, "results")
        self.program_fd = os.path.join(self.home_fd, "build/tutorial/cpp/")
        self.nsys_fd = os.path.join(self.results_fd, "nsys")
        self.log_fd = os.path.join(self.results_fd, "logs")

    def run_single_profiling(self, param):
        program_cmd = os.path.join(self.program_fd, 
            "IVFPQ-GPU --dim {} --nq {} --nb {} --nlist {} --nprobe {} --k {} --m {} --bits {} --u {}".format(
            param.dim, param.nq, param.nb, param.nlist, param.nprobe, param.k, param.m, param.bits, param.u))

        nsys_file_pth = os.path.join(self.nsys_fd, str(param) + ".nsys-rep")
        nsys_cmd = "sudo nsys profile --trace=cuda --gpu-metrics-device=0 -o {}".format(nsys_file_pth)

        log_file_pth = os.path.join(self.log_fd, str(param) + ".log")
        log_cmd = "> {}".format(log_file_pth)

        cmd = " ".join([nsys_cmd, program_cmd, log_cmd])
        if not os.path.exists(nsys_file_pth):
            os.system(cmd)
        return nsys_file_pth, log_file_pth
    
    def parse_kernel_info_from_nsys(self, kernel_name, nsys_file_pth, output_file_pth = None):
        csv_file_pth = nsys_file_pth.replace(".nsys-rep", "_gpukernsum.csv")
        parse_cmd = "nsys stats --report gpukernsum --format csv {} --output .".format(
            nsys_file_pth)
        if not os.path.exists:
            os.system(parse_cmd)
        with open(csv_file_pth) as f:
            reader = csv.reader(f)
            for line in reader:
                if kernel_name in line[-1]:
                    time_percent = float(line[0])
                    total_time = float(line[1])
                    instances = int(line[2])
                    avg_time = float(line[3])
                    grid = ",".join(line[8].split())
                    block = ",".join(line[9].split())
                    # return grid, block, time_percent, total_time, instances, avg_time #ns
                    output_s  = "({}) ({}) {:.1f}% {:.0f} {:d} {:.2f}".format(
                        grid, block, time_percent, total_time / 1e6, instances, avg_time / 1e6)
                    log(output_s, output_file_pth)
    
    def parse_log_info(self, log_file_pth, output_file_pth = None):
        with open(log_file_pth, "r") as f:
            lines = f.readlines()
            searchCoarse = 0.
            searchImpl = 0.
            table = 0.
            runpq = 0.

            for line in lines:
                if "PreComputeTable" in line:
                    table += float(line.split(":")[-1])
                elif "runPQScanMultiPassPrecomputed"in line:
                    runpq += float(line.split(":")[-1])
                elif "searchImpl_" in line:
                    searchImpl += float(line.split(":")[-1])
                elif "searchCoarseQuantizer:0.354" in line:
                    searchCoarse += float(line.split(":")[-1])
            
            all = searchCoarse + searchImpl
            log("precomputetable:{:.3f}, runpq:{:.3f}, searchImpl:{:.3f}".format(table, runpq, all, searchImpl), output_file_pth)

    def run_experiments(self, experiment):
        results_collection_file_pth = os.path.join(self.results_fd, experiment.name)
        if os.path.exists(results_collection_file_pth):
            os.remove(results_collection_file_pth)

        for param in experiment:
            log("**********", results_collection_file_pth)
            log(param, results_collection_file_pth)
            nsys_file_pth, log_file_pth = self.run_single_profiling(param)
            self.parse_kernel_info_from_nsys(experiment.kernel_name, nsys_file_pth, results_collection_file_pth)
            self.parse_log_info(log_file_pth)


class Analyzer:
    def __init__(self, results_fd):
        self.results_fd = results_fd
    
    def parse(self, experiment):
        log_file_pth = os.path.join(self.results_fd, experiment.name)
        save_fig_pth = os.path.join(self.results_fd, "{}.png".format(experiment.name))
        if experiment.name == "nb_scan":
            func = self.parse_pq_nb
        elif experiment.name == "m_scan":
            func = self.parse_pq_m
        elif experiment.name == "k":
            func = self.parse_k
    
        func(log_file_pth, save_fig_pth)

    def parse_pq_m(self, log_file_pth, save_fig_pth):
        with open(log_file_pth, "r") as f:
            lines = f.readlines()
            m_list = []
            bandwidth_list = []
            for i, line in enumerate(lines):
                if "*" in line:
                    config_line = lines[i + 1]
                    result_line = lines[i + 2]
                    m = int(config_line.split("_")[-1][1:])
                    m_list.append(m)
                    totaltime = float(result_line.split(" ")[-3]) / 1000.
                    totalmem = float(1000000. * m * 1000000)
                    bandwidth = totalmem / totaltime / 1e9
                    bandwidth_list.append(bandwidth)
        plt.figure()
        plt.plot(m_list, bandwidth_list)
        plt.axhline(y = 313, color = 'r')
        plt.xlabel("subQuantizers")
        plt.ylabel("memory bandwidth(GB/s)")
        plt.title("m_scan")
        plt.savefig(save_fig_pth)

    def parse_pq_nb(self, log_file_pth, save_fig_pth):
        with open(log_file_pth, "r") as f:
            lines = f.readlines()
            nb_list = []
            bandwidth_list = []
            for i, line in enumerate(lines):
                if "*" in line:
                    config_line = lines[i + 1]
                    result_line = lines[i + 2]
                    nb = int(config_line.split("_")[2][2:])
                    nb_list.append(nb)
                    totaltime = float(result_line.split(" ")[-3]) / 1000. # seconds
                    totalmem = float(nb * 8 * 1000000)
                    bandwidth = totalmem / totaltime / 1e9
                    bandwidth_list.append(bandwidth)

        plt.figure()
        plt.plot(nb_list, bandwidth_list)
        plt.axhline(y = 313, color = 'r')
        plt.xlabel("database_size")
        plt.ylabel("memory bandwidth(GB/s)")
        plt.title("nb_scan")
        plt.savefig(save_fig_pth)  

    def parse_k(self, log_file_pth, save_fig_pth):
        with open(log_file_pth, "r") as f:
            lines = f.readlines()
            k_list = []
            time_list = []
            for i, line in enumerate(lines):
                if "*" in line:
                    config_line = lines[i + 1]
                    result_line = lines[i + 2]
                    k = int(config_line.split("_")[5][1:])
                    k_list.append(k)
                    totaltime = float(result_line.split(" ")[-3]) / 1000. # seconds
                    time_list.append(totaltime)
        print(k_list)
        print(time_list)
        plt.figure()
        plt.plot(k_list, time_list)
        plt.xlabel("k")
        plt.ylabel("time")
        plt.title("k")
        plt.savefig(save_fig_pth)

if __name__ == '__main__':
    runner = ExperimentRunner()
    analyzer = Analyzer(runner.results_fd)

    exps = [Nb_Scan_Experiment(), M_Scan_Experiment(), K_Experiment()]
    for exp in exps:
        runner.run_experiments(exp)
        analyzer.parse(exp)

    