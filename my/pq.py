import os
import sys
import csv
import matplotlib.pyplot as plt
import numpy as np

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
        self.cpu = False
        self.nsys = True

    def __str__(self):
        s = "d{}_nq{}_nb{}_nlist{}_nprobe{}_k{}_m{}".format(
            self.dim, self.nq, self.nb, self.nlist, self.nprobe, self.k, self.m)
        if self.cpu:
            s += "_cpu"
        return s
    
class Experiment:
    def __init__(self):
        self.name = "empty"
        self.kernel_name = "empty"
        self.exe_file = "IVFPQ-GPU"
        self.priority_queue = False

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

class IVF_Nlist_Experiment(Experiment):
    def __init__(self):
        super(IVF_Nlist_Experiment, self).__init__()
        self.name = "ivf_nlist"
        self.kernel_name = "l2select"
        self.exe_file = "4-GPU"
        self.set_param_from_value_list('nlist', [16, 32, 64, 128, 256, 512, 1024, 2048])
        # self.set_param_from_value_list('nlist', [100000])
        for param in self.paramList:
            param.nprobe = 100
            param.nq = 10000 
            param.nsys = False

class IVF_Nprobe_Experiment(Experiment):
    def __init__(self):
        super(IVF_Nprobe_Experiment, self).__init__()
        self.name = "ivf_nprobe"
        self.exe_file = "4-GPU"
        self.set_param_from_value_list("nprobe", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
        for param in self.paramList:
            param.nlist = 100000
            param.nq = 10000
            param.nsys = False
        
class K_Experiment(Experiment):
    def __init__(self):
        super(K_Experiment, self).__init__()
        self.name = "k"
        self.kernel_name = "pass1SelectLists"
        self.set_param_from_value_list("k", [int(2 ** i) for i in range(0, 9)])

class M_Scan_Experiment(Experiment):
    def __init__(self):
        super(M_Scan_Experiment, self).__init__()
        self.name = "m_scan"
        self.exe_file = "IVFPQ-GPU-noscan"
        self.kernel_name = "pqScanPrecomputedMultiPass"
        self.set_param_from_value_list("m", [1,2,4,8,16,32])

class Nb_Scan_Experiment(Experiment):
    def __init__(self):
        super(Nb_Scan_Experiment, self).__init__()
        self.name = "nb_scan"
        self.exe_file = "IVFPQ-GPU-noscan"
        self.kernel_name = "pqScanPrecomputedMultiPass"
        self.set_param_from_value_list("nb", [i * 100000 for i in range(1, 11)])

class Nb_Scan_CPU_Experiment(Experiment):
    def __init__(self):
        super(Nb_Scan_CPU_Experiment, self).__init__()
        self.name = "nb_scan_cpu"
        self.exe_file = "2-IVFFlat"
        self.kernel_name = "pqScanPrecomputedMultiPass"
        self.set_param_from_value_list("nb", [i * 100000 for i in range(1, 11)])
        for param in self.paramList:
            param.cpu = True
            param.nq = 10000
            param.nsys = False

class Nb_Scan_GPU_10000_Experiment(Experiment):
    def __init__(self):
        super(Nb_Scan_GPU_10000_Experiment, self).__init__()
        self.name = "nb_scan"
        self.exe_file = "IVFPQ-GPU-noscan"
        self.kernel_name = "pqScanPrecomputedMultiPass"
        self.set_param_from_value_list("nb", [i * 100000 for i in range(1, 11)])
        for param in self.paramList:
            param.nq = 10000
            param.nsys = False

class K_Select_K_full(Experiment):
    def __init__(self):
        super(K_Select_K_full, self).__init__()
        self.name = "k_select_k_full"
        self.exe_file = "IVFPQ-GPU"
        self.kernel_name = "pqScanPrecomputedMultiPass"
        self.set_param_from_value_list("k", [1, 10, 20, 50, 100, 200, 500, 1000, 2000])
        for param in self.paramList:
            param.nsys = False
            param.nq = 10000
            param.nb = 100000
            param.m = 1 # 1 quantizer only

class K_Select_K_tominus(Experiment):
    def __init__(self):
        super(K_Select_K_tominus, self).__init__()
        self.name = "k_select_k_tominus"
        self.exe_file = "IVFPQ-GPU-noscan"
        self.kernel_name = "pqScanPrecomputedMultiPass"
        self.set_param_from_value_list("k", [1, 10, 20, 50, 100, 200, 500, 1000, 2000])
        for param in self.paramList:
            param.nsys = False
            param.nq = 10000
            param.nb = 100000
            param.m = 1

class K_Select_K_CPU(Experiment):
    def __init__(self):
        super(K_Select_K_CPU, self).__init__()
        self.name = "k_select_K_cpu"
        self.exe_file = "priority_queue"
        self.set_param_from_value_list("k", [1, 10, 20, 50, 100, 200, 500, 1000, 2000])
        self.priority_queue = True
        for param in self.paramList:
            param.nb = 100000

# ######## 
class K_Select_Nb_full(Experiment):
    def __init__(self):
        super(K_Select_Nb_full, self).__init__()
        self.name = "k_select_nb_full"
        self.exe_file = "IVFPQ-GPU"
        self.kernel_name = "pqScanPrecomputedMultiPass"
        self.set_param_from_value_list("nb", [10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000])
        for param in self.paramList:
            param.nsys = False
            param.nq = 10000
            param.k = 100
            param.m = 1 # 1 quantizer only

class K_Select_Nb_tominus(Experiment):
    def __init__(self):
        super(K_Select_Nb_tominus, self).__init__()
        self.name = "k_select_nb_tominus"
        self.exe_file = "IVFPQ-GPU-noscan"
        self.kernel_name = "pqScanPrecomputedMultiPass"
        self.set_param_from_value_list("nb", [10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000])
        for param in self.paramList:
            param.nsys = False
            param.nq = 10000
            param.k = 100
            param.m = 1

class K_Select_Nb_CPU(Experiment):
    def __init__(self):
        super(K_Select_Nb_CPU, self).__init__()
        self.name = "k_select_nb_cpu"
        self.exe_file = "priority_queue"
        self.set_param_from_value_list("nb", [10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000])
        self.priority_queue = True
        for param in self.paramList:
            param.k = 100


class ExperimentRunner:
    def __init__(self):
        self.home_fd = "../"
        self.results_fd = os.path.join(self.home_fd, "results")
        self.program_fd = os.path.join(self.home_fd, "build/tutorial/cpp/")
        self.nsys_fd = os.path.join(self.results_fd, "nsys")
        self.log_fd = os.path.join(self.results_fd, "logs")

    def run_single_profiling(self, param, exe_file = "IVFPQ-GPU"):
        program_cmd = os.path.join(self.program_fd, 
            "{} --dim {} --nq {} --nb {} --nlist {} --nprobe {} --k {} --m {} --bits {} --u {}".format(
            exe_file, param.dim, param.nq, param.nb, param.nlist, param.nprobe, param.k, param.m, param.bits, param.u))

        prefix = "{}_{}".format(exe_file,str(param))
        nsys_file_pth = os.path.join(self.nsys_fd, prefix + ".nsys-rep")
        nsys_cmd = "sudo nsys profile --trace=cuda --gpu-metrics-device=0 -o {}".format(nsys_file_pth)

        log_file_pth = os.path.join(self.log_fd, prefix + ".log")
        log_cmd = "> {}".format(log_file_pth)
        if not param.nsys:
            cmd = " ".join([program_cmd, log_cmd])
            if not os.path.exists(log_file_pth):
                os.system(cmd)
        else:
            cmd = " ".join([nsys_cmd, program_cmd, log_cmd])
            if not os.path.exists(log_file_pth):
                os.system(cmd)
        return nsys_file_pth, log_file_pth
    
    def run_priority_queue(self, param, exe_file = "priority_queue"):
        program_cmd = os.path.join(self.program_fd, "{} {} {}".format(exe_file, param.k, param.nb))
        print(program_cmd)
        prefix = "{}_{}_{}".format(exe_file, param.k, param.nb)
        log_file_pth = os.path.join(self.log_fd, prefix + ".log")
        log_cmd = "> {}".format(log_file_pth)
        cmd = " ".join([program_cmd, log_cmd])
        if not os.path.exists(log_file_pth):
            os.system(cmd)
        return log_file_pth

    def parse_kernel_info_from_nsys(self, kernel_name, nsys_file_pth, output_file_pth = None):
        return 
        csv_file_pth = nsys_file_pth.replace(".nsys-rep", "_gpukernsum.csv")
        # parse_cmd = "nsys stats --report gpukernsum --format csv {} --output .".format(
        #    nsys_file_pth)
        parse_cmd = "nsys stats --report gputrace --format csv {} --output .".format(
           nsys_file_pth)
        if not os.path.exists(csv_file_pth):
            os.system(parse_cmd)
        kernel_list = ["l2select", "pqScanPrecomputedMultiPass", "pass1SelectLists"]
        with open(csv_file_pth) as f:
            reader = csv.reader(f)
            start_range = -1
            end_range = -1
            for line in reader:
                if kernel_name in line[-1]:
                    t = int(line[0])
                    if start_range == -1:
                        start_range = t
                    else:
                        start_range = min(start_range, t)
                    if end_range == -1:
                        end_range = t
                    else:
                        end_range = max(end_range, t)
                    # return grid, block, time_percent, total_time, instances, avg_time #ns
                    # output_s  = "({}) ({}) {:.1f}% {:.0f} {:d} {:.2f}".format(
                    #     grid, block, time_percent, total_time / 1e6, instances, avg_time / 1e6)
                    # log(output_s, output_file_pth)
    

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
                elif "searchCoarseQuantizer" in line:
                    searchCoarse += float(line.split(":")[-1])
            
            all = searchCoarse + searchImpl
            log("searchCoarse:{:.3f}, precomputetable:{:.3f}, runpq:{:.3f}, searchImpl:{:.3f}".format(
                searchCoarse, table, runpq, all, searchImpl), output_file_pth)

    def parse_priority_queue_log_info(self, log_file_pth, output_file_pth = None):
        with open(log_file_pth, "r") as f:
            lines = f.readlines()
            log(lines[0], output_file_pth)

    def parse_nlist_log_info(self, log_file_pth, output_file_pth = None):
        with open(log_file_pth, "r") as f:
            lines = f.readlines()
            log(lines[0], output_file_pth)
    
    def run_experiments(self, experiment):
        results_collection_file_pth = os.path.join(self.results_fd, experiment.name)
        if os.path.exists(results_collection_file_pth):
            os.remove(results_collection_file_pth)
        if experiment.exe_file == "4-GPU": # flat 
            for param in experiment:
                log("**********", results_collection_file_pth)
                log(param, results_collection_file_pth)
                nsys_file_pth, log_file_pth = self.run_single_profiling(param, experiment.exe_file)
                self.parse_nlist_log_info(log_file_pth, results_collection_file_pth)
        elif experiment.priority_queue:
            for param in experiment:
                log("**********", results_collection_file_pth)
                log("k{}_nb{}".format(param.k, param.nb), results_collection_file_pth)
                log_file_pth = self.run_priority_queue(param)
                self.parse_priority_queue_log_info(log_file_pth, results_collection_file_pth)
        else:
            for param in experiment:
                log("**********", results_collection_file_pth)
                log(param, results_collection_file_pth)
                nsys_file_pth, log_file_pth = self.run_single_profiling(param, experiment.exe_file)
                if param.nsys:
                    self.parse_kernel_info_from_nsys(experiment.kernel_name, nsys_file_pth, results_collection_file_pth)
                self.parse_log_info(log_file_pth, results_collection_file_pth)


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
        elif experiment.name =="nb_scan_cpu":
            func = self.parse_pq_nb_cpu
        elif experiment.name == "ivf_nlist":
            func = self.parse_ivf_nlist
        elif experiment.name == "ivf_nprobe":
            func = self.parse_ivf_nprobe
        func(log_file_pth, save_fig_pth)

    def parse_together(self, exp1, exp2):
        log_file_pth = os.path.join(self.results_fd, exp1.name)
        save_fig_pth = os.path.join(self.results_fd, "{}.png".format(exp1.name))
        extract_x_func = lambda config_line: config_line.split("_")[2][2:] # nb
        calc_total_mem_func = lambda x: float(x * 8 * 1000000)
        nb_list_1, exp1_time, actual_bandwidth_list = \
            self.parse_pq(log_file_pth, extract_x_func, calc_total_mem_func)

        log_file_pth = os.path.join(self.results_fd, exp2.name)
        nb_list_2, exp2_time, actual_bandwidth_list = \
            self.parse_pq(log_file_pth, extract_x_func, calc_total_mem_func)
        
        print(exp1_time)
        print(exp2_time)
        for i in range(len(nb_list_1)):
            nb_list_1[i] = str(int(float(nb_list_1[i]) // 10000))
        plt.figure()
        with plt.style.context('Solarize_Light2'):
            plt.plot(nb_list_1, exp1_time, linestyle = '-', linewidth = 1.5, marker='D', label='cpu')
            plt.plot(nb_list_1, exp2_time, linestyle = '-', linewidth = 1.5, marker='D', label='gpu')
            plt.xlabel("Database size (x10^4)")
            plt.xticks(nb_list_1)
            plt.ylabel("Time consumption (s)")
            plt.yscale("log")
            plt.legend()
        # plt.title("m_scan")
        plt.savefig(save_fig_pth)

    def parse_k_select_k(self, exp1, exp2, exp_cpu):
        extract_x_func = lambda config_line: int(config_line.split("_")[5][1:])
    
        log_file_pth = os.path.join(self.results_fd, exp1.name)
        save_fig_pth = os.path.join(self.results_fd, "{}.png".format(exp1.name))
        k_list, time_list = self.parse_k_minus(log_file_pth, extract_x_func)

        log_file_pth = os.path.join(self.results_fd, exp2.name)
        k_list_2, time_list2 = self.parse_k_minus(log_file_pth, extract_x_func)

        log_file_pth = os.path.join(self.results_fd, exp_cpu.name)
        k_list_cpu, nb_list, sort_list, pq_list, select_list = self.parse_priority_queue(log_file_pth)

        for i in range(len(k_list)):
            time_list[i] -= time_list2[i]
            k_list[i] = str(k_list[i])
        print(k_list, time_list, sort_list, pq_list, select_list)

        plt.figure()

        with plt.style.context('Solarize_Light2'):
            # plt.bar(k_list, time_list, color='crimson')
            plt.plot(k_list, time_list, label="GPU", color = 'crimson', marker='D')
            plt.plot(k_list, sort_list, label="CPU-sort", marker='D')
            plt.plot(k_list, pq_list, label="CPU-priority-queue", marker='D')
            plt.plot(k_list, select_list, label="CPU-select", marker='D')
            plt.xticks(k_list)

            # plt.xscale('log')
            plt.xlabel("k")
            plt.yscale('log')
            plt.ylabel("Time consumption (ms)")
            plt.legend()
        plt.savefig(save_fig_pth)

    def parse_k_select_nb(self, exp1, exp2, exp_cpu):
        extract_x_func = lambda config_line: int(config_line.split("_")[2][2:])

        log_file_pth = os.path.join(self.results_fd, exp1.name)
        save_fig_pth = os.path.join(self.results_fd, "{}.png".format(exp1.name))
        nb_list, time_list = self.parse_k_minus(log_file_pth, extract_x_func)

        log_file_pth = os.path.join(self.results_fd, exp2.name)
        nb_list_2, time_list2 = self.parse_k_minus(log_file_pth, extract_x_func)

        log_file_pth = os.path.join(self.results_fd, exp_cpu.name)
        k_list_cpu, nb_list, sort_list, pq_list, select_list = self.parse_priority_queue(log_file_pth)

        for i in range(len(nb_list)):
            time_list[i] -= time_list2[i]
            nb_list[i] = str(nb_list[i] // 10000)
        print(nb_list, time_list, sort_list, pq_list, select_list)

        plt.figure()

        with plt.style.context('Solarize_Light2'):
            # plt.bar(k_list, time_list, color='crimson')
            plt.plot(nb_list, time_list, label="GPU", color = 'crimson', marker='D')
            plt.plot(nb_list, sort_list, label="CPU-sort", marker='D')
            plt.plot(nb_list, pq_list, label="CPU-priority-queue", marker='D')
            plt.plot(nb_list, select_list, label="CPU-select", marker='D')
            plt.xticks(nb_list)

            # plt.xscale('log')
            plt.xlabel("nb (x10^4)")
            plt.yscale('log')
            plt.ylabel("Time consumption (ms)")
            plt.legend()
        plt.savefig(save_fig_pth)

    def parse_ivf_nlist(self, log_file_pth, save_fig_pth):
        with open(log_file_pth, "r") as f:
            lines = f.readlines()
            nlist_list = []
            time_list = []
            for i, line in enumerate(lines):
                if "****" in line:
                    config_line = lines[i + 1]
                    log_line = lines[i + 2]
                    print(log_file_pth)
                    print(line)
                    print(config_line)
                    print(log_line)
                    nlist = int(config_line.split('_')[3][5:])
                    time = float(log_line.split(":")[1])
                    nlist_list.append(nlist)
                    time_list.append(time)
        for i in range(len(nlist_list)):
            nlist_list[i] = str(nlist_list[i])
        plt.figure()
        # plt.plot(m_list, kernel_bandwidth_list, label = "kernel bandwidth")
        # plt.plot(m_list, actual_bandwidth_list, label = "actual bandwidth")
        with plt.style.context('Solarize_Light2'):
            plt.plot(nlist_list, time_list, linestyle = '-', marker='D')
            plt.xlabel("nlist (#Voronoi centoirds)")
            plt.ylabel("Time consumption (ms)")
            plt.legend()
        # plt.title("m_scan")
        plt.savefig(save_fig_pth)

    def parse_ivf_nprobe(self, log_file_pth, save_fig_pth):
        with open(log_file_pth, "r") as f:
            lines = f.readlines()
            nprobe_list = []
            time_list = []
            for i, line in enumerate(lines):
                if "*" in line:
                    config_line = lines[i + 1]
                    log_line = lines[i + 2]
                    print(config_line)
                    nprobe = int(config_line.split('_')[4][6:])
                    time = float(log_line.split(":")[1])
                    nprobe_list.append(nprobe)
                    time_list.append(time)
        plt.figure()
        for i in range(len(nprobe_list)):
            nprobe_list[i] = str(nprobe_list[i])
        # plt.plot(m_list, kernel_bandwidth_list, label = "kernel bandwidth")
        # plt.plot(m_list, actual_bandwidth_list, label = "actual bandwidth")
        with plt.style.context('Solarize_Light2'):
            plt.plot(nprobe_list, time_list, linestyle = '-', marker='D')
            plt.xlabel("nprobe (#Voronoi centoirds to select from)")
            plt.ylabel("Time consumption (ms)")
            plt.legend()
        # plt.title("m_scan")
        plt.savefig(save_fig_pth)

    def parse_priority_queue(self, log_file_pth):
        with open(log_file_pth, "r") as f:
            lines = f.readlines()
            k_list = []
            nb_list = []
            sort_list = []
            pq_list = []
            select_list = []

            for i, line in enumerate(lines):
                if "*" in line:
                    config_line = lines[i + 1]
                    log_line = lines[i + 2]
                    k = int(config_line.split("_")[0][1:])
                    nb = int(config_line.split("_")[1][2:])
                    sort = float(log_line.split(",")[0].split(":")[1])
                    pq = float(log_line.split(",")[1].split(":")[1])
                    select = float(log_line.split(",")[2].split(":")[1])

                    k_list.append(k)
                    nb_list.append(nb)
                    sort_list.append(sort)
                    pq_list.append(pq)
                    select_list.append(select)
        return k_list, nb_list, sort_list, pq_list, select_list
    def parse_pq(self, log_file_pth, extract_x_func, calc_total_mem_func):
        with open(log_file_pth, "r") as f:
            lines = f.readlines()
            x_list = []
            actual_time_list = []
            actual_bandwidth_list = []
            for i, line in enumerate(lines):
                if "*" in line:
                    config_line = lines[i + 1]
                    # kernel_result_line = lines[i + 2]
                    j = i + 1
                    while j < len(lines):
                        if "runpq" in lines[j]:
                            break
                        j += 1
                    log_line = lines[j]
                    x = extract_x_func(config_line)
                    x_list.append(x)
                    actual_time = float(log_line.split(",")[2].split(":")[1]) / 1000. 
                    totalmem = calc_total_mem_func(x)
                    actual_bandwidth = totalmem / actual_time / 1e9
                    actual_time_list.append(actual_time)
                    actual_bandwidth_list.append(actual_bandwidth)
            return x_list, actual_time_list, actual_bandwidth_list

    def parse_pq_m(self, log_file_pth, save_fig_pth):
        extract_x_func = lambda config_line: int(config_line.split("_")[-1][1:])
        calc_total_mem_func = lambda x: float(1000000. * x * 1000000)
        m_list, actual_time, actual_bandwidth_list = \
            self.parse_pq(log_file_pth, extract_x_func, calc_total_mem_func)
        plt.figure()
        # plt.plot(m_list, kernel_bandwidth_list, label = "kernel bandwidth")
        # plt.plot(m_list, actual_bandwidth_list, label = "actual bandwidth")
        with plt.style.context('Solarize_Light2'):
            plt.plot(m_list, actual_bandwidth_list, linestyle = '-', marker='D', label='runtime')
            plt.axhline(y = 313, color = 'r', label='Peak memory bandwidth', linestyle='--')
            plt.xlabel("m (#subvectors per vector)")
            plt.ylabel("Throughput (GB/s)")
            plt.legend()
        # plt.title("m_scan")
        plt.savefig(save_fig_pth)

    def parse_pq_nb(self, log_file_pth, save_fig_pth):
        extract_x_func = lambda config_line: config_line.split("_")[2][2:]
        calc_total_mem_func = lambda x: float(x * 8 * 1000000)
        nb_list, actual_time_list, actual_bandwidth_list = \
            self.parse_pq(log_file_pth, extract_x_func, calc_total_mem_func)

        plt.figure()
        # plt.plot(nb_list, kernel_bandwidth_list, label = "kernel bandwidth")
        # plt.plot(nb_list, actual_bandwidth_list, label = "actual bandwidth")
        plt.plot(nb_list, actual_time_list, linestyle = '-', marker='D')

        # plt.axhline(y = 313, color = 'r')
        plt.xlabel("Database Size")
        plt.ylabel("Time on Stage PQDist")
        # plt.title("nb_scan")
        plt.savefig(save_fig_pth)  

    def parse_pq_nb_cpu(self, log_file_pth, save_fig_pth):
        extract_x_func = lambda config_line: config_line.split("_")[2][2:]
        calc_total_mem_func = lambda x: float(x * 8 * 1000000)
        nb_list, actual_time_list, actual_bandwidth_list = \
            self.parse_pq(log_file_pth, extract_x_func, calc_total_mem_func)
        l = len(nb_list) // 2
        actual_time_cpu = actual_time_list[:l]
        actual_time_gpu = actual_time_list[l:]
        plt.plot(nb_list[:l], actual_time_cpu, linestyle='-', marker='D')
        plt.plot(nb_list[l:],actual_time_gpu, linestyle='--', color='r', marker='D')
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
                    # totaltime = float(result_line.split(" ")[-3]) / 1000. # seconds
                    search_time = float(result_line.split("searchImpl:")[-1]) / 1000. # seconds
                    time_list.append(search_time)
        print(k_list)
        print(time_list)
        plt.figure()
        plt.plot(k_list, time_list)
        plt.xlabel("k")
        plt.ylabel("time")
        plt.title("k")
        plt.savefig(save_fig_pth)
    
    def parse_k_minus(self, log_file_pth, extract_x_func):
        with open(log_file_pth, "r") as f:
            lines = f.readlines()
            x_list = []
            time_list = []
            for i, line in enumerate(lines):
                if "*" in line:
                    config_line = lines[i + 1]
                    result_line = lines[i + 2]
                    x = extract_x_func(config_line)
                    x = int(config_line.split("_")[5][1:])
                    x_list.append(x)
                    search_time = float(result_line.split("searchImpl:")[-1]) # milli seconds
                    time_list.append(search_time)
        return x_list, time_list

def m_scan():
    runner = ExperimentRunner()
    analyzer = Analyzer(runner.results_fd)
    exps = [M_Scan_Experiment()]
    for exp in exps:
        runner.run_experiments(exp)
        analyzer.parse(exp)

def together():
    runner = ExperimentRunner()
    analyzer = Analyzer(runner.results_fd)
    exps = [Nb_Scan_CPU_Experiment(), Nb_Scan_GPU_10000_Experiment()]
    # exps = [M_Scan_Experiment()]
    for exp in exps:
        runner.run_experiments(exp)

    analyzer.parse_together(exps[0], exps[1])

def k_select_k():
    runner = ExperimentRunner()
    analyzer = Analyzer(runner.results_fd)

    exps = [K_Select_K_full(), K_Select_K_tominus()]
    for exp in exps:
        runner.run_experiments(exp)

    exp = K_Select_K_CPU()
    runner.program_fd = "."
    runner.run_experiments(exp)

    analyzer.parse_k_select_k(exps[0], exps[1], exp)

def k_select_nb():
    runner = ExperimentRunner()
    analyzer = Analyzer(runner.results_fd)

    exps = [K_Select_Nb_full(), K_Select_Nb_tominus()]
    for exp in exps:
        runner.run_experiments(exp)
    exp = K_Select_Nb_CPU()
    runner.program_fd = "."
    runner.run_experiments(exp)

    analyzer.parse_k_select_nb(exps[0], exps[1], exp)

def ivf_nlist():
    runner = ExperimentRunner()
    analyzer = Analyzer(runner.results_fd)
    exps = [IVF_Nlist_Experiment()]
    for exp in exps:
        runner.run_experiments(exp)
        analyzer.parse(exp)
def ivf_nprobe():
    runner = ExperimentRunner()
    analyzer = Analyzer(runner.results_fd)
    exps = [IVF_Nprobe_Experiment()]
    for exp in exps:
        runner.run_experiments(exp)
        analyzer.parse(exp)

if __name__ == '__main__':
    together()
    # k_select_k()
    