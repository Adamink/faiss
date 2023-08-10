#include<iostream>
#include<random>
#include<vector>
#include<algorithm>
#include<ctime>
#include<queue>
#include<string>

int main(int argc, char **argv){
    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;
    int n = 100000;
    int k = 100;
    if(argc > 1) {
        k = std::stoi(argv[1]);
        n = std::stoi(argv[2]);
    }

    std::vector<float> a(n);
    for(int i = 0; i < n; i++) {
        a[i] = distrib(rng);
    }
    std::vector<float> b(a);
    std::vector<float> c(a);

    auto t1 = clock();
    std::sort(b.begin(), b.end());
    std::cout << "sort:" << double(clock() - t1) / CLOCKS_PER_SEC * 1000. << ","; // 300ms
    // for(int i = 0; i < 5; i++){
    //     std::cout << b[i] << " "; // check
    // }
    // std::cout << std::endl;

    auto t2 = clock();
    
    std::priority_queue<float> p(a.begin(), a.begin() + k);
    for(int i = k; i < n; i++){
        if(a[i] < p.top()) {
            p.pop();
            p.push(a[i]);
        }
    }
    std::cout << "priorityqueue:" << double(clock() - t2) / CLOCKS_PER_SEC * 1000. << ","; // 9ms
    // for(int i = 0; i < 5; i++){
    //     std::cout << b[i] << " "; // check
    // }
    // std::cout << std::endl;

    auto t3 = clock();
    std::nth_element(c.begin(), c.begin() + k, c.end());
    std::cout << "select:" << double(clock() - t3) / CLOCKS_PER_SEC * 1000.; // 19ms
    // CPU 2.2GHz => 0.019 * 2.2GHz = 42 CPU cycle / element
    // 1e6 * 4 / 19ms = 0.2GB/s
}