#include <iostream>
#include <vector>
#include <omp.h>
#include <cstdlib>  // For rand(), srand()
#include <ctime>    // For time()
using namespace std;

void serialBubbleSort(vector<int>& arr) {
    int n = arr.size();
    bool swapped;
    for (int i = 0; i < n - 1; i++) {
        swapped = false;
        for (int j = 0; j < n - 1 - i; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
                swapped = true;
            }
        }
        if (!swapped) break;
    }
}

void parallelBubbleSort(vector<int>& arr) {
    int n = arr.size();
    bool sorted = false;

    while (!sorted) {
        sorted = true;

        // Odd phase
        #pragma omp parallel for shared(arr)
        for (int i = 1; i < n - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                swap(arr[i], arr[i + 1]);
                sorted = false;
            }
        }

        // Even phase
        #pragma omp parallel for shared(arr)
        for (int i = 0; i < n - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                swap(arr[i], arr[i + 1]);
                sorted = false;
            }
        }
    }
}

void printArray(const vector<int>& arr) {
    for (int val : arr) {
        cout << val << " ";
    }
    cout << endl;
}

int main() {
    omp_set_num_threads(4);  // Set OpenMP threads

    int size = 1000;  // Larger size for benchmarking
    vector<int> arr(size);

    srand(time(0));
    for (int i = 0; i < size; ++i) {
        arr[i] = rand() % 100;
    }

    // Make copies
    vector<int> arr_serial = arr;
    vector<int> arr_parallel = arr;

    // Serial sort timing
    double start_serial = omp_get_wtime();
    serialBubbleSort(arr_serial);
    double end_serial = omp_get_wtime();
    double time_serial = end_serial - start_serial;

    // Parallel sort timing
    double start_parallel = omp_get_wtime();
    parallelBubbleSort(arr_parallel);
    double end_parallel = omp_get_wtime();
    double time_parallel = end_parallel - start_parallel;

    // Output results
    cout << "Sorted array:\n";
    printArray(arr_serial);  // or arr_parallel

    cout << "\n--- Timing Results ---\n";
    cout << "Serial Bubble Sort Time:   " << time_serial << " seconds\n";
    cout << "Parallel Bubble Sort Time: " << time_parallel << " seconds\n";

    return 0;
}

