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

int main() {
    const int SIZE = 10000;  // Try increasing for more visible effects
    vector<int> originalArray(SIZE);

    // Initialize random array
    srand(time(nullptr));
    for (int i = 0; i < SIZE; i++) {
        originalArray[i] = rand() % 100000;
    }

    // Create copies for both versions
    vector<int> arrSerial = originalArray;
    vector<int> arrParallel = originalArray;

    // Serial sort timing
    double startSerial = omp_get_wtime();
    serialBubbleSort(arrSerial);
    double endSerial = omp_get_wtime();

    // Parallel sort timing
    double startParallel = omp_get_wtime();
    parallelBubbleSort(arrParallel);
    double endParallel = omp_get_wtime();

    // Output results
    cout << "Serial Bubble Sort Time:   " << (endSerial - startSerial) << " seconds\n";
    cout << "Parallel Bubble Sort Time: " << (endParallel - startParallel) << " seconds\n";

    // Optional: verify correctness
    bool isEqual = (arrSerial == arrParallel);
    cout << "Arrays sorted identically? " << (isEqual ? "Yes" : "No") << endl;

    return 0;
}
