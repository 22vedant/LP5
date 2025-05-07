#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>
using namespace std;

// Merge function
void merge(vector<int>& arr, int left, int mid, int right) {
    vector<int> temp(right - left + 1);
    int i = left, j = mid + 1, k = 0;

    while (i <= mid && j <= right)
        temp[k++] = (arr[i] < arr[j]) ? arr[i++] : arr[j++];
    while (i <= mid) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];
    for (int i = 0; i < k; i++)
        arr[left + i] = temp[i];
}

// Serial Merge Sort
void serialMergeSort(vector<int>& arr, int left, int right) {
    if (left >= right) return;
    int mid = left + (right - left) / 2;
    serialMergeSort(arr, left, mid);
    serialMergeSort(arr, mid + 1, right);
    merge(arr, left, mid, right);
}

// Parallel Merge Sort
void parallelMergeSort(vector<int>& arr, int left, int right, int depth = 0) {
    if (left >= right) return;
    int mid = left + (right - left) / 2;

    if (depth < 4) {
        #pragma omp task shared(arr)
        parallelMergeSort(arr, left, mid, depth + 1);

        #pragma omp task shared(arr)
        parallelMergeSort(arr, mid + 1, right, depth + 1);

        #pragma omp taskwait
    } else {
        serialMergeSort(arr, left, mid);
        serialMergeSort(arr, mid + 1, right);
    }

    merge(arr, left, mid, right);
}

int main() {
    const int SIZE = 1000000;
    vector<int> original(SIZE);
    for (int i = 0; i < SIZE; ++i)
        original[i] = rand() % 1000000;

    // Serial sort
    vector<int> arrSerial = original;
    double startSerial = omp_get_wtime();
    serialMergeSort(arrSerial, 0, arrSerial.size() - 1);
    double endSerial = omp_get_wtime();
    double serialTime = endSerial - startSerial;

    // Parallel sort
    vector<int> arrParallel = original;
    double startParallel = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        parallelMergeSort(arrParallel, 0, arrParallel.size() - 1);
    }
    double endParallel = omp_get_wtime();
    double parallelTime = endParallel - startParallel;

    // Validate correctness
    bool correct = arrSerial == arrParallel;

    // Output timings and speedup
    cout << "Serial Merge Sort Time:   " << serialTime << " seconds\n";
    cout << "Parallel Merge Sort Time: " << parallelTime << " seconds\n";
    cout << "Speedup:                  " << serialTime / parallelTime << "x\n";
    cout << "Sorted correctly?         " << (correct ? "Yes" : "No") << "\n";

    return 0;
}
