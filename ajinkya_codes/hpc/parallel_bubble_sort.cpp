#include <iostream>
#include <omp.h>
#include <vector>
#include <cstdlib>
#include <ctime>
using namespace std;

// Swap function to swap two elements
void swap(int &a, int &b) {
    int temp = a;
    a = b;
    b = temp;
}

// Sequential Bubble Sort
void bubbleSort(vector<int> &arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

// Parallel Bubble Sort using OpenMP
void parallelBubbleSort(vector<int> &arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        // Parallel for loop with OpenMP
        #pragma omp parallel for
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

int main() {
    const int size = 1000; // Size of the array
    vector<int> arr1(size), arr2(size);
    srand(time(0));
    
    // Generate random numbers
    for (int i = 0; i < size; i++) {
        arr1[i] = arr2[i] = rand() % 10000;
    }
    
    // Measure time for Sequential Bubble Sort
    double start = omp_get_wtime();
    bubbleSort(arr1);
    double end = omp_get_wtime();
    cout << "Sequential Bubble Sort Time: " << (end - start) << " seconds" << endl;
    
    // Measure time for Parallel Bubble Sort
    start = omp_get_wtime();
    parallelBubbleSort(arr2);
    end = omp_get_wtime();
    cout << "Parallel Bubble Sort Time: " << (end - start) << " seconds" << endl;
    
    return 0;
}


