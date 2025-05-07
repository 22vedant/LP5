#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>

using namespace std;

// Function to merge two halves
void merge(vector<int> &arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;
    vector<int> L(n1), R(n2);

    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (int i = 0; i < n2; i++)
        R[i] = arr[mid + 1 + i];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++; k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++; k++;
    }
}

// Sequential Merge Sort
void mergeSort(vector<int> &arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

// Parallel Merge Sort
void parallelMergeSort(vector<int> &arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        #pragma omp parallel sections
        {
            #pragma omp section
            parallelMergeSort(arr, left, mid);
            
            #pragma omp section
            parallelMergeSort(arr, mid + 1, right);
        }
        merge(arr, left, mid, right);
    }
}

int main() {
    const int size = 100000;
    vector<int> arr1(size), arr2(size);
    srand(time(0));
    
    for (int i = 0; i < size; i++) {
        arr1[i] = arr2[i] = rand() % 100000;
    }
    
    double start, end;
    
    start = omp_get_wtime();
    mergeSort(arr1, 0, size - 1);
    end = omp_get_wtime();
    cout << "Sequential Merge Sort Time: " << (end - start) << " seconds" << endl;
    
    start = omp_get_wtime();
    parallelMergeSort(arr2, 0, size - 1);
    end = omp_get_wtime();
    cout << "Parallel Merge Sort Time: " << (end - start) << " seconds" << endl;
    
    return 0;
}
