#include <iostream>
#include <vector>
#include <omp.h>
#include <cstdlib>
using namespace std;

int main() {
    const int SIZE = 1000;
    vector<int> arr(SIZE);
    
    // Initialize array with random numbers
    srand(time(0));
    for (int i = 0; i < SIZE; i++) {
        arr[i] = rand() % 1000;
    }
    
    int minVal = arr[0], maxVal = arr[0], sum = 0;
    double average;
    
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) reduction(+:sum)
    for (int i = 0; i < SIZE; i++) {
        if (arr[i] < minVal) minVal = arr[i];
        if (arr[i] > maxVal) maxVal = arr[i];
        sum += arr[i];
    }
    
    average = sum / (double)SIZE;
    
    cout << "Minimum Value: " << minVal << endl;
    cout << "Maximum Value: " << maxVal << endl;
    cout << "Sum: " << sum << endl;
    cout << "Average: " << average << endl;
    
    return 0;
}


