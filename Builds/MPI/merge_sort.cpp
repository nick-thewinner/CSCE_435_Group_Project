#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <iostream>
void merge(int* arr, int* temp, int low, int mid, int high) {
    int i = low;
    int j = mid + 1;
    int k = low;

    while (i <= mid && j <= high) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }

    while (i <= mid) {
        temp[k++] = arr[i++];
    }

    while (j <= high) {
        temp[k++] = arr[j++];
    }

    for (i = low; i <= high; i++) {
        arr[i] = temp[i];
    }
}

void mergeSort(int* arr, int* temp, int low, int high) {
    if (low < high) {
        int mid = (low + high) / 2;
        mergeSort(arr, temp, low, mid);
        mergeSort(arr, temp, mid + 1, high);
        merge(arr, temp, low, mid, high);
    }
}

void parallelMergeSort(int* global_arr, int* temp, int n) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_n = n / size;
    int *local_arr = (int*)malloc(local_n * sizeof(int));

    // Scatter the data to all processes
    MPI_Scatter(global_arr, local_n, MPI_INT, local_arr, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    // Local merge sort
    int *local_temp = (int*)malloc(local_n * sizeof(int));
    mergeSort(local_arr, local_temp, 0, local_n - 1);

    // Gather the sorted data to the root process
    MPI_Gather(local_arr, local_n, MPI_INT, global_arr, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    // Free local arrays
    free(local_arr);
    free(local_temp);

    // Perform merging on the root process
    if (rank == 0) {
        // You need to merge the sorted blocks two at a time
        for (int i = 1; i < size; i *= 2) {
            for (int j = 0; j < size - i; j += 2*i) {
                int low = j * local_n;
                int mid = (j + i) * local_n - 1;
                int high = (j + 2 * i) * local_n - 1 < n ? (j + 2 * i) * local_n - 1 : n - 1;
                merge(global_arr, temp, low, mid, high);
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("%d\n", size);
    int n = std::stoi(argv[1]); // Total number of elements
    int* arr = NULL;
    int* temp = (int*)malloc(n * sizeof(int));

    if (rank == 0) {
        // Initialize the array with random values on the root process
        arr = (int*)malloc(n * sizeof(int));
        for (int i = 0; i < n; i++) {
            arr[i] = rand() % 100;
        }
    }

    parallelMergeSort(arr, temp, n);

    if (rank == 0) {
        // Print the sorted array on the root process
        printf("Sorted Array: ");
        for (int i = 0; i < n; i++) {
            printf("%d ", arr[i]);
        }
        printf("\n");
        free(arr);
    }

    free(temp);
    MPI_Finalize();
    return 0;
}
