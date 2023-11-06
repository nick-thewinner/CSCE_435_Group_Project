#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <iostream>

void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

int partition(int* arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

void quickSort(int* arr, int low, int high) {
    if (low < high) {
        int pivot = partition(arr, low, high);
        quickSort(arr, low, pivot - 1);
        quickSort(arr, pivot + 1, high);
    }
}

void parallelQuickSort(int* arr, int low, int high) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("%d\n", size);
    int local_n = (high - low + 1) / size;
    int local_low = low + rank * local_n;
    int local_high = local_low + local_n - 1;

    int* local_arr = (int*)malloc(local_n * sizeof(int));

    // Scatter the data to all processes
    MPI_Scatter(arr, local_n, MPI_INT, local_arr, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    // Local quicksort
    quickSort(local_arr, 0, local_n - 1);

    // Gather the sorted data to the root process
    MPI_Gather(local_arr, local_n, MPI_INT, arr, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    // Perform final merge on the root process
    if (rank == 0) {
        quickSort(arr, low, high);
    }

    free(local_arr);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = std::stoi(argv[1]); // Total number of elements
    int* arr = NULL;

    if (rank == 0) {
        // Initialize the array with random values on the root process
        arr = (int*)malloc(n * sizeof(int));
        for (int i = 0; i < n; i++) {
            arr[i] = rand() % 100;
        }
    }

    parallelQuickSort(arr, 0, n - 1);

    if (rank == 0) {
        // Print the sorted array on the root process
        printf("Sorted Array: ");
        for (int i = 0; i < n; i++) {
            printf("%d ", arr[i]);
        }
        printf("\n");
        free(arr);
    }

    MPI_Finalize();
    return 0;
}