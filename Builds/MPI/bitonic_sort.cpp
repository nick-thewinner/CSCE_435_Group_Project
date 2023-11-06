#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <iostream>

void compareSwap(int* a, int* b, int dir) {
    if ((*a > *b) == dir) {
        int temp = *a;
        *a = *b;
        *b = temp;
    }
}

void bitonicMerge(int* arr, int low, int cnt, int dir) {
    if (cnt > 1) {
        int k = cnt / 2;
        for (int i = low; i < low + k; i++) {
            compareSwap(&arr[i], &arr[i + k], dir);
        }
        bitonicMerge(arr, low, k, dir);
        bitonicMerge(arr, low + k, k, dir);
    }
}

void bitonicSort(int* arr, int low, int cnt, int dir) {
    if (cnt > 1) {
        int k = cnt / 2;
        bitonicSort(arr, low, k, 1);
        bitonicSort(arr, low + k, k, 0);
        bitonicMerge(arr, low, cnt, dir);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = std::stoi(argv[1]); // Total number of elements
    int local_n = n / size; // Number of elements per process

    int* arr = (int*)malloc(local_n * sizeof(int));
    int* recvbuf = NULL;
    if(rank == 0) {
        recvbuf = (int*)malloc(n * sizeof(int));  // Note that it's 'n', not 'local_n' because root will gather all data.
    }   


    // Initialize local array with random values
    for (int i = 0; i < local_n; i++) {
        arr[i] = rand() % 100;
    }

    // Each process sorts its local array
    bitonicSort(arr, 0, local_n, 1); // 1 for ascending order

    // Gather all local arrays to the root process
    MPI_Gather(arr, local_n, MPI_INT, recvbuf, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Final merge on the root process
        bitonicSort(recvbuf, 0, n, 1); // 1 for ascending order

        // Print the sorted array
        printf("Sorted Array: ");
        for (int i = 0; i < n; i++) {
            printf("%d ", recvbuf[i]);
        }
        printf("\n");
    }

    free(arr);
    if(rank == 0) {
        free(recvbuf);
    }

    MPI_Finalize();

    return 0;
}