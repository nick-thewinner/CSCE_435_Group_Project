#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <iostream>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
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
    // Start of Comm
    CALI_MARK_BEGIN(comm);
    // Start of CommLarge
    CALI_MARK_BEGIN(comm_large);
    MPI_Scatter(global_arr, local_n, MPI_INT, local_arr, local_n, MPI_INT, 0, MPI_COMM_WORLD);
    // End of CommLarge
    CALI_MARK_END(comm_large);
    // End of Comm
    CALI_MARK_END(comm);
    // Local merge sort
    int *local_temp = (int*)malloc(local_n * sizeof(int));
    // Start of Comp
    CALI_MARK_BEGIN(comp);
    // Start of CompSmall
    CALI_MARK_BEGIN(comp_small);
    mergeSort(local_arr, local_temp, 0, local_n - 1);
    // End of CommSmall
    CALI_MARK_END(comp_small);
    // End of Comm
    CALI_MARK_END(comp);

    // Gather the sorted data to the root process
    // Start of Comm 
    CALI_MARK_BEGIN(comm);
    // Start of CommLarge
    CALI_MARK_BEGIN(comm_large);
    MPI_Gather(local_arr, local_n, MPI_INT, global_arr, local_n, MPI_INT, 0, MPI_COMM_WORLD);
    // End of CommLarge
    CALI_MARK_END(comm_large);
    // End of Comm
    CALI_MARK_END(comm);

    // Free local arrays
    free(local_arr);
    free(local_temp);

    // Perform merging on the root process
    if (rank == 0) {
        // You need to merge the sorted blocks two at a time
        // Start of Comp 
        CALI_MARK_BEGIN(comp);
        // Start of CompLarge 
        CALI_MARK_BEGIN(comp_large);
        for (int i = 1; i < size; i *= 2) {
            for (int j = 0; j < size - i; j += 2*i) {
                int low = j * local_n;
                int mid = (j + i) * local_n - 1;
                int high = (j + 2 * i) * local_n - 1 < n ? (j + 2 * i) * local_n - 1 : n - 1;
                merge(global_arr, temp, low, mid, high);
            }
        }
        // End of CompLarge
        CALI_MARK_END(comp_large);
        // End of Comp
        CALI_MARK_END(comp);
    }
}

int main(int argc, char** argv) {
    // Start of Main
    CALI_MARK_BEGIN(main_region);
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
        // Start of Data Init
        CALI_MARK_BEGIN(data_init);
        arr = (int*)malloc(n * sizeof(int));
        for (int i = 0; i < n; i++) {
            arr[i] = rand() % 100;
        }
        // End of Data Init
        CALI_MARK_END(data_init);
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

    // End of Main
    CALI_MARK_END(main_region);

    adiak::init(NULL);
    adiak::user();
    adiak::launchdate();                                         // launch date of the job
    adiak::libraries();                                          // Libraries used
    adiak::cmdline();                                            // Command line used to launch the job
    adiak::clustername();                                        // Name of the cluster
    adiak::value("Algorithm", "Merge_Sort");                   // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI");                     // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "Int");                          // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int));              // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", n);                        // The number of elements in input dataset (1000)
    adiak::value("InputType", "Random");                        // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", size);                        // The number of processors (MPI ranks)
    adiak::value("group_num", "11");                     // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online, AI") // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();

    return 0;
}
