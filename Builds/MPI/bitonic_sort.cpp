#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <iostream>

const char* comp = "comp";
const char* comp_small = "comp_small";
const char* comp_large = "comp_large";
const char* comm = "comm";
const char* comm_small = "comm_small";
const char* comm_large = "comm_large";
const char* main_region = "main_region";
const char* cudaMemcpy1 = "cudaMemcpy1";
const char* cudaMemcpy2 = "cudaMemcpy2";
const char* data_init = "data_init";

void compareSwap(int *a, int *b, int dir)
{
    if ((*a > *b) == dir)
    {
        int temp = *a;
        *a = *b;
        *b = temp;
    }
}

void bitonicMerge(int *arr, int low, int cnt, int dir)
{
    if (cnt > 1)
    {
        int k = cnt / 2;
        for (int i = low; i < low + k; i++)
        {
            compareSwap(&arr[i], &arr[i + k], dir);
        }
        bitonicMerge(arr, low, k, dir);
        bitonicMerge(arr, low + k, k, dir);
    }
}

void bitonicSort(int *arr, int low, int cnt, int dir)
{
    if (cnt > 1)
    {
        int k = cnt / 2;
        bitonicSort(arr, low, k, 1);
        bitonicSort(arr, low + k, k, 0);
        bitonicMerge(arr, low, cnt, dir);
    }
}

int main(int argc, char **argv)
{
    // Start of Main
    CALI_MARK_BEGIN(main_region);
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = std::stoi(argv[1]); // Total number of elements
    int local_n = n / size;     // Number of elements per process

    // Start of Data Init
    CALI_MARK_BEGIN(data_init);
    int *arr = (int *)malloc(local_n * sizeof(int));
    int *recvbuf = NULL;
    if (rank == 0)
    {
        recvbuf = (int *)malloc(n * sizeof(int)); // Note that it's 'n', not 'local_n' because root will gather all data.
    }

    // Initialize local array with random values
    for (int i = 0; i < local_n; i++)
    {
        arr[i] = rand() % 100;
    }
    // End of Data Init
    CALI_MARK_END(data_init);

    // Each process sorts its local array
    // Start of Comp
    CALI_MARK_BEGIN(comp);
    // Start of CompSmall
    CALI_MARK_BEGIN(comp_small);
    bitonicSort(arr, 0, local_n, 1); // 1 for ascending order
    // End of CompSmall
    CALI_MARK_END(comp_small);
    // End of Comp 
    CAL_MARK_END(comp);

    // Gather all local arrays to the root process
    // Start of Comm
    CALI_MARK_BEGIN(comm);
    // Start of CommLarge
    CALI_MARK_BEGIN(comm_large);
    MPI_Gather(arr, local_n, MPI_INT, recvbuf, local_n, MPI_INT, 0, MPI_COMM_WORLD);
    // Start of CommLarge
    CALI_MARK_END(comm_large);
    // End of Comm
    CALI_MARK_END(comm);

    if (rank == 0)
    {
        // Final merge on the root process
        // Start of Comp 
        CALI_MARK_BEGIN(comp);
        // Start of CompLarge
        CALI_MARK_BEGIN(comp_large);
        bitonicSort(recvbuf, 0, n, 1); // 1 for ascending order
        // End of CompLarge
        CALI_MARK_END(comp_large);
        // End of Comp
        CALI_MARK_END(comp);

        // Print the sorted array - need to comment out this when running 
        printf("Sorted Array: ");
        for (int i = 0; i < n; i++)
        {
            printf("%d ", recvbuf[i]);
        }
        printf("\n");
    }

    free(arr);
    if (rank == 0)
    {
        free(recvbuf);
    }

    MPI_Finalize();

    adiak::init(NULL);
    adiak::user();
    adiak::launchdate();                                         // launch date of the job
    adiak::libraries();                                          // Libraries used
    adiak::cmdline();                                            // Command line used to launch the job
    adiak::clustername();                                        // Name of the cluster
    adiak::value("Algorithm", "Bitonic_Sort");                   // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI");                     // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "Int");                          // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int));              // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", n);                        // The number of elements in input dataset (1000)
    adiak::value("InputType", "Random");                        // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", size);                        // The number of processors (MPI ranks)
    adiak::value("group_num", "11");                     // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online, AI") // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    // End of Main
    CALI_MARK_END(main_region);
    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();

    return 0;
}