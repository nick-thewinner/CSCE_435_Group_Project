#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <iostream>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

const char* comp = "comp";
const char* comp_small = "comp_small";
const char* comp_large = "comp_large";
const char* comm = "comm";
const char* comm_small = "comm_small";
const char* comm_large = "comm_large";
const char* main_region = "main";
const char* data_init = "data_init";
const char* correct = "correctness_check";
const char* gather = "MPI_Gather";
const char* scatter = "MPI_Scatter";

int SORT_TYPE; // 1: random, 2: reverse, 3: sorted, 4: 1%
std::string SORT_TYPE_STR;

int random_int()
{
  return (int)rand();
}

void array_fill(int *arr, int length, int sort_type)
{
  //fill array with random values
  if (sort_type == 1) 
  {
    srand(time(NULL));
    int i;
    for (i = 0; i < length; ++i)
    {
      arr[i] = random_int();
    }
    SORT_TYPE_STR = "random";
  }
  //reverse sorted array
  else if (sort_type == 2) 
  {
    int i;
    for (i = 0; i < length; ++i)
    {
      arr[i] = length - i;
    }
    SORT_TYPE_STR = "reverse";
  }
  //sorted array
  else if (sort_type == 3) 
  {
    int i;
    for (i = 0; i < length; ++i)
    {
      arr[i] = i;
    }
    SORT_TYPE_STR = "sorted";
  }
  // 1% perturbation of the array
  else if(sort_type == 4)
  {
    int i;
    for (i = 0; i < length; ++i)
    {
      if(i <= static_cast<float>(length) * 0.01)
      {
        arr[i] = random_int();
      }
      else
      {
        arr[i] = i;
      }
    }
    SORT_TYPE_STR = "1% perturbation";
  }
  else
  {
    printf("Invalid sort type.\n");
  }
}

void array_print(int *arr, int length) 
{
  int i;
  for (i = 0; i < length; ++i) {
    std::cout << arr[i] << " ";
  }
  std::cout << std::endl;
}

bool correctness_check(int *arr, int length) 
{
    int i;
    for (i = 1; i < length; i++)  
    {
        if (arr[i - 1] > arr[i]) {
            return false;
        }
    }
    return true;
}

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
    SORT_TYPE = std::stoi(argv[2]);
    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();

    int n = std::stoi(argv[1]); // Total number of elements
    int local_n = n / size;     // Number of elements per process

    // Start of Data Init
    CALI_MARK_BEGIN(data_init);

    int *arr = nullptr;
    int *recvbuf = nullptr;

    if (rank == 0)
    {
        // Allocate memory for the entire array on the root process
        arr = (int *)malloc(n * sizeof(int));
        // Fill the array with random values
        array_fill(arr, n, SORT_TYPE);
    }

    // Allocate memory for the local array on each process
    recvbuf = (int *)malloc(local_n * sizeof(int));

    //array_print(arr, local_n);
    // End of Data Init
    CALI_MARK_END(data_init);

    // Start of Comm
    CALI_MARK_BEGIN(comm);
    // Start of Comm Large
    CALI_MARK_BEGIN(comm_large);
    // Start of MPI scatter
    CALI_MARK_BEGIN(scatter);
    // Scatter different portions of the array to each process
    MPI_Scatter(arr, local_n, MPI_INT, recvbuf, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    // End of MPI scatter
    CALI_MARK_END(scatter);
    // End of Comm Large
    CALI_MARK_END(comm_large);
    // End of Comm
    CALI_MARK_END(comm);

    // Each process sorts its local array
    // Start of Comp
    CALI_MARK_BEGIN(comp);
    // Start of CompSmall
    CALI_MARK_BEGIN(comp_small);
    bitonicSort(recvbuf, 0, local_n, 1); // 1 for ascending order
    // End of CompSmall
    CALI_MARK_END(comp_small);
    // End of Comp 
    CALI_MARK_END(comp);

    // Gather all local arrays to the root process
    // Start of Comm
    CALI_MARK_BEGIN(comm);
    // Start of CommLarge
    CALI_MARK_BEGIN(comm_large);
    // Start of MPI Gather
    CALI_MARK_BEGIN(gather);
    MPI_Gather(recvbuf, local_n, MPI_INT, arr, local_n, MPI_INT, 0, MPI_COMM_WORLD);
    // End of MPI Gather
    CALI_MARK_END(gather);
    // End of CommLarge
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
        bitonicSort(arr, 0, n, 1); // 1 for ascending order
        // End of CompLarge
        CALI_MARK_END(comp_large);
        // End of Comp
        CALI_MARK_END(comp);

        //array_print(arr, n);
        // Start of correctness check
        CALI_MARK_BEGIN(correct);
        if (correctness_check(arr,n)) {
            std::cout << "The array is correctly sorted." << std::endl;
        } else {
            std::cout << "The array is not correctly sorted." << std::endl;
        }
        // End of correctness check
        CALI_MARK_END(correct);
    }

    //free(arr);
    if (rank == 0)
    {
        free(recvbuf);
        free(arr);
    }

    // End of Main
    CALI_MARK_END(main_region);

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
    adiak::value("InputType", SORT_TYPE_STR);                        // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", size);                        // The number of processors (MPI ranks)
    adiak::value("group_num", "11");                     // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online, AI"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();

    MPI_Finalize();

    return 0;
}