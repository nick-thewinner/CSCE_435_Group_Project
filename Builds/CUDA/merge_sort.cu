#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <cuda_runtime.h>
#include <cuda.h> 
#include <iostream>
#include <cstdlib>
#include <ctime>

const char* comp = "comp";
const char* comp_large = "comp_large";
const char* main_region = "main_region";
const char* comm = "comm";
const char* comm_large = "comm_large";
const char* data_init = "data_init";

int THREADS;
int BLOCKS;
int NUM_VALS;

void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.3fs\n", elapsed);
}

int random_int()
{
  return (int)rand()/(int)RAND_MAX;
}

void array_fill(int *arr, int length)
{
  srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = random_int();
  }
}

void array_print(int *arr, int length) 
{
  int i;
  for (i = 0; i < length; ++i) {
    printf("%1.3f ",  arr[i]);
  }
  printf("\n");
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


__global__ void merge(int *list, int *sorted, int n, int width) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int start = idx * width * 2;
    int mid = start + width;
    int end = start + 2 * width;
    int i = start;
    int j = mid;
    int k = start;

    if (start >= n) return;

    if (end > n) end = n;
    if (mid > n) mid = n;

    while (i < mid && j < end) {
        if (list[i] < list[j]) {
            sorted[k++] = list[i++];
        } else {
            sorted[k++] = list[j++];
        }
    }

    while (i < mid) sorted[k++] = list[i++];
    while (j < end) sorted[k++] = list[j++];
}

void merge_sort(int *values) {
    int *d_values, *d_sorted;
    size_t size = NUM_VALS * sizeof(int);
    int *temp;

    cudaMalloc((void**) &d_values, size);
    cudaMalloc((void**) &d_sorted, size);

    // Copy from host to device
    // Start of Comm
    CALI_MARK_BEGIN(comm);
    // Start of Comm Large
    CALI_MARK_BEGIN(comm_large);
    cudaMemcpy(d_values, values, size, cudaMemcpyHostToDevice);

    // End of Comm Large 
    CALI_MARK_END(comm_large);
    // End of Comm
    CALI_MARK_END(comm);

    dim3 blocks(BLOCKS, 1);
    dim3 threads(THREADS, 1);

    int width;
    // Start of Comp
    CALI_MARK_BEGIN(comp);
    // Start of Comp Large
    CALI_MARK_BEGIN(comp_large);
    for (width = 1; width < NUM_VALS; width *= 2) {
        merge<<<blocks, threads>>>(d_values, d_sorted, NUM_VALS, width);
        cudaDeviceSynchronize();

        // Swap pointers
        temp = d_values;
        d_values = d_sorted;
        d_sorted = temp;
    }
    // End of Comp Large 
    CALI_MARK_END(comp_large);
    // End of Comp
    CALI_MARK_END(comp);

    // Copy from device to host
    // Start of Comm
    CALI_MARK_BEGIN(comm);
    // Start of Comm Large 
    CALI_MARK_BEGIN(comm_large);
    cudaMemcpy(values, d_values, size, cudaMemcpyDeviceToHost);
    // End of Comm Large
    CALI_MARK_END(comm_large);
    // End of Comm
    CALI_MARK_END(comm);

    cudaFree(d_values);
    cudaFree(d_sorted);
}

int main(int argc, char *argv[]) {
  // Start of Main
  CALI_MARK_BEGIN(main_region);

  THREADS = atoi(argv[1]);
  NUM_VALS = atoi(argv[2]);
  BLOCKS = NUM_VALS / THREADS;

  printf("Number of threads: %d\n", THREADS);
  printf("Number of values: %d\n", NUM_VALS);
  printf("Number of blocks: %d\n", BLOCKS);

  // Create caliper ConfigManager object
  cali::ConfigManager mgr;
  mgr.start();

  clock_t start, stop;

  // Start of Data Init
  CALI_MARK_BEGIN(data_init);
  int *random_values = (int*) malloc( NUM_VALS * sizeof(int));

  array_fill(random_values, NUM_VALS);
  // End of Data Init
  CALI_MARK_END(data_init);

  start = clock();
  merge_sort(random_values); /* Inplace */
  stop = clock();

  print_elapsed(start, stop);

  // End of Main
  CALI_MARK_END(main_region);
  
  //array_print(random_values, NUM_VALS);
  if (correctness_check(random_values, NUM_VALS)) {
            std::cout << "The array is correctly sorted." << std::endl;
        } else {
            std::cout << "The array is not correctly sorted." << std::endl;
        }
  
  adiak::init(NULL);
  adiak::user();
  adiak::launchdate();
  adiak::libraries();
  adiak::cmdline();
  adiak::clustername();
  adiak::value("Algorithm", "Merge_Sort");
  adiak::value("Programming_Model", "CUDA");
  adiak::value("Datatype", "int");
  adiak::value("SizeOfDatatype", sizeof(int));
  adiak::value("InputSize", NUM_VALS);
  adiak::value("InputType", "Random");
  adiak::value("num_threads", THREADS);
  adiak::value("num_blocks", BLOCKS);
  adiak::value("group_num", "11");
  adiak::value("implementation_source", "Online, AI");

  // Flush Caliper output before finalizing MPI
  mgr.stop();
  mgr.flush();
}
