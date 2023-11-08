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
const char* main_region = "main_region";
const char* cudaMemcpy1 = "cudaMemcpy1";
const char* cudaMemcpy2 = "cudaMemcpy2";
const char* data_init = "data_init";

int THREADS;
int BLOCKS;
int NUM_VALS;

void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.3fs\n", elapsed);
}

float random_float()
{
  return (float)rand()/(float)RAND_MAX;
}

void array_fill(float *arr, int length)
{
  srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = random_float();
  }
}

void array_print(float *arr, int length) 
{
  int i;
  for (i = 0; i < length; ++i) {
    printf("%1.3f ",  arr[i]);
  }
  printf("\n");
}

bool correctness_check(float *arr, int length) 
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


__global__ void merge(float *list, float *sorted, int n, int width) {
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

void merge_sort(float *values) {
    float *d_values, *d_sorted;
    size_t size = NUM_VALS * sizeof(float);
    float *temp;

    cudaMalloc((void**) &d_values, size);
    cudaMalloc((void**) &d_sorted, size);

    // Copy from host to device
    CALI_MARK_BEGIN(cudaMemcpy1);
    cudaMemcpy(d_values, values, size, cudaMemcpyHostToDevice);
    CALI_MARK_END(cudaMemcpy1);

    dim3 blocks(BLOCKS, 1);
    dim3 threads(THREADS, 1);

    int width;
    CALI_MARK_BEGIN(comp);
    for (width = 1; width < NUM_VALS; width *= 2) {
        merge<<<blocks, threads>>>(d_values, d_sorted, NUM_VALS, width);
        cudaDeviceSynchronize();

        // Swap pointers
        temp = d_values;
        d_values = d_sorted;
        d_sorted = temp;
    }
    CALI_MARK_END(comp);

    // Copy from device to host
    CALI_MARK_BEGIN(cudaMemcpy2);
    cudaMemcpy(values, d_values, size, cudaMemcpyDeviceToHost);
    CALI_MARK_END(cudaMemcpy2);

    cudaFree(d_values);
    cudaFree(d_sorted);
}

int main(int argc, char *argv[]) {

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

  CALI_MARK_BEGIN(data_init);
  float *random_values = (float*) malloc( NUM_VALS * sizeof(float));

  array_fill(random_values, NUM_VALS);
  CALI_MARK_END(data_init);

  start = clock();
  merge_sort(random_values); /* Inplace */
  stop = clock();

  print_elapsed(start, stop);

  // Store results in these variables.
  float comp_time;
  float cudaMemcpy1_time;
  float cudaMemcpy2_time;
  float main_time;
  float data_init_time;

  CALI_MARK_END(main_region);
  
  array_print(random_values, NUM_VALS);

  printf("Correct: %s", correctness_check(random_values, NUM_VALS) ? "true" : "false");
  
  adiak::init(NULL);
  adiak::user();
  adiak::launchdate();
  adiak::libraries();
  adiak::cmdline();
  adiak::clustername();
  adiak::value("Algorithm", "Merge_Sort");
  adiak::value("Programming_Model", "CUDA");
  adiak::value("Datatype", "float");
  adiak::value("SizeOfDatatype", sizeof(float));
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
