/*
 * Parallel bitonic sort using CUDA.
 * Compile with
 * nvcc bitonic_sort.cu
 * Based on http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm
 * License: BSD 3
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <cuda_runtime.h>
#include <cuda.h> 

int THREADS;
int BLOCKS;
int NUM_VALS;

// const char* bitonic_sort_step_region = "bitonic_sort_step";
// const char* cudaMemcpy_host_to_device = "cudaMemcpy_host_to_device";
// const char* cudaMemcpy_device_to_host = "cudaMemcpy_device_to_host";
const char* comp = "comp";
//const char* comp_large = "comp_large";
const char* main_region = "main_region";
// const char* comm = "comm";
//const char* comm_large = "comm_large";
const char* cudaMemcpy1 = "cudaMemcpy1";
const char* cudaMemcpy2 = "cudaMemcpy2";
const char* data_init = "data_init";

cudaEvent_t start_comp, stop_comp;
cudaEvent_t start_cudaMemcpy1, stop_cudaMemcpy1;
cudaEvent_t start_cudaMemcpy2, stop_cudaMemcpy2;
cudaEvent_t start_main, stop_main;
cudaEvent_t start_data_init, stop_data_init;

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

__global__ void bitonic_sort_step(float *dev_values, int j, int k)
{
  unsigned int i, ixj; /* Sorting partners: i and ixj */
  i = threadIdx.x + blockDim.x * blockIdx.x;
  ixj = i^j;

  /* The threads with the lowest ids sort the array. */
  if ((ixj)>i) {
    if ((i&k)==0) {
      /* Sort ascending */
      if (dev_values[i]>dev_values[ixj]) {
        /* exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
    if ((i&k)!=0) {
      /* Sort descending */
      if (dev_values[i]<dev_values[ixj]) {
        /* exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
  }
}

/**
 * Inplace bitonic sort using CUDA.
 */
void bitonic_sort(float *values)
{
  float *dev_values;
  size_t size = NUM_VALS * sizeof(float);

  cudaMalloc((void**) &dev_values, size);
  
  //MEM COPY FROM HOST TO DEVICE
  CALI_MARK_BEGIN(cudaMemcpy1);
  cudaEventRecord(start_cudaMemcpy1);
  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);
  cudaEventRecord(stop_cudaMemcpy1);
  cudaEventSynchronize(stop_cudaMemcpy1);
  CALI_MARK_END(cudaMemcpy1);

  dim3 blocks(BLOCKS,1);    /* Number of blocks   */
  dim3 threads(THREADS,1);  /* Number of threads  */

  int j, k;
  CALI_MARK_BEGIN(comp);
  // CALI_MARK_BEGIN(comp_large);
  cudaEventRecord(start_comp);
  /* Major step */
  for (k = 2; k <= NUM_VALS; k <<= 1) {
    /* Minor step */
    for (j=k>>1; j>0; j=j>>1) {
      //BITONIC_SORT_STEP 
      bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
      cudaDeviceSynchronize();
    }
  }
  cudaEventRecord(stop_comp);
  cudaEventSynchronize(stop_comp);
  // CALI_MARK_END(comp_large);
  CALI_MARK_END(comp);

  
  //MEM COPY FROM DEVICE TO HOST
  CALI_MARK_BEGIN(cudaMemcpy2);
  cudaEventRecord(start_cudaMemcpy2);
  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
  cudaEventRecord(stop_cudaMemcpy2);
  cudaEventSynchronize(stop_cudaMemcpy2);
  CALI_MARK_END(cudaMemcpy2);
  
  cudaFree(dev_values);
}

int main(int argc, char *argv[])
{

  CALI_MARK_BEGIN(main_region);
  cudaEventCreate(&start_main);
  cudaEventCreate(&stop_main);
  cudaEventRecord(start_main);

  THREADS = atoi(argv[1]);
  NUM_VALS = atoi(argv[2]);
  BLOCKS = NUM_VALS / THREADS;

  cudaEventCreate(&start_comp);
  cudaEventCreate(&stop_comp);
  cudaEventCreate(&start_cudaMemcpy1);
  cudaEventCreate(&stop_cudaMemcpy1);
  cudaEventCreate(&start_cudaMemcpy2);
  cudaEventCreate(&stop_cudaMemcpy2);
  cudaEventCreate(&start_data_init);
  cudaEventCreate(&stop_data_init);

  printf("Number of threads: %d\n", THREADS);
  printf("Number of values: %d\n", NUM_VALS);
  printf("Number of blocks: %d\n", BLOCKS);

  // Create caliper ConfigManager object
  cali::ConfigManager mgr;
  mgr.start();

  clock_t start, stop;

  CALI_MARK_BEGIN(data_init);
  cudaEventRecord(start_data_init);
  float *random_values = (float*) malloc( NUM_VALS * sizeof(float));

  array_fill(random_values, NUM_VALS);
  cudaEventRecord(stop_data_init);
  CALI_MARK_END(data_init);

  start = clock();
  bitonic_sort(random_values); /* Inplace */
  stop = clock();

  print_elapsed(start, stop);

  // Store results in these variables.
  float comp_time;
  float cudaMemcpy1_time;
  float cudaMemcpy2_time;
  float main_time;
  float data_init_time;

  cudaEventElapsedTime(&comp_time, start_comp, stop_comp);
  cudaEventElapsedTime(&cudaMemcpy1_time, start_cudaMemcpy1, stop_cudaMemcpy1);
  cudaEventElapsedTime(&cudaMemcpy2_time, start_cudaMemcpy2, stop_cudaMemcpy2);
  cudaEventElapsedTime(&data_init_time, start_data_init, stop_data_init);
  cudaEventRecord(stop_main);
  CALI_MARK_END(main_region);
  cudaEventElapsedTime(&main_time, start_main, stop_main);
  
  array_print(random_values, NUM_VALS);

  printf("Correct: %s", correctness_check(random_values, NUM_VALS) ? "true" : "false")
  
  adiak::init(NULL);
  adiak::user();
  adiak::launchdate();
  adiak::libraries();
  adiak::cmdline();
  adiak::clustername();
  adiak::value("Algorithm", "Bitonic_Sort");
  adiak::value("Programming_Model", "CUDA");
  adiak::value("Datatype", "float");
  adiak::value("SizeOfDatatype", sizeof(float));
  adiak::value("InputSize", NUM_VALS);
  adiak::value("InputType", "Random");
  adiak::value("num_threads", THREADS);
  adiak::value("num_blocks", BLOCKS);
  adiak::value("group_num", "11");
  adiak::value("implementation_source", "Online, AI");
  adiak::value("main", main_time);
  adiak::value("comp", comp_time);
  adiak::value("comm", cudaMemcpy1_time + cudaMemcpy2_time);
  adiak::value("data_init", data_init_time);

  // Flush Caliper output before finalizing MPI
  mgr.stop();
  mgr.flush();
}