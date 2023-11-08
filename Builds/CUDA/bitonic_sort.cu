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

const char* comp = "comp";
const char* comp_large = "comp_large";
const char* main_region = "main_region";
const char* comm = "comm";
const char* comm_large = "comm_large";
const char* data_init = "data_init";

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
  // Start of Comm
  CALI_MARK_BEGIN(comm);
  // Start of Comm Large
  CALI_MARK_BEGIN(comm_large);
  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);
  // End of Comm Large
  CALI_MARK_END(comm_large);
  // End of Comm
  CALI_MARK_END(comm);

  dim3 blocks(BLOCKS,1);    /* Number of blocks   */
  dim3 threads(THREADS,1);  /* Number of threads  */

  int j, k;
  // Start of Comp
  CALI_MARK_BEGIN(comp);
  // Start of Comp Large
  CALI_MARK_BEGIN(comp_large);
  /* Major step */
  for (k = 2; k <= NUM_VALS; k <<= 1) {
    /* Minor step */
    for (j=k>>1; j>0; j=j>>1) {
      //BITONIC_SORT_STEP 
      bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
      cudaDeviceSynchronize();
    }
  }

  // End of Comp Large
  CALI_MARK_END(comp_large);
  // End of Comp
  CALI_MARK_END(comp);

  
  //MEM COPY FROM DEVICE TO HOST
  // Start of Comm
  CALI_MARK_BEGIN(comm);
  // Start of Comm Large
  CALI_MARK_BEIGN(comm_large);
  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
  // End of Comm Large
  CALI_MARK_END(comm_large);
  // End of Comm
  CALI_MARK_END(comm);

  cudaFree(dev_values);
}

int main(int argc, char *argv[])
{
  // Begin of Main
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
  float *random_values = (float*) malloc( NUM_VALS * sizeof(float));

  array_fill(random_values, NUM_VALS);
  // End of Data Init
  CALI_MARK_END(data_init);

  start = clock();
  bitonic_sort(random_values); /* Inplace */
  stop = clock();

  print_elapsed(start, stop);

  // End of Main
  CALI_MARK_END(main_region);
  
  array_print(random_values, NUM_VALS);

  printf("Correct: %s", correctness_check(random_values, NUM_VALS) ? "true" : "false");
  
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

  // Flush Caliper output before finalizing MPI
  mgr.stop();
  mgr.flush();
}