#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <cuda.h>

const char *comp = "comp";
const char *comp_large = "comp_large";
const char *main_region = "main";
const char *comm = "comm";
const char *comm_large = "comm_large";
const char *data_init = "data_init";
const char *correct = "correctness_check";
const char *cudaMem = "cudaMemcpy";

int THREADS;
int BLOCKS;
int NUM_VALS;
int SORT_TYPE; // 1: random, 2: reverse, 3: sorted, 4: 1%
std::string SORT_TYPE_STR;

void print_elapsed(clock_t start, clock_t stop)
{
    double elapsed = ((double)(stop - start)) / CLOCKS_PER_SEC;
    printf("Elapsed time: %.3fs\n", elapsed);
}

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
    for (i = 0; i < length; ++i)
    {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

bool correctness_check(int *arr, int length)
{
    int i;
    for (i = 1; i < length; i++)
    {
        if (arr[i - 1] > arr[i])
        {
            return false;
        }
    }
    return true;
}

__device__ int d_size;

__global__ void partition(int *vals, int *stack_l, int *stack_h, int n)
{
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    d_size = 0;
    __syncthreads();
    if (z < n)
    {
        int h = stack_h[z];
        int l = stack_l[z];
        int x = vals[h];
        int i = (l - 1);
        int temp;

        for (int j = l; j <= h - 1; j++)
        {
            if (x > vals[j])
            {
                i++;
                temp = vals[i];
                vals[i] = vals[j];
                vals[j] = temp;
            }
        }
        temp = vals[i + 1];
        vals[i + 1] = vals[h];
        vals[h] = temp;
        int p_val = (i + 1);
        if (p_val + 1 < h)
        {
            int index = atomicAdd(&d_size, 1);
            stack_h[index] = h;
            stack_l[index] = p_val + 1;
        }
        if (p_val - 1 > l)
        {
            int index = atomicAdd(&d_size, 1);
            stack_h[index] = p_val - 1;
            stack_l[index] = l;
        }
    }
}

void quick_sort(int *values)
{
    int *d_data;
    int *d_l;
    int *d_h;
    int lstack[NUM_VALS], hstack[NUM_VALS];
    int top = -1;
    lstack[++top] = 0;
    hstack[top] = NUM_VALS - 1;

    size_t size = NUM_VALS * sizeof(int);

    cudaMalloc((void **)&d_data, size);
    cudaMalloc((void **)&d_h, size);
    cudaMalloc((void **)&d_l, size);

    // Memcpy host to device
    // Start of Comm
    CALI_MARK_BEGIN(comm);
    // Start of Comm Large
    CALI_MARK_BEGIN(comm_large);
    // Start of cudaMemcpy
    CALI_MARK_BEGIN(cudaMem);
    cudaMemcpy(d_data, values, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, lstack, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, hstack, size, cudaMemcpyHostToDevice);
    // End of cudaMemcpy
    CALI_MARK_END(cudaMem);

    // End of Comm Large
    CALI_MARK_END(comm_large);
    // End of Comm
    CALI_MARK_END(comm);

    // Launch kernel
    // Start of Comp
    CALI_MARK_BEGIN(comp);
    // Start of Comp Large
    CALI_MARK_BEGIN(comp_large);
    int n_i = 1;
    dim3 blocks(BLOCKS, 1);
    dim3 threads(THREADS, 1);
    while (n_i > 0)
    {
        partition<<<blocks, threads>>>(d_data, d_l, d_h, n_i);
        int answer;
        cudaMemcpyFromSymbol(&answer, d_size, sizeof(int), 0, cudaMemcpyDeviceToHost);
        n_i = answer;
    }
    // End of Comp Large
    CALI_MARK_END(comp_large);
    // End of Comp
    CALI_MARK_END(comp);

    // Memcpy device to host
    // Start of Comm
    CALI_MARK_BEGIN(comm);
    // Start of Comm Large
    CALI_MARK_BEGIN(comm_large);
    // Start of cudaMemcpy
    CALI_MARK_BEGIN(cudaMem);
    cudaMemcpy(values, d_data, size, cudaMemcpyDeviceToHost);
    // End of cudaMemcpy
    CALI_MARK_END(cudaMem);
    // End of Comm Large
    CALI_MARK_END(comm_large);
    // End of Comm
    CALI_MARK_END(comm);

    cudaFree(d_data);
}

int main(int argc, char *argv[])
{
    // Start of Main
    CALI_MARK_BEGIN(main_region);

    THREADS = atoi(argv[1]);
    NUM_VALS = atoi(argv[2]);
    SORT_TYPE = atoi(argv[3]);
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
    int *random_values = (int *)malloc(NUM_VALS * sizeof(int));

    array_fill(random_values, NUM_VALS, SORT_TYPE);
    // End of Data Init
    CALI_MARK_END(data_init);

    start = clock();
    quick_sort(random_values); /* Inplace */
    stop = clock();

    print_elapsed(start, stop);

    array_print(random_values, NUM_VALS);
    // Start of correctness check
    CALI_MARK_BEGIN(correct);
    if (correctness_check(random_values, NUM_VALS))
    {
        std::cout << "The array is correctly sorted." << std::endl;
    }
    else
    {
        std::cout << "The array is not correctly sorted." << std::endl;
    }
    // End of correctness check
    CALI_MARK_END(correct);

    // End of Main
    CALI_MARK_END(main_region);

    adiak::init(NULL);
    adiak::user();
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();
    adiak::value("Algorithm", "Quick_Sort");
    adiak::value("Programming_Model", "CUDA");
    adiak::value("Datatype", "int");
    adiak::value("SizeOfDatatype", sizeof(int));
    adiak::value("InputSize", NUM_VALS);
    adiak::value("InputType", SORT_TYPE_STR);
    adiak::value("num_threads", THREADS);
    adiak::value("num_blocks", BLOCKS);
    adiak::value("group_num", "11");
    adiak::value("implementation_source", "Online, AI");

    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();

    return 0;
}