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
const char *main_region = "main_region";
const char *cudaMemcpy1 = "cudaMemcpy1";
const char *cudaMemcpy2 = "cudaMemcpy2";
const char *data_init = "data_init";

cudaEvent_t start_comp, stop_comp;
cudaEvent_t start_cudaMemcpy1, stop_cudaMemcpy1;
cudaEvent_t start_cudaMemcpy2, stop_cudaMemcpy2;
cudaEvent_t start_main, stop_main;
cudaEvent_t start_data_init, stop_data_init;

int THREADS;
int BLOCKS;
int NUM_VALS;

void print_elapsed(clock_t start, clock_t stop)
{
    double elapsed = ((double)(stop - start)) / CLOCKS_PER_SEC;
    printf("Elapsed time: %.3fs\n", elapsed);
}

float random_float()
{
    return (float)rand() / (float)RAND_MAX;
}

void array_fill(float *arr, int length)
{
    srand(time(NULL));
    int i;
    for (i = 0; i < length; ++i)
    {
        arr[i] = random_float();
    }
}

void array_print(float *arr, int length)
{
    int i;
    for (i = 0; i < length; ++i)
    {
        printf("%1.3f ", arr[i]);
    }
    printf("\n");
}

bool correctness_check(float *arr, int length)
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


__global__ void partition (float *vals, int *stack_l, int *stack_h, int n)
{
    int z = blockIdx.x*blockDim.x+threadIdx.x;
    d_size = 0;
    __syncthreads();
    if (z<n)
      {
        int h = stack_h[z];
        int l = stack_l[z];
        float x = vals[h];
        int i = (l - 1);
        float temp;
        
        for (int j = l; j <= h- 1; j++)
          {
            if (x > vals[j])
              {
                i++;
                temp = vals[i];
                vals[i] = vals[j];
                vals[j] = temp;
              }
          }
        temp = vals[i+1];
        vals[i+1] = vals[h];
        vals[h] = temp;
        int p_val = (i + 1);
        if ( p_val+1 < h )
          {
            int index = atomicAdd(&d_size, 1);
            stack_h[index] = h; 
            stack_l[index] = p_val+1;
          }
        if (p_val-1 > l)
          {
            int index = atomicAdd(&d_size, 1);
            stack_h[index] = p_val-1;  
            stack_l[index] = l;
          }
      }
}

void quick_sort(float *values)
{
    float *d_data;
    int* d_l;
    int *d_h;
    int lstack[ NUM_VALS ], hstack[ NUM_VALS ];
    int top = -1;
    lstack[ ++top ] = 0;
    hstack[ top ] = NUM_VALS - 1;

 
    size_t size = NUM_VALS * sizeof(float);
   
    cudaMalloc((void **)&d_data, size);
    cudaMalloc((void **)&d_h, size);
    cudaMalloc((void **)&d_l, size);

    // Memcpy host to device
    CALI_MARK_BEGIN(cudaMemcpy1);
    cudaEventRecord(start_cudaMemcpy1);

    cudaMemcpy(d_data, values, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, lstack, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, hstack, size, cudaMemcpyHostToDevice);
    
    cudaEventRecord(stop_cudaMemcpy1);
    cudaEventSynchronize(stop_cudaMemcpy1);
    CALI_MARK_END(cudaMemcpy1);

    // Launch kernel
    CALI_MARK_BEGIN(comp);
    cudaEventRecord(start_comp);
    int n_i = 1; 
    dim3 blocks(BLOCKS, 1);
    dim3 threads(THREADS, 1);
    while ( n_i > 0 )
    {
        partition<<<blocks,threads>>>( d_data, d_l, d_h, n_i);
        int answer;
        cudaMemcpyFromSymbol(&answer, d_size, sizeof(int), 0, cudaMemcpyDeviceToHost); 
        n_i = answer;
    }
    
    cudaEventRecord(stop_comp);
    cudaEventSynchronize(stop_comp);
    CALI_MARK_END(comp);

    // Memcpy device to host
    CALI_MARK_BEGIN(cudaMemcpy2);
    cudaEventRecord(start_cudaMemcpy2);
    cudaMemcpy(values, d_data, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_cudaMemcpy2);
    cudaEventSynchronize(stop_cudaMemcpy2);
    CALI_MARK_END(cudaMemcpy2);

    cudaFree(d_data);
    
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
    float *random_values = (float *)malloc(NUM_VALS * sizeof(float));

    array_fill(random_values, NUM_VALS);
    cudaEventRecord(stop_data_init);
    CALI_MARK_END(data_init);

    start = clock();
    quick_sort(random_values); /* Inplace */
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

    printf("Correct: %s", correctness_check(random_values, NUM_VALS) ? "true" : "false");

    adiak::init(NULL);
    adiak::user();
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();
    adiak::value("Algorithm", "Quick_Sort");
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

    return 0;
}