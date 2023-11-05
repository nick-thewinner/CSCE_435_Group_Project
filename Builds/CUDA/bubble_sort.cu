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

const char *cudaMemcpy1 = "cudaMemcpy1";
const char *cudaMemcpy2 = "cudaMemcpy2";
const char *comp = "comp";
const char *main_region = "main_region";
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
    std::cout << "Elapsed time: " << elapsed << "s\n";
}

float random_float()
{
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

void array_fill(float *arr, int length)
{
    srand(static_cast<unsigned int>(time(NULL)));
    for (int i = 0; i < length; ++i)
    {
        arr[i] = random_float();
    }
}

void array_print(float *arr, int length)
{
    for (int i = 0; i < length; ++i)
    {
        std::cout << arr[i] << " ";
    }
    std::cout << "\n";
}

bool correctness_check(float *arr, int length)
{
    for (int i = 1; i < length; i++)
    {
        if (arr[i - 1] > arr[i])
        {
            return false;
        }
    }
    return true;
}

__global__ void odd_swaps(float *random_vals, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if((i % 2 != 0) && (i < n-2) && (random_vals[i] >= random_vals[i+1])){
        float temp = random_vals[i];
        random_vals[i] = random_vals[i+1];
        random_vals[i+1] = temp;
    }
}

__global__ void even_swaps(float *random_vals, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if((i % 2 == 0) && (i < n-1) && (random_vals[i] >= random_vals[i+1])){
        float temp = random_vals[i];
        random_vals[i] = random_vals[i+1];
        random_vals[i+1] = temp;
    }
}



void bubble_sort(float *values)
{
    float *d_values;

    size_t size = NUM_VALS * sizeof(float);
    cudaMalloc(&d_values, size);

    CALI_MARK_BEGIN(cudaMemcpy1);
    cudaEventRecord(start_cudaMemcpy1);
    cudaMemcpy(d_values, values, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop_cudaMemcpy1);
    cudaEventSynchronize(stop_cudaMemcpy1);
    CALI_MARK_END(cudaMemcpy1);

    dim3 blocks(BLOCKS, 1);
    dim3 threads(THREADS, 1);

    CALI_MARK_BEGIN(comp);
    cudaEventRecord(start_comp);

    //invoke the kernel functions (both even swapping and odd swapping)
    int i = 0;
    while (i < NUM_VALS){
        even_swaps<<<threads, blocks>>>(d_values, NUM_VALS);
        odd_swaps<<<threads, blocks>>>(d_values, NUM_VALS);
        i++;
    }

    cudaDeviceSynchronize();
    cudaEventRecord(stop_comp);
    cudaEventSynchronize(stop_comp);
    CALI_MARK_END(comp);

    CALI_MARK_BEGIN(cudaMemcpy2);
    cudaEventRecord(start_cudaMemcpy2);
    cudaMemcpy(values, d_values, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_cudaMemcpy2);
    cudaEventSynchronize(stop_cudaMemcpy2);
    CALI_MARK_END(cudaMemcpy2);
    
    cudaFree(d_values);
}

int main(int argc, char *argv[])
{

    CALI_MARK_BEGIN(main_region);
    cudaEventCreate(&start_main);
    cudaEventCreate(&stop_main);
    cudaEventRecord(start_main);

    THREADS = std::stoi(argv[1]);
    NUM_VALS = std::stoi(argv[2]);
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
    bubble_sort(random_values); /* Inplace */
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
    adiak::value("Algorithm", "Bubble_Sort");
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
