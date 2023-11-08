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

const char *comm = "comm";
const char *comm_large = "comm_large";
const char *comp = "comp";
const char *comp_large = "comp_large"
const char *main_region = "main_region";
const char *data_init = "data_init";

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

    // Start of Comp
    CALI_MARK_BEGIN(comp);
    // Start of Comp large
    CALI_MARK_BEGIN(comp_large);
    //invoke the kernel functions (both even swapping and odd swapping)
    int i = 0;
    while (i < NUM_VALS){
        even_swaps<<<threads, blocks>>>(d_values, NUM_VALS);
        odd_swaps<<<threads, blocks>>>(d_values, NUM_VALS);
        i++;
    }

    cudaDeviceSynchronize();
    // End of Comp Large
    CALI_MARK_END(comp_large);
    // End of Comp
    CALI_MARK_END(comp);

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
}

int main(int argc, char *argv[])
{
    // Start of Main
    CALI_MARK_BEGIN(main_region);

    THREADS = std::stoi(argv[1]);
    NUM_VALS = std::stoi(argv[2]);
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
    float *random_values = (float *)malloc(NUM_VALS * sizeof(float));

    array_fill(random_values, NUM_VALS);
    // End of Data Init
    CALI_MARK_END(data_init);

    start = clock();
    bubble_sort(random_values); /* Inplace */
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

    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();
}
