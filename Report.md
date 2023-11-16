# CSCE 435 Group project

## 1. Group members:
1. Nicholas Nguyen
2. Andrew Hooper
3. Vinay Mannem
4. Ibrahim Semary

## Form of communication
Form of communication: Discord
---

## 2. _due 10/25_ Project topic

Comparing various sorting algorithms

## 2. _due 10/25_ Brief project description (what algorithms will you be comparing and on what architectures)
- Bitonic Sort (CUDA)
- Bitonic Sort (MPI)
- Merge Sort (CUDA)
- Merge Sort (MPI)
- Quick Sort (CUDA)
- Quick Sort (MPI)
- Bubble Sort (CUDA)
- Bubble Sort (MPI)

## inputs
We will do inputs that encompass strong scaling and weak scaling

It will also have a reverse order input

## psuedo code
**MPI Merge Sort**
```
MPI_Init()
MPI_Comm_size(MPI_COMM_WORLD, num_processes)
MPI_Comm_rank(MPI_COMM_WORLD, my_rank)

function merge(arr, left, mid, right)
    # Merge two sorted arrays

function parallel_merge_sort(arr, left, right)
    if left < right
        mid = (left + right) / 2
        subarray_size = (mid - left + 1) / num_processes
        my_left = my_rank * subarray_size
        my_right = my_left + subarray_size
        my_left_subarray = arr[left + my_left : left + my_right]

        my_left_subarray = merge_sort(my_left_subarray)
        sorted_subarrays = MPI_Gather(my_left_subarray, root=0)

        if my_rank == 0
            result = merge(sorted_subarrays, 0, num_processes - 1)
            return result
        else
            return None

if my_rank == 0
    arr = generate_random_array()
else
    arr = None

arr = MPI_Scatter(arr, root=0)
sorted_arr = parallel_merge_sort(arr, 0, len(arr) - 1)

if my_rank == 0
    sorted_arr = MPI_Gather(sorted_arr, root=0)
    print("Sorted Array:", sorted_arr)

MPI_Finalize()
```
**MPI Bitonic Sort**
```
MPI_Init()
MPI_Comm_size(MPI_COMM_WORLD, num_processes)
MPI_Comm_rank(MPI_COMM_WORLD, my_rank)

function compare_and_swap(arr, i, j, direction)
    # Compare and swap two elements in the bitonic sequence

function bitonic_merge(arr, left, right, direction)
    if right > 1
        mid = right / 2

        # Perform bitonic merge recursively
        bitonic_merge(arr, left, mid, ascending)
        bitonic_merge(arr, left + mid, mid, descending)

        # Merge two bitonic sequences
        compare_and_swap(arr, left, left + mid, direction)
        bitonic_merge(arr, left, right, direction)

function parallel_bitonic_sort(arr, left, right, direction)
    if right > 1
        mid = right / 2

        # Sort the first half in ascending order
        parallel_bitonic_sort(arr, left, mid, ascending)
        
        # Sort the second half in descending order
        parallel_bitonic_sort(arr, left + mid, mid, descending)

        # Perform bitonic merge
        bitonic_merge(arr, left, right, direction)

if my_rank == 0
    arr = generate_random_array()
else
    arr = None

arr = MPI_Scatter(arr, root=0)
parallel_bitonic_sort(arr, 0, len(arr), ascending)

# Gather the sorted subarrays to get the final sorted array
sorted_arr = MPI_Gather(arr, root=0)

if my_rank == 0
    print("Sorted Array:", sorted_arr)

MPI_Finalize()
```
**MPI Quick Sort**
```
// Parallel Quick Sort using MPI (Pseudocode)

// Function to partition an array and return the pivot index
function partition(inputArray, low, high):
    pivot = inputArray[high]  // Choose the pivot element (in this case, the last element)
    i = low - 1              // Initialize an index for the smaller element

    for j from low to high - 1:
        if inputArray[j] <= pivot:
            Swap inputArray[i+1] and inputArray[j]
            Increment i

    Swap inputArray[i+1] and inputArray[high]  // Place the pivot element in its correct position
    return i + 1  // Return the pivot index

// Parallel Quick Sort function
function parallelQuickSort(inputArray, low, high):
    if low < high:
        pivotIndex = partition(inputArray, low, high)

        // Create communicator for splitting processes
        comm = MPI_COMM_WORLD

        // Calculate the number of processes in each subarray
        processesInLeftSubarray = pivotIndex - low + 1
        processesInRightSubarray = high - pivotIndex

        // Determine the rank of the process in the current communicator
        MPI_Comm_rank(comm, rank)
        MPI_Comm_size(comm, size)

        // Determine the communicator for the left and right subarrays
        if rank < processesInLeftSubarray:
            Split communicator into left_comm
        else:
            Split communicator into right_comm

        // Determine the low and high indices for the left and right subarrays
        if rank < processesInLeftSubarray:
            leftLow = low
            leftHigh = pivotIndex - 1
        else:
            rightLow = pivotIndex + 1
            rightHigh = high

        // Recursively sort the left and right subarrays
        if rank < processesInLeftSubarray:
            parallelQuickSort(inputArray, leftLow, leftHigh)
        else:
            parallelQuickSort(inputArray, rightLow, rightHigh)

// Main program
function main:
    Initialize MPI
    Get rank and size of the MPI communicator

    if rank == 0:
        Initialize inputArray with unsorted values

    // Scatter inputArray to all processes
    Scatter inputArray to all processes in MPI_COMM_WORLD

    // Call parallelQuickSort for each process
    parallelQuickSort(inputArray, 0, length(inputArray) - 1)

    // Gather sorted subarrays from all processes to reconstruct the sorted inputArray
    Gather sorted subarrays from all processes in MPI_COMM_WORLD to inputArray at rank 0

    if rank == 0:
        // The sorted inputArray is now available in inputArray[0]

    Finalize MPI

// Usage:
Call the main program to perform parallel quicksort using MPI
```
**MPI Bubble Sort**
```
MPI_Init()
MPI_Comm_size(MPI_COMM_WORLD, num_processes)
MPI_Comm_rank(MPI_COMM_WORLD, my_rank)

function parallel_bubble_sort(arr)
    for i from 0 to len(arr) - 1
        if i % 2 == 0
            # Even pass: Compare and swap adjacent elements
            for j from 0 to len(arr) - 2 step 2
                if arr[j] > arr[j + 1]
                    swap(arr[j], arr[j + 1])
        else
            # Odd pass: Compare and swap adjacent elements
            for j from 1 to len(arr) - 2 step 2
                if arr[j] > arr[j + 1]
                    swap(arr[j], arr[j + 1])

if my_rank == 0
    arr = generate_random_array()
else
    arr = None

# Scatter the array to all processes
arr = MPI_Scatter(arr, root=0)

# Perform parallel bubble sort
parallel_bubble_sort(arr)

# Gather the sorted subarrays to get the final sorted array
sorted_arr = MPI_Gather(arr, root=0)

if my_rank == 0
    print("Sorted Array:", sorted_arr)

MPI_Finalize()
```
**CUDA Merge Sort**
```
# Define constants and parameters
block_size = 256  # Choose an appropriate block size for your GPU
n = length of input array
num_blocks = n / block_size

# Allocate memory on the GPU
cudaMalloc(arr_gpu, n * sizeof(int))

# Transfer data from CPU to GPU
cudaMemcpy(arr_gpu, input_array, n * sizeof(int), cudaMemcpyHostToDevice)

# Define the CUDA kernel for merge sort
__global__ void merge_sort_kernel(int* arr, int left, int right) {
    // Get thread ID and block ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Implement merge sort algorithm here
    if (left < right) {
        int mid = (left + right) / 2;
        
        // Sort the left half
        merge_sort_kernel(arr, left, mid);
        
        // Sort the right half
        merge_sort_kernel(arr, mid + 1, right);
        
        // Merge the sorted halves
        merge(arr, left, mid, right);
    }
}

# Define the merge function for merging two sorted arrays
__device__ void merge(int* arr, int left, int mid, int right) {
    // Implement merge algorithm here
}

# Launch the merge sort kernel
merge_sort_kernel<<<num_blocks, block_size>>>(arr_gpu, 0, n - 1)

# Wait for kernel to finish
cudaDeviceSynchronize()

# Transfer sorted data back from GPU to CPU
cudaMemcpy(output_array, arr_gpu, n * sizeof(int), cudaMemcpyDeviceToHost)

# Free GPU memory
cudaFree(arr_gpu)
```
**CUDA Bitonic Sort**
```
# Define constants and parameters
block_size = 256  # Choose an appropriate block size for your GPU
n = length of input array
num_blocks = n / block_size

# Allocate memory on the GPU
cudaMalloc(arr_gpu, n * sizeof(int))

# Transfer data from CPU to GPU
cudaMemcpy(arr_gpu, input_array, n * sizeof(int), cudaMemcpyHostToDevice)

# Define the CUDA kernel for bitonic sort
__global__ void bitonic_sort_kernel(int* arr, int n) {
    // Get thread ID and block ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Implement bitonic sort algorithm here
    // Ensure proper thread cooperation for parallel sorting
}

# Launch the bitonic sort kernel
for i from 2 to n:
    for j from i / 2 down to 1:
        bitonic_sort_kernel<<<num_blocks, block_size>>>(arr_gpu, n, j, i)
        cudaDeviceSynchronize()     

# Transfer sorted data back from GPU to CPU
cudaMemcpy(output_array, arr_gpu, n * sizeof(int), cudaMemcpyDeviceToHost)

# Free GPU memory
cudaFree(arr_gpu)
```
**CUDA Quick Sort**
```
# Define constants and parameters
block_size = 256  # Choose an appropriate block size for your GPU
n = length of input array
num_blocks = n / block_size

# Allocate memory on the GPU
cudaMalloc(arr_gpu, n * sizeof(int))

# Transfer data from CPU to GPU
cudaMemcpy(arr_gpu, input_array, n * sizeof(int), cudaMemcpyHostToDevice)

# Define the CUDA kernel for quick sort
__global__ void quick_sort_kernel(int* arr, int left, int right) {
    // Get thread ID and block ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Implement quick sort algorithm here
}

# Launch the quick sort kernel
quick_sort_kernel<<<num_blocks, block_size>>>(arr_gpu, 0, n - 1)
cudaDeviceSynchronize()

# Transfer sorted data back from GPU to CPU
cudaMemcpy(output_array, arr_gpu, n * sizeof(int), cudaMemcpyDeviceToHost)

# Free GPU memory
cudaFree(arr_gpu)
```
**CUDA Bubble Sort**
```
# Define constants and parameters
block_size = 256  # Choose an appropriate block size for your GPU
n = length of input array
num_blocks = n / block_size

# Allocate memory on the GPU
cudaMalloc(arr_gpu, n * sizeof(int))

# Transfer data from CPU to GPU
cudaMemcpy(arr_gpu, input_array, n * sizeof(int), cudaMemcpyHostToDevice)

# Define the CUDA kernel for parallelized bubble sort
__global__ void bubble_sort_kernel(int* arr, int n) {
    // Get thread ID and block ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Implement parallel bubble sort algorithm here
    if (tid < n) {
        for (int i = 0; i < n - tid - 1; i++) {
            if (arr[i] > arr[i + 1]) {
                // Swap arr[i] and arr[i + 1]
                int temp = arr[i];
                arr[i] = arr[i + 1];
                arr[i + 1] = temp;
            }
        }
    }
}

# Launch the bubble sort kernel
bubble_sort_kernel<<<num_blocks, block_size>>>(arr_gpu, n)
cudaDeviceSynchronize()

# Transfer sorted data back from GPU to CPU
cudaMemcpy(output_array, arr_gpu, n * sizeof(int), cudaMemcpyDeviceToHost)

# Free GPU memory
cudaFree(arr_gpu)
```
## psuedo code

# Bitonic Sort MPI

- This C++ code implements a parallel version of the Bitonic Sort algorithm using the MPI library.

- The program begins by initializing the MPI environment and determining the rank, which is the unique identifier for each process and the siza and the total number of processes running in parallel. 

- Each process calculates the portion of the array it needs to sort based on the total number of elements n, which is provided as an input argument to the program.

- Once the local sorting is complete, all sorted subsets are gathered to the root process, the process with rank 0, using MPI_Gather. The root process then performs a final bitonic sort on the collected array to merge the sorted subsets into a single sorted array.

- Dynamic memory is freed at the end

# Bubble Sort MPI

- The program requires at least two processes to run. 

- The process with rank 0 populates a vector with a given number of random elements and sends this vector to process 1 using MPI_Send. 

- Process 1 receives the array with MPI_Recv and performs a bubble sort on it. 

- After sorting, the program ensures synchronization with MPI_Barrier. 

# quick sort MPI
- This is a parallel version of the Quick Sort algorithm using the MPI library.

- Initially, MPI is initialized, and each process identifies its rank within the MPI_COMM_WORLD communicator. The root process, rank 0,  allocates an array and populates it with random numbers. T
 
- The parallelQuickSort function then partitions this array into equal segments that are distributed to all processes with MPI_Scatter. Each process executes a local quick sort on its segment. 
 
- After sorting locally, the segments are gathered back to the root process using MPI_Gather. 
 
- The root process then performs a final quick sort to merge these sorted segments into a fully sorted array.

# merge sort MPI
- This is an MPI-based implementation of the parallel merge sort algorithm

- The array is then divided among all processes using MPI_Scatter, and each process performs a local merge sort on its part of the array. After sorting, the local arrays are gathered back using MPI_Gather.

- This recrusive algorithim keeps merging the sorted subarrays to get the final whole sorted array. 

# bitonic sort CUDA

- bitonic sort algorithm for sorting floating-point numbers on Nvidia GPUs. 

- This algorithim initializes an array with random values and uses the GPU's parallel processing capabilities to perform the sort operation. 

- Additionally, the code uses Adiak to collect metadata about the program's execution, such as input size and the number of threads and blocks used. 

# merge sort CUDA

Tbe merge sort code includes functions for generating an array of random floating-point numbers, sorting them using a merge sort algorithm parallelized with CUDA

- The process begins by defining constants and variables for computation, communication, data initialization, and the CUDA grid configuration (threads, blocks, and number of values). The merge kernel function performs the merging step of the merge sort algorithm on the GPU. The merge_sort function manages memory allocation on the GPU, data transfer between the host and device, and the iterative merging process. T

# Bubble sort CUDA

- This sorting algorithim consists of two CUDA kernel functions, odd_swaps and even_swaps, and a host function bubble_sort.

Kernel Functions:
- odd_swaps performs swapping on odd-indexed elements if they are larger than their immediate successors.
- even_swaps performs swapping on even-indexed elements under the same condition.
- These kernels are launched in parallel across multiple threads, where each thread operates on different indices of the array. By splitting the work into odd and even indices, the algorithm takes advantage of parallelism to speed up the traditional bubble sort.

Bubble Sort Function:
- Allocates memory on the GPU and copies the input array from the host to this memory.
- Sequentially calls the even_swaps and odd_swaps kernels a number of times proportional to the number of values to be sorted.
- Ensures all device operations are complete with cudaDeviceSynchronize.
- Copies the sorted array back to the host memory.

# quick sort CUDA
- The core of this code is the partition kernel, which is responsible for partitioning the array segments. Each thread takes a segment of the array, determined by stack_l and stack_h arrays, and partitions it around a pivot element, swapping elements to ensure all values less than the pivot are on the left, and all values greater are on the right. Following the partitioning, each thread dynamically updates the stack with indices of the new segments to be sorted, utilizing atomic operations to avoid conflicts when multiple threads try to update the stack size simultaneously.

- The normal function quick_sort manages the overall sorting process. It allocates memory on the GPU for the data and the stack arrays and copies the initial data from the host to the GPU. It uses a loop to repeatedly invoke the partition kernel until the array is fully sorted, with the sorted segments growing larger with each iteration. After sorting, it transfers the sorted array back to the host memory and frees the GPU memory. 
 
# Bitonic Random Input Type Plot CUDA
![image](https://github.com/nick-thewinner/CSCE_435_Group_Project/assets/123513631/3cb697e2-4545-4142-b112-d26209a60b98)

Observation:
- The data shows a pretty consistent time across all input values on all threads, but the higher the amount of input values the higher the time. This makes sense because overall there are more values to sort through. An interesting finding would be a spike for the the lowest input value amount when using higher thread counts, this was seen in a previous lab where at a point it becomes inefficient to have a higher thread count when it comes to a smaller set.

# Merge Random Input Type Plot CUDA
![image](https://github.com/nick-thewinner/CSCE_435_Group_Project/assets/123513631/6dc13810-ec9f-436d-80b8-62c505c80244)

Observation:
- The data shows a pretty consistent time across all input values on all threads, but the higher the amount of input values the higher the time. This makes sense because overall there are more values to sort through. Soemthing interesting that occured would be the nearly identical GPU time and overall run time for the highest input value amount.


# Bubble Random Input Type Plot CUDA

# Quick Random Input Type Plot CUDA

# Bitonic Random Input Type Plot MPI
![image](https://github.com/nick-thewinner/CSCE_435_Group_Project/assets/123513631/e092a409-d282-47da-aa72-869f47d0618a)

Observation:
- The data shows a pretty consistent increase in time as processes and input size increase, this may be because it is run on a network.


# Merge Random Input Type Plot MPI
![image](https://github.com/nick-thewinner/CSCE_435_Group_Project/assets/123513631/b8a17638-d599-4e6c-b2af-ad660b406b43)

Observation:
- Similar to bitonic sort MPI, the data shows a pretty consistent increase in time as processes and input size increase, this may be because it is run on a network.


# Bubble Random Input Type Plot MPI

# Quick Random Input Type Plot MPI

# Final Observations
- In CUDA it seems it is better for weak scaling because as we increase number of threads and input values, the time still seems to be consistent. Opposed to MPI where weak scaling does not seem to be optimal as there is a positive trend.


