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



# Evaluation/Observations

The plots can be seen in this Google Doc: https://docs.google.com/document/d/1nc6wON_MOL7XFeRSu65jXC7z7FEGBnIomcGJwNWotKY/edit#heading=h.gngmqtlgzdt

# Bubble Plot CUDA
Observation: 
- Strong Scaling
    - The CUDA-based bubble sort implementation shows improved performance. The best performance being at the thread count of 256. As the number of threads increases, there is a slight reduction in execution times. The best performance at 256 threads might show that there might be a balance between parralization and resources. Beyond 256 threads, we hypothesize that there is increased overhead that leads to more hardm than good. 
- Speed Up
    - The speedup in the CUDA implementation of bubble sort strongly makes sense given the execution time. There is notable speedup until reaching 256 threads. Up to this point, increasing the amount of threads leads to enhanced performance. However, after 256 threads, the speedup begins to go down. This decrease we think is likely attributed to factors such as increased overhead.
- Weak Scaling
    - The graphs for quick sort are flat, and this is expected. This flatness demonstrates our quick sort implmentation can efficiently scale with an increase in the number of thread while maintaining a fixed problem size per thread.

# Bubble Plot MPI
Observation:
- Similar to the other plots, the data shows that the time is increasing as the number of processes increases and as the size of the array increases. The smallest array size, however, shows a relatively shallow slope in comparison to the other ones.

# Quick Plot CUDA
Observation: 
- Strong Scaling
    - Quick sort is a naturally a sequential algorithim. This is due to its recurisve and sequential nature. While we did parralilize the algorithim by paritioning the input array and treating them as seperate problems, the dependencies and synchronization points limit the benifts of parralizing it. Therefore, as the number of threads increases, there is not a significant decrease in time. The overhead of managing threads and synchronization cancels out the benefits of parallelization. There would be more significant changes if our implmentation of quick sort is more optimal.
- Speed up
    - On average, parallelized quicksort shows a slight speedup, though with somevariability. While attempts to parallelize the algorithm lead to slight improvements in execution time, the challenges of dependencies and synchronization persists. The variability in speedup most likely comes from the parallelization implmentation that was used. Despite the potential for increased parallelism, the overall benefits might size and characteristics of the input data, as well as the efficiencynot be consistently significant. 
- Weak Scaling
    - The graphs for quick sort are flat, and this is expected. This flatness demonstrates our quick sort implmentation can efficiently scale with an increase in the number of thread while maintaining a fixed problem size per thread.

# Quick Plot MPI
Observation:

# Bitonic Plot CUDA
Observation:
-   Strong Scaling
    -   For strong scaling, we can see that in our first set of plots with a smaller input size, everything seems to be pretty constant excluding random input type. For the most part we can tell that there is tiny increase in runtime as the tread increase and we can assume that may be from the overhead of having too many threads for a small set of data. This is also supported by the better downward trends as our input size grows. It also seems the trends seen are consistent across comp_large, comm, and main the only slight deviation would be a slight higher runtime on the smaller threads in main.
-   Speed up
    -   For the sped up plots, it seems our results from strong scaling are shown much better with half of the input types having a slight speedup intially then a drop, and the other half of the inputs seems to just decrease in speed up right outside the gate. It also seems that with the large and smaller input sizes the speed ups are not as good compared a size like 2^12. Interestingly enough, it also seems that the sorted input type was the worst, but this may be becaause even tho the data is sorted, it still gets split up and set in bitonic order then remerged together
-   Weak Scaling
    -   For the most part our data seems to be pretty constant for comp_large; however, with comm and main there seems to be some variation. The variation with comm seems to be expected as it's more abnormal with the smaller and larger thread count ends. The variation in main is very sparatic and we can't really correlate it much with anything beside it being an accumulation of multiple variables. Overall, the data seems to be correct when comparing comp_large across all input types as in comp_large plots the run times increase as the input size increases but the times seems to stay constant when the threads increase.
# Bitonic Plot MPI
Observation:
-   Strong Scaling
    -   For strong scaling, comp_large and comm are increasing over time for all of the input sizes. This makes sense for comm since there is more overhead for communicating between processes. The comp_large section is a little weird since more processes should split up the large computing and decrease the runtime, but one potential reason why it’s increasing is due to how we set up the section and that it performs one last merge. We can see this if we look at the comp section which shows a decrease in runtime with more processes. The main section decreases to a certain point on all 3 input sizes (4, 8, 32) and then increases in runtime. This shows that the implementation hits an optimal number of processes and then starts to be less efficient when the number of processes increases.
-   Speed Up
    -   For the speedup, the comp_large decreases in speedup. We weren’t really sure why it was decreasing, but when we looked at the comp section for speedup, the plot was increasing. Both the comm and main sections increased to a certain processor size and started to decrease afterward. The main makes sense we know it hits an optimal process count and becomes less efficient in a bigger process count. The comm section is interesting since the strong scale plot was decreasing for the most part, but the speedup graph shows it increasing to a certain process size and starting to fall off after.
-   Weak Scaling
    -   For weak scaling, comp_large is relatively flat. This shows that our program has good weak scaling for large computations. Both main and comm plots are increasing overall (with some plots decreasing and then increasing). This shows that the main and comm sections don't have good weak scaling since the runtimes increase with more processes.


# Merge Plot CUDA
Observation:
-   Strong Scaling
    -   For strong scaling, in our smaller input size plots, our trend lines are mostly constant or maybe very slightly descreasing. The only outlier would be for random and sorted input types. For the input type lines that do seem to be doing well, they seem to decrease in runtime intitally but then evetually increase backup. Although they go back up they don't reach their intial runtime so there will be some speedup.
-   Speed Up
    -   The speed up is essentailly what we expected to see based off of our strong scaling. On the input types that actually do speed up, they seem to have a good sharp increase at on the smaller count of threads, but as the thread counts increase there is a decrease in speedup but it still remains above 1. Interestingly enough for reverse and sorted input type, the shape of their lines are the complete opposite, with an intial drop in speed up and then the eventual increase which hits around or above 1.0.
-   Weak Scaling
    -   For comp_large it seems that the weak scaling plots are great with all of them being almost perfectly constant with no little to no positive or negative trend. For both comm and main there seems to be some random trends.

# Merge Plot MPI
Observation:
-   Strong Scaling
    -   For strong scaling, all three sections are increasing overall. The comp_large section is strictly increasing why the other two sections have a dip. The main section makes sense since it hits an optimal process count and starts to decrease in performance afterward. The comm section increase is most likely due to the increase in overhead for communication between processes. The comp_large section could be increasing due to needing to compute with multiplier threads and merge back together.
-   Speed Up
    -   For the speed up, all three sections are decreasing overall. The comp_large section decreasing reflects the overhead issue of computation on a single process for merging. The comm and main sections increase to a certain amount of processes and then decrease. For main this correlates to the optimal amount of processes, but the comm section is interesting since it should be slower with the overhead.
-   Weak Scaling
    -   For weak scaling, comp_large has good weak scaling as it is relatively flat. Both comm and main plots increase overall. This shows that both sections have poor weak scaling since the runtime is increasing due to overhead and other factors.

# Comparison Plot CUDA
    - Looking at the comparison graphs for all the CUDA implmentations, it is clear that bitonic and merge sort outperform bubble and quicksort significantly. This faster execution time of bitonic and merge sort is most likely because of their intrinsic parallel nature, making them better suited for parralelization. On the other hand, bubble and quicksort, are known as sequential algorithms, and face challenges in achieving efficient parallelization. 
    
    - Furthemore, the inefficiency in quicksort may be due to communication overhead between threads, a consequence of its recursive implmentation. This implmentation can be shown as a parrallel one because all the comm plots show an overhead that is significant. Additionally, another thing is that the parallelization of quicksort might not have been optimized. In contrast, bitonic and merge sort are easy to parralize, and that is shown in these comparison graphs.This analysis shows that naturally some algorthims perform better than others when parallelized and it also highlights potential impact that overhead might have in increasing execution times.
    
# Comparison Plot MPI

# Final Observations
- In CUDA it seems it is better for weak scaling because as we increase number of threads and input values, the time still seems to be consistent. Opposed to MPI where weak scaling does not seem to be optimal as there is a positive trend. It also seems that looking into our MPI implementaions vs CUDA, our MPI data seems to be more cosnsitent amongst the plots, this includes much more defined trend lines and better spikes and associaties when it comes to correlations such as threads. The data also seems to be more conssitent amongst every input type and more reliant on the thread count.

