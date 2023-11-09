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

function mergeSort(array, l, r, m) {
	n1 = m - l + 1
	n2 = r - m
	Left[n1+1]; // create left sorted array of size n1
	Right[n2+1]; // create right sorted array of size n2
	For i in range(n1):
		Left[i] = array[l+i-1]
	For j in range(n2):
		Right[j] = array[m+j]
	Left[n1+1] = inf
	Right[n2+1] = inf
	I = j = 1
	For k in range(l, r):
		If (Left[i] <= Right[i])
			Array[k] = Left[i]
			i+=1
		Else
			Array[k] = Right[j]
			j+=1

function BitonicSort(arr,direction):
   	 n = length of array
   	 if n > 1 then
        		BitonicSort(arr[1:n/2],”ascending”)
		BitonicSort(arr[n/2 + 1:n],”descending”)
		Merge(arr, direction)
	end
return

function Merge(arr, direction):
   	 n = length of array
   	if n > 1 then
        		For i = 1,2……n/2 do
			If arr[i] > arr[i+n/2]
				Swap places
			End
		End
		Merge(arr[1 : n/2], direction)
Merge(arr[n/2+1 : n], direction)
	end
return

function quicksort(array) {
	if (array.length > 1) 
		choose a pivot.
		while (there are items left in array) 
			if (item < pivot)
				Put item into subarray1;
			else	
				Put item into subarray2;
		end
		quicksort(subarray1);
		quicksort(subarray2);
	end
return

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
 





