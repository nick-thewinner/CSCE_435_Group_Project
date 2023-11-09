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
 





