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
- Bitonic Sort (MPI + CUDA)
- Bitonic Sort (MPI)
- Merge Sort (MPI + CUDA)
- Merge Sort (MPI)
- Quick Sort (MPI + CUDA)
- Quick Sort (MPI)

We will use inputs of varying sizes and compare amongst each sort on multiple GPUs and on a CPU.

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