#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <mpi.h>



#define N 100000

using namespace std;

void array_print(vector<int> arr)
{
    int i;
    for (i = 0; i < arr.size(); ++i)
    {
        printf("%1.3f ", arr[i]);
    }
    printf("\n");
}

bool correctness_check(vector<int> arr)
{
    int i;
    for (i = 1; i < arr.size(); i++)
    {
        if (arr[i - 1] > arr[i])
        {
            return false;
        }
    }
    return true;
}

// Merging two subarrays
void merge(vector<int> &array, int l, int m, int r)
{
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;

    vector<int> L(n1), R(n2);

    for (i = 0; i < n1; i++)
        L[i] = array[l + i];
    for (j = 0; j < n2; j++)
        R[j] = array[m + 1 + j];

    i = 0;
    j = 0;
    k = l;
    while (i < n1 && j < n2)
    {
        if (L[i] <= R[j])
        {
            array[k] = L[i];
            i++;
        }
        else
        {
            array[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1)
    {
        array[k] = L[i];
        i++;
        k++;
    }

    while (j < n2)
    {
        array[k] = R[j];
        j++;
        k++;
    }
}

void mergeSort(vector<int> &array, int l, int r)
{
    if (l < r)
    {
        int m = l + (r - l) / 2;

        mergeSort(array, l, m);
        mergeSort(array, m + 1, r);

        merge(array, l, m, r);
    }
}

int main(int argc, char **argv)
{
    int my_rank, p, source, dest, tag = 0;
    vector<int> array(N);

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    // Split the array across processes
    int elements_per_proc = N / p;
    vector<int> sub_array(elements_per_proc);
    MPI_Scatter(&array.front(), elements_per_proc, MPI_INT, &sub_array.front(), elements_per_proc, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process performs merge sort on their piece of data
    mergeSort(sub_array, 0, elements_per_proc - 1);

    // Gather the sorted sub-arrays into one
    MPI_Gather(&sub_array.front(), elements_per_proc, MPI_INT, &array.front(), elements_per_proc, MPI_INT, 0, MPI_COMM_WORLD);

    // At root, merge the result
    if (my_rank == 0)
    {
        for (int i = 1; i < p; i++)
        {
            merge(array, 0, i * elements_per_proc - 1, (i + 1) * elements_per_proc - 1);
        }
    }
    // Finalize MPI
    MPI_Finalize();

    printf("Correct: %s", correctness_check(random_values) ? "true" : "false");
    array_print(array);
    /*/
    adiak::init(NULL);
    adiak::launchdate();                           // launch date of the job
    adiak::libraries();                            // Libraries used
    adiak::cmdline();                              // Command line used to launch the job
    adiak::clustername();                          // Name of the cluster
    adiak::value("Algorithm", "Merge_Sort");       // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI");       // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "Float");             // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", NUM_VALS);           // The number of elements in input dataset (1000)
    adiak::value("InputType", "Random");           // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", num_procs);          // The number of processors (MPI ranks)
    adiak::value("group_num", "11");               // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online, AI");*/

    return 0;
}
