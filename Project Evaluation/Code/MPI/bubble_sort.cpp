#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <mpi.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
const char *comp = "comp";
const char *comp_large = "comp_large";
const char *comp_small = "comp_small";
const char *main_region = "main";
const char *comm = "comm";
const char *comm_large = "comm_large";
const char *comm_small = "comm_small";
const char *data_init = "data_init";
const char *correct = "correctness_check";
const char *send_m = "MPI_Send";
const char *recv_m = "MPI_Recv";
const char *barrier = "MPI_Barrier";

std::string SORT_TYPE_STR;

bool correctness_check(const std::vector<int> &arr)
{
    for (size_t i = 1; i < arr.size(); i++)
    {
        if (arr[i - 1] > arr[i])
        {
            return false;
        }
    }
    return true;
}

void array_print(const std::vector<int> &arr)
{
    for (int value : arr)
    {
        std::cout << value << " ";
    }
    std::cout << "\n";
}

int random_int()
{
    return (int)rand();
}

void bubble_sort(std::vector<int> &arr)
{
    bool swapped;
    do
    {
        swapped = false;
        for (size_t i = 0; i < arr.size() - 1; i++)
        {
            if (arr[i] > arr[i + 1])
            {
                std::swap(arr[i], arr[i + 1]);
                swapped = true;
            }
        }
    } while (swapped);
}

std::vector<int> vector_fill(size_t num_elements, int sort_type)
{
    std::vector<int> arr(num_elements);
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    if (sort_type == 1) {
        for (auto &value : arr)
        {
            value = random_int();
        }
        SORT_TYPE_STR = "random";

    } else if (sort_type == 2) {
        int cnt = num_elements;
        for (auto &value : arr) {
            value = cnt;
            cnt--;
        }
        SORT_TYPE_STR = "reverse";

    } else if (sort_type == 3) {
        int cnt = 0;
        for (auto &value : arr) {
            value = cnt;
            cnt++;
        }
        SORT_TYPE_STR = "sorted";

    } else if (sort_type == 4) {
        int i;
            for (i = 0; i < num_elements; ++i)
            {
            if(i <= static_cast<float>(num_elements) * 0.01)
            {
                arr[i] = random_int();
            }
            else
            {
                arr[i] = i;
            }
        }
        SORT_TYPE_STR = "1% perturbation";
    } else {
        printf("Invalid sort type.\n");
    }
    return arr;
}

int main(int argc, char *argv[])
{
    // Start of Main
    CALI_MARK_BEGIN(main_region);
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    printf("%d\n", world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    const int number_of_elements = std::stoi(argv[1]);
    
    // printf("%d\n", world_size);
    //  Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();

    if (world_size < 2)
    {
        std::cerr << "World size must be greater than 1 for " << argv[0] << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(1);
    }

    if (world_rank == 0)
    {
        // Start of Data Init
        CALI_MARK_BEGIN(data_init);
        std::vector<int> host_array = vector_fill(number_of_elements, std::stoi(argv[2]));
        // End of Data Init
        CALI_MARK_END(data_init);
        // Start Comm
        CALI_MARK_BEGIN(comm);
        // Start of CommLarge
        CALI_MARK_BEGIN(comm_large);
        // Start of MPI send 
        CALI_MARK_BEGIN(send_m);
        MPI_Send(host_array.data(), number_of_elements, MPI_INT, 1, 0, MPI_COMM_WORLD);
        // End of MPI send
        CALI_MARK_END(send_m);
        // End of CommLarge
        CALI_MARK_END(comm_large);
        // End of Comm
        CALI_MARK_END(comm);
    }
    else if (world_rank == 1)
    {
        std::vector<int> device_array(number_of_elements);
        // Start of Comm
        CALI_MARK_BEGIN(comm);
        // Start of CommSmall
        CALI_MARK_BEGIN(comm_small);
        // Start of MPI recv
        CALI_MARK_BEGIN(recv_m);
        MPI_Recv(device_array.data(), number_of_elements, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // End of MPI recv
        CALI_MARK_END(recv_m);
        // End of Comm Small
        CALI_MARK_END(comm_small);
        // End of Comm
        CALI_MARK_END(comm);

        // Start of Comp
        CALI_MARK_BEGIN(comp);
        // Start of Comp Large
        CALI_MARK_BEGIN(comp_large);
        bubble_sort(device_array);
        // End of CompLarge
        CALI_MARK_END(comp_large);
        // End of Comp
        CALI_MARK_END(comp);

        //array_print(device_array);
        //  Start of correctness check
        CALI_MARK_BEGIN(correct);
        if (correctness_check(device_array))
        {
            std::cout << "The array is correctly sorted." << std::endl;
        }
        else
        {
            std::cout << "The array is not correctly sorted." << std::endl;
        }
        // End of correctness check
        CALI_MARK_END(correct);
    }
    // Start of Comm
    CALI_MARK_BEGIN(comm);
    // Start of MPI Barrier
    CALI_MARK_BEGIN(barrier);
    MPI_Barrier(MPI_COMM_WORLD);
    // End of MPI Barrier
    CALI_MARK_END(barrier);
    // End of Comm
    CALI_MARK_END(comm);

    // End of Main
    CALI_MARK_END(main_region);

    adiak::init(NULL);
    adiak::user();
    adiak::launchdate();                                 // launch date of the job
    adiak::libraries();                                  // Libraries used
    adiak::cmdline();                                    // Command line used to launch the job
    adiak::clustername();                                // Name of the cluster
    adiak::value("Algorithm", "Bubble_Sort");            // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI");             // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "Int");                     // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int));         // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", number_of_elements);       // The number of elements in input dataset (1000)
    adiak::value("InputType", SORT_TYPE_STR);                 // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", world_size);               // The number of processors (MPI ranks)
    adiak::value("group_num", "11");                     // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online, AI"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();

    MPI_Finalize();

    return 0;
}