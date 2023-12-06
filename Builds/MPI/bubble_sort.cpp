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

bool correctness_check(const std::vector<int> &arr) {
    for (size_t i = 1; i < arr.size(); i++) {
        if (arr[i - 1] > arr[i]) {
            return false;
        }
    }
    return true;
}

void array_print(const std::vector<int> &arr) {
    for (int value : arr) {
        std::cout << value << " ";
    }
    std::cout << "\n";
}

int random_int() {
    return rand() % 100; // limiting the range for simplicity
}

void bubble_sort(std::vector<int> &arr) {
    bool swapped;
    do {
        swapped = false;
        for (size_t i = 0; i < arr.size() - 1; i++) {
            if (arr[i] > arr[i + 1]) {
                std::swap(arr[i], arr[i + 1]);
                swapped = true;
            }
        }
    } while (swapped);
}

void parallel_bubble_sort(std::vector<int>& arr, int world_rank, int world_size) {
    for (int phase = 0; phase < world_size; ++phase) {
        if (phase % 2 == 0) { // Even phase
            if (world_rank % 2 == 0 && world_rank < world_size - 1) {
                MPI_Sendrecv_replace(arr.data(), arr.size(), MPI_INT, world_rank + 1, 0, world_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                bubble_sort(arr);
            } else if (world_rank % 2 != 0) {
                MPI_Sendrecv_replace(arr.data(), arr.size(), MPI_INT, world_rank - 1, 0, world_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                bubble_sort(arr);
            }
        } else { // Odd phase
            if (world_rank % 2 != 0 && world_rank < world_size - 1) {
                MPI_Sendrecv_replace(arr.data(), arr.size(), MPI_INT, world_rank + 1, 0, world_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                bubble_sort(arr);
            } else if (world_rank % 2 == 0 && world_rank > 0) {
                MPI_Sendrecv_replace(arr.data(), arr.size(), MPI_INT, world_rank - 1, 0, world_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                bubble_sort(arr);
            }
        }
    }
}
std::vector<int> merge(const std::vector<int>& left, const std::vector<int>& right) {
    std::vector<int> result;
    unsigned left_i = 0, right_i = 0;

    while (left_i < left.size() && right_i < right.size()) {
        if (left[left_i] < right[right_i]) {
            result.push_back(left[left_i]);
            left_i++;
        } else {
            result.push_back(right[right_i]);
            right_i++;
        }
    }

    while (left_i < left.size()) {
        result.push_back(left[left_i]);
        left_i++;
    }

    while (right_i < right.size()) {
        result.push_back(right[right_i]);
        right_i++;
    }

    return result;
}

std::vector<int> merge_sort(const std::vector<int>& arr) {
    if (arr.size() <= 1) {
        return arr;
    }

    std::vector<int> left(arr.begin(), arr.begin() + arr.size() / 2);
    std::vector<int> right(arr.begin() + arr.size() / 2, arr.end());

    left = merge_sort(left);
    right = merge_sort(right);

    return merge(left, right);
}

std::vector<int> vector_fill(size_t num_elements, int sort_type) {
    std::vector<int> arr(num_elements);
    srand(static_cast<unsigned>(time(nullptr)));
    if (sort_type == 1) {
        for (auto &value : arr) {
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
        for (int i = 0; i < num_elements; ++i) {
            if(i <= static_cast<float>(num_elements) * 0.01) {
                arr[i] = random_int();
            } else {
                arr[i] = i;
            }
        }
        SORT_TYPE_STR = "1% perturbation";
    } else {
        printf("Invalid sort type.\n");
    }
    return arr;
}

int main(int argc, char *argv[]) {
    CALI_MARK_BEGIN(main_region);
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (argc < 3) {
        if (world_rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <number_of_elements> <sort_type>" << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    const int number_of_elements = std::stoi(argv[1]);
    const int sort_type = std::stoi(argv[2]);

    cali::ConfigManager mgr;
    mgr.start();

    if (world_size < 2) {
        std::cerr << "World size must be greater than 1 for " << argv[0] << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    std::vector<int> local_array(number_of_elements / world_size);

    if (world_rank == 0) {
        CALI_MARK_BEGIN(data_init);
        std::vector<int> host_array = vector_fill(number_of_elements, sort_type);
        CALI_MARK_END(data_init);

        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_large);
        CALI_MARK_BEGIN(send_m);
        for (int i = 1; i < world_size; ++i) {
            MPI_Send(&host_array[i * local_array.size()], local_array.size(), MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        CALI_MARK_END(send_m);
        CALI_MARK_END(comm_large);
        CALI_MARK_END(comm);

        local_array = std::vector<int>(host_array.begin(), host_array.begin() + local_array.size());
    } else {
        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_small);
        CALI_MARK_BEGIN(recv_m);
        MPI_Recv(local_array.data(), local_array.size(), MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        CALI_MARK_END(recv_m);
        CALI_MARK_END(comm_small);
        CALI_MARK_END(comm);
    }

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    parallel_bubble_sort(local_array, world_rank, world_size);
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    if (world_rank == 0) {
        std::vector<int> sorted_array(number_of_elements);
        MPI_Gather(local_array.data(), local_array.size(), MPI_INT, sorted_array.data(), local_array.size(), MPI_INT, 0, MPI_COMM_WORLD);
        std::vector<int> final_array = merge_sort(sorted_array);
        CALI_MARK_BEGIN(correct);
        if (correctness_check(final_array)) {
            std::cout << "The array is correctly sorted." << std::endl;
        } else {
            std::cout << "The array is not correctly sorted." << std::endl;
        }
        CALI_MARK_END(correct);

        // Uncomment to print the sorted array
        // array_print(sorted_array);
    } else {
        MPI_Gather(local_array.data(), local_array.size(), MPI_INT, nullptr, 0, MPI_INT, 0, MPI_COMM_WORLD);
    }

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(barrier);
    MPI_Barrier(MPI_COMM_WORLD);
    CALI_MARK_END(barrier);
    CALI_MARK_END(comm);

    CALI_MARK_END(main_region);

    adiak::init(NULL);
    adiak::user();
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();
    adiak::value("Algorithm", "Bubble_Sort");
    adiak::value("ProgrammingModel", "MPI");
    adiak::value("Datatype", "Int");
    adiak::value("SizeOfDatatype", sizeof(int));
    adiak::value("InputSize", number_of_elements);
    adiak::value("InputType", SORT_TYPE_STR);
    adiak::value("num_procs", world_size);
    adiak::value("group_num", "11");
    adiak::value("implementation_source", "Online, AI");

    mgr.stop();
    mgr.flush();

    MPI_Finalize();
    return 0;
}
