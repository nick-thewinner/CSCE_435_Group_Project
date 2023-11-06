#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <mpi.h>

bool correctness_check(const std::vector<int>& arr) {
    for (size_t i = 1; i < arr.size(); i++) {
        if (arr[i - 1] > arr[i]) {
            return false;
        }
    }
    return true;
}

void array_print(const std::vector<int>& arr) {
    for (int value : arr) {
        std::cout << value << " ";
    }
    std::cout << "\n";
}

void bubble_sort(std::vector<int>& arr) {
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

std::vector<int> populate_random_array(size_t num_elements) {
    std::vector<int> arr(num_elements);
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    for (auto& value : arr) {
        value = std::rand() % 100 + 1;
    }
    return arr;
}

  int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    printf("%d\n", world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    const int number_of_elements = std::stoi(argv[1]);
    printf("%d\n", world_size);

    if (world_size < 2) {
        std::cerr << "World size must be greater than 1 for " << argv[0] << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(1);
    }

    if (world_rank == 0) {
        std::vector<int> host_array = populate_random_array(number_of_elements);
        MPI_Send(host_array.data(), number_of_elements, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } else if (world_rank == 1) {
        std::vector<int> device_array(number_of_elements);
        MPI_Recv(device_array.data(), number_of_elements, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        bubble_sort(device_array);
        array_print(device_array);

        if (correctness_check(device_array)) {
            std::cout << "The array is correctly sorted." << std::endl;
        } else {
            std::cout << "The array is not correctly sorted." << std::endl;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}