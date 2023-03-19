#include <cstdio>
#include <iostream>
#include <cmath>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <chrono>
#define N 300000 // number of data points
#define D 30 // dimension of each data point
#define K 3 // number of nearest neighbors
#define BLOCK_SIZE_DIS 8 // number of threads per block for distance calculation
#define BLOCK_SIZE_KNN 1 // number of threads per block for knn calculation

class Timer {
public:
    Timer() {
        stopped=started=std::chrono::high_resolution_clock::now();
    };

    Timer& start() {
        started=std::chrono::high_resolution_clock::now();
        return *this;
    }

    Timer& stop() {
        stopped=std::chrono::high_resolution_clock::now();
        return *this;
    }

    double elapsed() {
        if(started!=stopped) {
            std::chrono::duration<double> elapsed = stopped - started;
            return elapsed.count();
        }
        return 0.0;
    }

    std::chrono::time_point<std::chrono::high_resolution_clock> started;
    std::chrono::time_point<std::chrono::high_resolution_clock> stopped;
};

//--------------------------------------------------------------------------------------//
//                                     CUDA Kernels                                     //
//--------------------------------------------------------------------------------------//

/**
 * This CUDA kernel calculates the Euclidean distance between a query point and a set of data points.
 *
 * input：dimension, num_points, data_input, query
 * @param dimension The dimension of each point (number of features).
 * @param num_points The number of data points to calculate distances for.
 * @param data_input An array containing the data points.
 * @param query An array containing the query point.
 *
 * output：distances
 * @param distances An array to store the calculated distances.
 *
 */
__global__ void euclidean_distance_kernel(float* distances, float* data_input, float* query, int dimension, int num_points) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_points) {
        float sum = 0;
        for (int i = 0; i < dimension; i++) {
            float diff = data_input[idx * dimension + i] - query[i];
            sum += diff * diff;
        }
        distances[idx] = sqrt(sum);
    }
}

/**
 * This CUDA kernel use a merge sort to sort the distances and indices.
 * It is used in the main function to find the k nearest neighbors.
 *
 * input：num_points, ind_temp, dis_temp
 * @param num_points The number of distances to sort.
 * @param ind_temp An array temprarly storing the indices to pass to the next iteration.
 * @param dis_temp An array temprarly storing the distances to pass to the next iteration.
 *
 * output：indices, distances
 * @param indices An array to store the indices of the sorted distances.
 * @param distances An array to store the sorted distances.
 *
 * @NOTE: The number of threads per block must be 1. And the total number of the data points must be a power of 2.
 */
__global__ void mergeBlocks_kernel(int* indices, float* distances, int* ind_temp, float* dis_temp, int sortedsize) {
    int id = blockIdx.x;

    //index for distances
    int index1 = id * 2 * sortedsize;
    int endIndex1 = index1 + sortedsize;
    int index2 = endIndex1;
    int endIndex2 = index2 + sortedsize;
    int targetIndex = id * 2 * sortedsize;

    //pesudo index for indices
    //let the indices divide and merge like distances
    int index1_ind = index1;
    int endIndex1_ind = endIndex1;
    int index2_ind = index2;
    int endIndex2_ind = endIndex2;
    int targetIndex_ind = targetIndex;

    int done = 0;
    while (!done)
    {
        //if the first block is not finished and the second block is not finished
        if ((index1 == endIndex1) && (index2 < endIndex2)) {
            dis_temp[targetIndex++] = distances[index2++];
            ind_temp[targetIndex_ind++] = indices[index2_ind++];
        }

        //if the second block is finished and the first block is not finished
        else if ((index2 == endIndex2) && (index1 < endIndex1)) {
            dis_temp[targetIndex++] = distances[index1++];
            ind_temp[targetIndex_ind++] = indices[index1_ind++];
        }

        //if the first block is smaller than the second block
        else if (distances[index1] < distances[index2]) {
            dis_temp[targetIndex++] = distances[index1++];
            ind_temp[targetIndex_ind++] = indices[index1_ind++];
        }

        //else
        else {
            dis_temp[targetIndex++] = distances[index2++];
            ind_temp[targetIndex_ind++] = indices[index2_ind++];
        }

        //if both blocks are finished, merge done.
        if ((index1 == endIndex1) && (index2 == endIndex2))
            done = 1;
    }
}



//--------------------------------------------------------------------------------------//
//                                     CPU functions                                    //
//--------------------------------------------------------------------------------------//


/**
 * This CPU function calculates the Euclidean distance between a query point and a set of data points.
 *
 * input：dimension, num_points, data_input, query
 * @param dimension The dimension of each point (number of features).
 * @param num_points The number of data points to calculate distances for.
 * @param data_input An array containing the data points.
 * @param query An array containing the query point.
 *
 * output：distances
 * @param distances An array to store the calculated distances.
 *
 */
float euclidean_distance_cpu(std::vector<float> &vec1, std::vector<float> &vec2) {
    float distance = 0.0;
    for (int j = 0; j < vec1.size(); j++) {
        float diff = vec1[j] -vec2[j];
        distance += diff * diff;
    }

    return sqrt(distance);
}

/**
 * want some CPU function to find the k nearest neighbors
 * to compare running time with the GPU function in the main function
 *
 */

 /*DOESN'T SEEMS TO WORK*/

void knn_cpu(std::vector<std::pair<std::vector<float>, float>> *distances, std::vector<std::vector<float>> &trainSet, std::vector<float>& queryData) {
    for (auto  j : trainSet) {
        distances->push_back(std::make_pair( j, euclidean_distance_cpu(queryData,j)));
    }

    std::sort(distances->begin(), distances->end(),[](const std::pair<std::vector<float>, float>& p1, const std::pair<std::vector<float>, float>& p2){
        return p1.second < p2.second;
    });
}

//--------------------------------------------------------------------------------------//
//                                     main function                                    //
//--------------------------------------------------------------------------------------//


/**
 * main function to see if GPU and CPU functions work and compare running time
 */

int main() {

    /*============== Initialize and Memory Allocated ==============*/

    // (HOST) Define and allocate memory for data input
    float *data_input;
    data_input = (float *) malloc(N * D * sizeof(float));



    // (HOST) Initialize data input
    for (int i = 0; i < N * D; i++) {
        data_input[i] = (float) rand() / (float) RAND_MAX;
    }
    std::vector<std::vector<float> > data_input_vec(N, std::vector<float>(D));
    for(int i = 0; i < N*D; i++) {
        data_input_vec[i / D][i % D] = data_input[i];
    }
//    std::vector<float> data_input_vec(data_input, data_input + N * D);
    /*

    //using this to test if distance is calculated correctly and if the sorting works
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            data_input[i * D + j] = i+1;
        }
    }

    */

    // (HOST) Define and allocate memory for query data
    float *data_query;
    data_query = (float *) malloc(D * sizeof(float));

    // (HOST) Initialize query data
    for (int i = 0; i < D; i++) {
        data_query[i] = (float) rand() / (float) RAND_MAX;
    }
    std::vector<float> data_query_vec(data_query, data_query + D);

    /*
    //using this to test if distance is calculated correctly and if the sorting works
    for (int i = 0; i <D; i++) {
        data_query[i] = -1;
    }

    */

    // (DEVICE) Copy input data to device memory
    float *d_input;
    cudaMalloc(&d_input, N * D * sizeof(float));
    cudaMemcpy(d_input, data_input, N * D * sizeof(float), cudaMemcpyHostToDevice);


    // (DEVICE) Copy query data to device memory
    float *d_query;
    cudaMalloc(&d_query, D * sizeof(float));
    cudaMemcpy(d_query, data_query, D * sizeof(float), cudaMemcpyHostToDevice);

    // (DEVICE) Allocate memory for temporary distances and indices
    float *d_dis_temp;
    cudaMalloc(&d_dis_temp, N * sizeof(float));
    int *d_ind_temp;
    cudaMalloc(&d_ind_temp, N * sizeof(int));

    // (DEVICE) Allocate memory for indices
    int *d_indices;
    cudaMalloc(&d_indices, N * sizeof(int));

    /*

    // Allocate memory for unsorted distances
    float* distances_unsorted = (float*)malloc(N * sizeof(float));
    cudaMemcpy(distances_unsorted, d_distances, N * sizeof(float), cudaMemcpyDeviceToHost);

    */

    // (DEVICE) Initialize indices and copy to device memory
    int *indices = (int *) malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        indices[i] = i;
    }
    cudaMemcpy(d_indices, indices, N * sizeof(int), cudaMemcpyHostToDevice);

    // (DEVICE) Allocate memory for distances
    float *d_distances;
    cudaMalloc(&d_distances, N * sizeof(float));

    // (HOST) Allocate memory for sorted distances and indices computed by GPU
    auto *distances_sorted = (float *) malloc(N * sizeof(float));
    int *indices_sorted = (int *) malloc(N * sizeof(int));

    // (HOST) Allocate memory for sorted distances and indices computed by CPU
    float * distances_sorted_CPU = (float *) malloc(N * sizeof(float));
    auto indices_sorted_CPU_vec = new std::vector<float>();


    /*============== GPU functions ==============*/

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaEventRecord(start));

    // Compute distances between current query point and data input
    euclidean_distance_kernel <<<(N + BLOCK_SIZE_DIS - 1) / BLOCK_SIZE_DIS, BLOCK_SIZE_DIS >>>(d_distances, d_input,
                                                                                               d_query, D, N);


    // Sort distances and indices
    int blocks = (N + BLOCK_SIZE_KNN - 1) / BLOCK_SIZE_KNN / 2;
    int sortedsize = BLOCK_SIZE_KNN;
    while (blocks > 0) {
        mergeBlocks_kernel <<<blocks, 1 >>>(d_indices, d_distances, d_ind_temp, d_dis_temp, sortedsize);
        cudaMemcpy(d_distances, d_dis_temp, N * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_indices, d_ind_temp, N * sizeof(int), cudaMemcpyDeviceToDevice);
        blocks /= 2;
        sortedsize *= 2;
    }
    checkCudaErrors(cudaEventRecord(stop));
    // Copy sorted distances and indices to host memory
    cudaMemcpy(indices_sorted, d_indices, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(distances_sorted, d_distances, N * sizeof(float), cudaMemcpyDeviceToHost);

//    std::vector<float> distances_sorted_vec_GPU(distances_sorted, distances_sorted + N);


    checkCudaErrors(cudaEventSynchronize(stop));

    printf("============== GPU ==============\n\n");

    // Compute and print kernel execution time
    float kernel_time;
    checkCudaErrors(cudaEventElapsedTime(&kernel_time, start, stop));
    printf("Kernel execution time\t\t\t: %f ms\n\n", kernel_time);

    // Print query point
    printf("Query point : ");
    for (int i = 0; i < D; i++) {
        printf("%f ", data_query[i]);
    }
    printf("\n\n");

    // Print sorted indices and distances
    printf("Find the indices of K nearest neighbors(GPU) : \n");
    for (int j = 0; j < K; j++) {
        printf("%d ", indices_sorted[j]);
    }
    printf("\n\n");

    printf("Find the distances of K nearest neighbors(GPU) : \n");
    for (int j = 0; j < K; j++) {
        printf("%f ", distances_sorted[j]);
    }
    printf("\n\n");


    /*============== CPU functions ==============*/

    // this knn_cpu function is not working properly
    // and i'm thinking to use different sorting algorithm on CPU to compare with GPU
    //  merge sort is necessary, we can add more sorting algorithm to compare
    auto started=std::chrono::high_resolution_clock::now();
    auto indices_sorted_CPU = new std::vector<std::pair<std::vector<float>, float>>();
    knn_cpu(indices_sorted_CPU,data_input_vec,data_query_vec);

    auto stopped =std::chrono::high_resolution_clock::now();

    /*============== Compare result ==============*/

//    QueryPerformanceCounter(&end_time);
//
    printf("============== CPU ==============\n\n");

    std::chrono::duration<double, std::milli> elapsed = stopped - started;
    std::cout << "CPU execution time\t\t\t:" << elapsed.count() <<"\n\n";

    // Print sorted indices and distances
//    printf("Find the indices of K nearest neighbors(CPU) : \n");
//    for (int j = 0; j < K; j++) {
//        printf("%f ", indices_sorted_CPU->at(j).second);
//    }
    printf("\n\n");

    printf("Find the distances of K nearest neighbors(CPU) : \n");
    for (int j = 0; j < K; j++) {
        printf("%f ", indices_sorted_CPU->at(j).second);
    }
    printf("\n\n");
//

    /*============== Free memory ==============*/

    // Free memory
    free(data_input);
    free(data_query);
    free(indices_sorted);
    free(distances_sorted);
    free(distances_sorted_CPU);
//    free(indices_sorted_CPU);
    free(indices);
    //free(distances_unsorted);
    cudaFree(d_dis_temp);
    cudaFree(d_ind_temp);
    cudaFree(d_query);
    cudaFree(d_input);
    cudaFree(d_distances);
    cudaFree(d_indices);

    return 0;
}
