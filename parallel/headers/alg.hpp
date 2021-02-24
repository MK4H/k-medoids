#ifndef K_MEDOIDS_ALG_HPP
#define K_MEDOIDS_ALG_HPP

#include "cuda/cuda.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

#include <memory>
#include <vector>
#include <array>
#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <limits>

template<typename T>
void generateRandomPermutation(std::vector<T> &res, std::size_t count, std::size_t seed) {
    res.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
        res[i] = i;
    }

    std::mt19937 gen((unsigned int)seed);
    std::shuffle(res.begin(), res.end(), gen);
}

template<int DIM, typename FLOAT = float>
class CudaAlg
{
public:
    CudaAlg(
        int rank,
        int numRanks,
        std::size_t limit,
        std::size_t seed,
        // Make sure that the db instance lives longer than the instance of this class
        const DBSignatureList<DIM, FLOAT> &inputData,
        FLOAT alpha,
        std::size_t numClusters,
        std::size_t blockSize,
        std::size_t sigPerBlock,
        std::size_t blocksPerKernel,
        std::size_t cudaStreams
    )
        :   sharedMemSize(0),
            rank(rank),
            numRanks(numRanks),
            limit(limit),
            seed(seed),
            blockSize(blockSize),
            sigPerBlock(sigPerBlock),
            blocksPerKernel(blocksPerKernel),
            alpha(alpha),
            numClusters(numClusters),
            inputData(inputData)
    {

        // Adapted from https://on-demand.gputechconf.com/gtc/2014/presentations/S4236-multi-gpu-programming-mpi.pdf
        int numDevices = 0;
        CUCH(cudaGetDeviceCount(&numDevices)); // numDevices == ranks per node
        int localRank = rank % numDevices;
        CUCH(cudaSetDevice(localRank));
        cudaDeviceProp gpuProps;
        CUCH(cudaGetDeviceProperties(&gpuProps, localRank));
        sharedMemSize = gpuProps.sharedMemPerBlock;

        streams.resize(cudaStreams);
        for (std::size_t i = 0; i < cudaStreams; ++i) {
            CUCH(cudaStreamCreate(&streams[i]));
        }
    }

    virtual ~CudaAlg() {

    }

    virtual void initialize() = 0;
    virtual bool runIteration() = 0;
    virtual void fillResults(KMedoidsResults &results) const = 0;
    virtual void moveResults(KMedoidsResults &results) = 0;

protected:
    int rank;
    int numRanks;
    std::size_t limit;
    std::size_t seed;
    std::size_t sharedMemSize;
    std::size_t blockSize;
    std::size_t sigPerBlock;
    std::size_t blocksPerKernel;
    const FLOAT alpha;
    const std::size_t numClusters;
    const DBSignatureList<DIM, FLOAT> &inputData;

    std::vector<cudaStream_t> streams;

    MPI_Datatype assign_mpi_type = my_MPI_SIZE_T;
    MPI_Datatype score_mpi_type = std::is_same_v<FLOAT, float> ? MPI_FLOAT : MPI_DOUBLE;

    static std::size_t indexToFloatOffset(std::size_t index) {
        return index * (DIM + 1);
    }

private:

};

#endif