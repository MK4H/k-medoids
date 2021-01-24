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

    /*
    void runAssignmentConsolidatedInitial(
        const db_offset_t *dnIndexes,
        const FLOAT *dnData,
        const std::size_t nNumSig,
        const db_offset_t *dMedIndexes,
        const FLOAT *dMedData,
        const std::size_t numClusters,
        const FLOAT alpha,
        const std::size_t maxSignatureSize,
        std::size_t * dAssignments
    ) {

        // TODO: These are basically constant values, as nNumSig is known during creation of this class
        //      So precalculate these in constructor
        auto numBlocks = nNumSig / this->sigPerBlock + (nNumSig % this->sigPerBlock != 0 ? 1 : 0);
        auto numKernels = numBlocks / this->blocksPerKernel + (numBlocks % this->blocksPerKernel != 0 ? 1 : 0);
        auto maxSigPerKernel = this->sigPerBlock * this->blocksPerKernel;

        // Start the data transfers for first STREAMS kernels
        auto sigToProcess = nNumSig;
        for (std::size_t kernelID = 0; kernelID < numKernels && kernelID < STREAMS; ++kernelID, sigToProcess -= maxSigPerKernel) {
            auto kernelSig = std::min(sigToProcess, maxSigPerKernel);
            startDataTransfer(dnIndexes, dnData, maxSigPerKernel, kernelSig, kernelID, this->streams[kernelID]);
        }

        // TODO: The medoid data transfer blocks until all previously started data transfers are not finished
        consolidateMedoids(dMedIndexes, dMedData, numClusters, maxSignatureSize);

        sigToProcess = nNumSig;
        // Start the first kernels in each stream which already have their data transfer started
        for (std::size_t kernelID = 0; kernelID < numKernels && kernelID < STREAMS; ++kernelID, sigToProcess -= maxSigPerKernel) {
            auto stream = this->streams[kernelID % STREAMS];
            auto kernelSig = std::min(sigToProcess, maxSigPerKernel);
            startAssignmentKernel(
                dnIndexes,
                dnData,
                dMedIndexes,
                dMedData,
                numClusters,
                alpha,
                kernelID,
                kernelSig,
                maxSigPerKernel,
                stream,
                dAssignments
            );
            CUCH(cudaMemcpyAsync(dAssignments, hAssignments + kernel * maxSigPerKernel, kernelSig * sizeof(std::size_t), cudaMemcpyDeviceToHost, stream));
        }

        // Start additional kernels which will need to have their data transfers initiated
        for (auto kernelID = STREAMS; kernelID < numKernels; ++kernelID, sigToProcess -= maxSigPerKernel) {
            auto stream = this->streams[kernelID % STREAMS];
            auto kernelSig = std::min(sigToProcess, maxSigPerKernel);

            startDataTransfer(dnIndexes, dnData, maxSigPerKernel, kernelSig, kernelID, stream);
            // TODO: Run assignment kernel
            CUCH(cudaMemcpyAsync(kerdAssignments, hAssignments + kernelID * kernelSig, nNumSig * sizeof(std::size_t), cudaMemcpyDeviceToHost, stream));
        }
        CUCH(cudaDeviceSynchronize());
    }
    */

private:

    /*

    void startDataTransfer(
        const db_offset_t *dnIndexes,
        const FLOAT *dnData,
        std::size_t maxSigPerKernel,
        std::size_t numSig,
        std::size_t kernelID,
        cudaStream_t stream
    ) {
        CUCH(cudaMemcpyAsync(
            dnIndexes + kernelID * maxSigPerKernel,
            hIndexes + kernelID * maxSigPerKernel,
            numSig * std::size_t(db_offset_t),
            cudaMemcpyHostToDevice,
            stream));

        auto startSig = kernelID * maxSigPerKernel;
        auto dataStart = (startSig != 0 ? hIndexes[startSig - 1] : 0) * (DIM + 1);
        auto dataEnd = hIndexes[startSig + numSig] * (DIM + 1);
        CUCH(cudaMemcpyAsync(
            dnData + dataStart,
            hData,
            (dataEnd - dataStart) * std::size_t(FLOAT),
            cudaMemcpyHostToDevice,
            stream));
    }
    */

};

#endif