#ifndef K_MEDOIDS_WORK_HPP
#define K_MEDOIDS_WORK_HPP

#include "mpi.h"

#include "commons.hpp"

#include "cuda/cuda.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

#include <memory>
#include <vector>
#include <array>
#include <algorithm>
#include <stdexcept>
#include <type_traits>

#include "kernels.cuh"
#include "helpers.hpp"
#include "results.hpp"


template<int DIM, typename FLOAT = float>
class CudaAlg
{
public:
    CudaAlg(
        int device,
        // Make sure that the db instance lives longer than the instance of this class
        const DBSignatureList<DIM, FLOAT> &inputData,
        std::size_t nodeDataOffset,
        std::size_t nodeDataSize,
        FLOAT alpha,
        std::size_t numClusters,
        std::size_t blockSize,
        std::size_t sigPerBlock,
        std::size_t blocksPerKernel,
        std::size_t cudaStreams
    )
        :   sharedMemSize(0),
            blockSize(blockSize),
            sigPerBlock(sigPerBlock),
            blocksPerKernel(blocksPerKernel),
            alpha(alpha),
            numClusters(numClusters),
            inputData(inputData),
            nodeDataOffset(nodeDataOffset),
            nodeDataSize(nodeDataSize)
    {

        CUCH(cudaSetDevice(device));
        cudaDeviceProp gpuProps;
        CUCH(cudaGetDeviceProperties(&gpuProps, device));
        sharedMemSize = gpuProps.sharedMemPerBlock;

        // NOTE: We could have separate sigPerBlock for Assignment kernels and Scores kernels, but this might be enough
        auto maxSigPerBlock = std::max(
            getMaxSigPerBlockAssignment(this->sharedMemSize, this->inputData.getMaxSignatureLength(), this->blockSize),
            getMaxSigPerBlockScores(this->sharedMemSize, this->inputData.getMaxSignatureLength(), this->blockSize)
        );

        if (this->sigPerBlock > maxSigPerBlock) {
            throw std::invalid_argument("SigPerBlock too large.");
        }

        if (this->sigPerBlock == 0) {
            this->sigPerBlock = maxSigPerBlock;
        }

        streams.resize(cudaStreams);
        for (std::size_t i = 0; i < cudaStreams; ++i) {
            CUCH(cudaStreamCreate(&streams[i]));
        }
    }

    virtual ~CudaAlg() {

    }

    virtual void initialize() = 0;
    virtual void runAssignment(const std::vector<std::size_t> &medoids, std::vector<std::size_t> &assignments) = 0;
    virtual void runScores(const std::vector<std::size_t> &assignments, std::vector<FLOAT> &scores) = 0;
    virtual bool computeMedoids(const std::vector<std::size_t> &assignments, const std::vector<FLOAT> &scores, std::vector<std::size_t> &medoid) = 0;

protected:
    std::size_t sharedMemSize;
    std::size_t blockSize;
    std::size_t sigPerBlock;
    std::size_t blocksPerKernel;
    const FLOAT alpha;
    const std::size_t numClusters;
    const DBSignatureList<DIM, FLOAT> &inputData;

    // In number of signatures
    const std::size_t nodeDataOffset;
    const std::size_t nodeDataSize;

    std::vector<cudaStream_t> streams;

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
    void runAssignment(

    ) {

    }

    void runScoresOverlapped() {

    }

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

template<int DIM, typename FLOAT = float>
class ConsCudaAlg : public CudaAlg<DIM, FLOAT>
{
public:
    ConsCudaAlg(
        int device,
        // Make sure that the db instance lives longer than the instance of this class
        const DBSignatureList<DIM, FLOAT> &inputData,
        std::size_t nodeDataOffset,
        std::size_t nodeDataSize,
        FLOAT alpha,
        std::size_t numClusters,
        std::size_t blockSize,
        std::size_t sigPerBlock,
        std::size_t blocksPerKernel,
        std::size_t cudaStreams
    )
        : CudaAlg<DIM, FLOAT>(device, inputData, nodeDataOffset, nodeDataSize, alpha, numClusters, blockSize, sigPerBlock, blocksPerKernel, cudaStreams)
    {
        this->numBlocks = this->inputData.size() / this->sigPerBlock + (this->inputData.size() % this->sigPerBlock != 0 ? 1 : 0);
        this->numKernels = this->numBlocks / this->blocksPerKernel + (this->numBlocks % this->blocksPerKernel != 0 ? 1 : 0);
        this->maxSigPerKernel = this->sigPerBlock * this->blocksPerKernel;

        CUCH(cudaMalloc(&this->dData, nodeDataSize * this->inputData.getMaxSignatureLength() * sizeof(FLOAT)));
        CUCH(cudaMalloc(&this->dIndexes, nodeDataSize * sizeof(db_offset_t)));
        CUCH(cudaMalloc(&this->dMedData, numClusters * this->inputData.getMaxSignatureLength() * sizeof(FLOAT)));
        CUCH(cudaMalloc(&this->dMedIndexes, numClusters * sizeof(db_offset_t)));

        /* TODO: We can split this to node assignments and target assignments
            We only need node assignment and target assignment large enough that
            we can run kernel in each CUDA stream
        */
        CUCH(cudaMalloc(&this->dAssignments, this->inputData.size() * sizeof(std::size_t)));

        // TODO: We also only need scores for node data
        CUCH(cudaMalloc(&this->dNScores, nodeDataSize * sizeof(FLOAT)));
    }

    ~ConsCudaAlg() override {
        CUCH(cudaFree(this->dNScores));
        CUCH(cudaFree(this->dAssignments));
        CUCH(cudaFree(this->dMedIndexes));
        CUCH(cudaFree(this->dMedData));
        CUCH(cudaFree(this->dIndexes));
        CUCH(cudaFree(this->dData));
    }

    void initialize() override {
        CUCH(cudaMemcpy(this->dData, this->inputData.data(), this->inputData.dataSize() * sizeof(FLOAT), cudaMemcpyHostToDevice));
        CUCH(cudaMemcpy(this->dIndexes, this->inputData.indexData(), this->inputData.size() * sizeof(db_offset_t), cudaMemcpyHostToDevice));
    }

    void runAssignment(const std::vector<std::size_t> &medoids, std::vector<std::size_t> &assignments) override {
        consolidateMedoids(medoids, this->dMedIndexes, this->dMedData, this->numClusters, this->inputData.getMaxSignatureLength());

        auto numBlocks = this->nodeDataSize / this->sigPerBlock + (this->nodeDataSize % this->sigPerBlock != 0 ? 1 : 0);
        auto numKernels = numBlocks / this->blocksPerKernel + (numBlocks % this->blocksPerKernel != 0 ? 1 : 0);
        auto maxSigPerKernel = this->sigPerBlock * this->blocksPerKernel;

        auto sigToProcess = this->nodeDataSize;
        for (std::size_t kernelID = 0; kernelID < numKernels; ++kernelID, sigToProcess -= maxSigPerKernel) {
            auto kernelSig = std::min(sigToProcess, maxSigPerKernel);
            auto stream = this->streams[kernelID % this->streams.size()];

            auto kernelBlocks = std::min(
                this->blocksPerKernel,
                (kernelSig / this->sigPerBlock) + (kernelSig % this->sigPerBlock != 0 ? 1 : 0)
            );
            auto kerDAssignments = this->dAssignments + kernelID * maxSigPerKernel;
            runComputeAssignmentsConsolidated<DIM, FLOAT>(
                this->dNIndexes + kernelID * maxSigPerKernel,
                this->dNData + this->inputData.indexData()[kernelID * maxSigPerKernel] * (DIM + 1),
                kernelSig,
                this->dMedIndexes,
                this->dMedData,
                this->numClusters,
                this->alpha,
                this->inputData.getMaxSignatureLength(),
                kerDAssignments,
                this->blockSize,
                kernelBlocks,
                stream
            );
            CUCH(cudaMemcpyAsync(kerDAssignments, assignments.data() + this->nodeDataOffset + kernelID * maxSigPerKernel, kernelSig * sizeof(std::size_t), cudaMemcpyDeviceToHost, stream));
        }
        CUCH(cudaDeviceSynchronize());
    }

    /** Runs the GetScores kernel to calculate scores of the data block assigned to this node
     *
     * Expects all data to be present on the GPU, does no data transfer apart from the assignments in and results out
     */
    void runScores(const std::vector<std::size_t> &assignments, std::vector<FLOAT> &scores) override {
        auto sigToProcess = this->nodeDataSize;
        // TODO: Split and overlap the copy
        CUCH(cudaMemcpy(this->dAssignments, assignments.data(), assignments.size() * sizeof(FLOAT), cudaMemcpyHostToDevice));
        for (std::size_t kernelID = 0; kernelID < numKernels; ++kernelID, sigToProcess -= this->maxSigPerKernel) {
            auto kernelSig = std::min(sigToProcess, this->maxSigPerKernel);
            auto stream = this->streams[kernelID % this->streams.size()];

            auto kerDScores = this->dNScores + kernelID * this->maxSigPerKernel;
            runGetScores<DIM, FLOAT>(
                this->dNIndexes + kernelID * this->maxSigPerKernel,
                this->dNData + this->inputData.indexData()[kernelID * this->maxSigPerKernel] * (DIM + 1),
                this->dAssignments + this->nodeDataOffset + kernelID * this->maxSigPerKernel,
                kernelSig,
                this->dIndexes,
                this->dData,
                this->inputData.size(),
                this->dAssignments,
                this->alpha,
                this->inputData.getMaxSignatureLength(),
                kernelID,
                kerDScores,
                this->blockSize,
                this->numBlocks,
                stream
            );
            CUCH(cudaMemcpyAsync(kerDScores, scores.data() + this->nodeDataOffset + kernelID * this->maxSigPerKernel, kernelSig * sizeof(std::size_t), cudaMemcpyDeviceToHost, stream));
        }
        CUCH(cudaDeviceSynchronize());
    }

    // TODO: Try restricted modifier
    bool computeMedoids(const std::vector<std::size_t> &assignments, const std::vector<FLOAT> &scores, std::vector<std::size_t> &medoids) override {
        // TODO: Preallocate in constructor
        std::vector<FLOAT> minScores(medoids.size());
        std::vector<std::size_t> newMedoids(medoids.size());
        // TODO: Use OpenMP
        bool changed = false;
        for (std::size_t i = 0; i < assignments.size(); ++i) {
            auto assignment = assignments[i];
            auto score = scores[i];
            if (score < minScores[assignment]) {
                newMedoids[assignment] = i;
                minScores[assignment] = score;

                if (medoids[assignment] != i) {
                    changed = true;
                }
            }
        }

        return changed;
    }

private:
    // All data
    FLOAT *dData;
    db_offset_t *dIndexes;

    // Node data
    FLOAT *dNData;
    db_offset_t *dNIndexes;

    // For consolidated medoids
    FLOAT *dMedData;
    db_offset_t *dMedIndexes;

    std::size_t *dAssignments;
    FLOAT *dNScores;

    std::size_t numBlocks;
    std::size_t numKernels;
    std::size_t maxSigPerKernel;

    void consolidateMedoids(
        const std::vector<std::size_t> &medoids,
        const db_offset_t *dMedIndexes,
        const FLOAT *dMedData,
        const std::size_t numClusters,
        const std::size_t maxSignatureSize
    ) {
        // TODO: Preallocate these arrays
        std::vector<db_offset_t> medIndexes(numClusters + 1);
        std::vector<FLOAT> medData(numClusters * maxSignatureSize * (DIM + 1));

        const FLOAT *data = this->inputData.data();
        const db_offset_t *indexes = this->inputData.indexData();

        medIndexes[0] = 0;
        for (std::size_t i = 0; i < medoids.size(); ++i) {
            auto medIdx = medoids[i];
            auto medStart = medIdx != 0 ? indexes[medIdx - 1] : 0;
            auto medEnd = indexes[medIdx];
            auto medSize = medEnd - medStart;
            medIndexes.push_back(medIndexes.back() + medSize);
            medData.insert(medData.end(), data + medStart * (DIM + 1), data + medEnd * (DIM + 1));
        }

        CUCH(cudaMemcpy(medIndexes.data(), dMedIndexes, medIndexes.size() * sizeof(std::size_t), cudaMemcpyHostToDevice));
        CUCH(cudaMemcpy(medData.data(), dMedData, medData.size() * sizeof(FLOAT), cudaMemcpyHostToDevice));
    }

};

template<int DIM, typename FLOAT = float>
void work(
    int rank,
    int num_ranks,
    FLOAT alpha,
    std::size_t k,
    std::size_t maxIters,
    std::size_t imageLimit,
    std::size_t blockSize,
    std::size_t sigPerBlock,
    std::size_t blocksPerKernel,
    std::size_t cudaStreams,
    DBSignatureListMapped<DIM, FLOAT> &db,
    KMedoidsResults &results
) {
    static_assert(!std::is_same_v<FLOAT, float> || !std::is_same_v<FLOAT, double>, "Invalid FLOAT type");

    MPI_Datatype assign_mpi_type = my_MPI_SIZE_T;
    MPI_Datatype score_mpi_type = std::is_same_v<FLOAT, float> ? MPI_FLOAT : MPI_DOUBLE;

    // Adapted from https://on-demand.gputechconf.com/gtc/2014/presentations/S4236-multi-gpu-programming-mpi.pdf
    int numDevices = 0;
    CUCH(cudaGetDeviceCount(&numDevices)); // numDevices == ranks per node
    int localRank = rank % numDevices;

    std::size_t nodeDataSize = (db.size() / num_ranks) + (db.size() % num_ranks != 0 ? 1 : 0);
    std::size_t nodeDataOffset = rank * nodeDataSize;

    nodeDataSize = std::min(nodeDataSize, db.size() - nodeDataOffset);

    std::vector<FLOAT> scores(db.size());

    ConsCudaAlg<DIM, FLOAT> alg{
        localRank,
        db,
        nodeDataOffset,
        nodeDataSize,
        alpha,
        k,
        blockSize,
        sigPerBlock,
        blocksPerKernel,
        cudaStreams
    };

    alg.initialize();

    for (std::size_t iter = 0; iter < maxIters; ++iter) {
        alg.runAssignment(results.mMedoids, results.mAssignment);

        MPICH(MPI_Allgather(results.mAssignment.data() + nodeDataOffset, nodeDataSize, assign_mpi_type, results.mAssignment.data(), results.mAssignment.size(), assign_mpi_type, MPI_COMM_WORLD));

        alg.runScores(results.mAssignment, scores);

        MPICH(MPI_Allgather(scores.data() + nodeDataOffset, nodeDataSize, score_mpi_type, scores.data(), scores.size(), score_mpi_type, MPI_COMM_WORLD));

        if (!alg.computeMedoids(results.mAssignment, scores, results.mMedoids)) {
            return;
        }
    }
}

#endif