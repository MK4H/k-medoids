#ifndef K_MEDOIDS_CONSALG_HPP
#define K_MEDOIDS_CONSALG_HPP


#include "commons.hpp"

#include <memory>
#include <vector>
#include <array>
#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <limits>

#include "alg.hpp"
#include "helpers.hpp"
#include "kernels.cuh"

#include "cuda/cuda.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

template<int DIM, typename FLOAT = float>
class ConsCudaAlg : public CudaAlg<DIM, FLOAT>
{
public:
    ConsCudaAlg(
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
        : CudaAlg<DIM, FLOAT>(rank, numRanks, limit, seed, inputData, alpha, numClusters, blockSize, sigPerBlock, blocksPerKernel, cudaStreams)
    {

        std::size_t normNodeDataSize = (inputData.size() / numRanks) + (inputData.size() % numRanks != 0 ? 1 : 0);
        this->nodeDataOffset = rank * normNodeDataSize;

        // Last node may have a little less data
        std::size_t lastNodeDataSize = inputData.size() - normNodeDataSize * (numRanks - 1);
        this->nodeDataSize = rank != numRanks - 1 ? normNodeDataSize : lastNodeDataSize;

        // NOTE: We could have separate sigPerBlock for Assignment kernels and Scores kernels, but this might be enough
        auto maxSigPerBlock = std::max(
            getMaxSigPerBlockAssignment<DIM, FLOAT>(this->sharedMemSize, this->inputData.getMaxSignatureLength(), this->blockSize),
            getMaxSigPerBlockScores<DIM, FLOAT>(this->sharedMemSize, this->inputData.getMaxSignatureLength(), this->blockSize)
        );

        if (this->sigPerBlock > maxSigPerBlock) {
            throw std::invalid_argument("SigPerBlock too large.");
        }

        if (this->sigPerBlock == 0) {
            this->sigPerBlock = maxSigPerBlock;
        }


        this->numBlocks = this->nodeDataSize / this->sigPerBlock + (this->nodeDataSize % this->sigPerBlock != 0 ? 1 : 0);
        this->numKernels = this->numBlocks / this->blocksPerKernel + (this->numBlocks % this->blocksPerKernel != 0 ? 1 : 0);
        this->maxSigPerKernel = this->sigPerBlock * this->blocksPerKernel;

        this->recvCounts = std::vector<int>(this->numRanks, this->nodeDataSize);
        this->recvCounts.back() = lastNodeDataSize;
        this->displacements.resize(this->numRanks);
        for (std::size_t i = 0; i < this->displacements.size(); ++i) {
            this->displacements[i] = i * this->nodeDataSize;
        }


        CUCH(cudaMalloc(&this->dData, this->inputData.dataSize() * sizeof(FLOAT)));
        // +1 so that we can store the leading 0 to make all access consistent
        CUCH(cudaMalloc(&this->dIndexes, (this->inputData.size() + 1) * sizeof(db_offset_t)));
        CUCH(cudaMalloc(&this->dMedData, numClusters * this->inputData.getMaxSignatureLength() * (DIM + 1) * sizeof(FLOAT)));
        // +1 so that we can store the leading 0 to make all access consistent
        CUCH(cudaMalloc(&this->dMedIndexes, (numClusters + 1) * sizeof(db_offset_t)));

        /* TODO: We can split this to node assignments and target assignments
            We only need node assignment and target assignment large enough that
            we can run kernel in each CUDA stream
        */
        CUCH(cudaMalloc(&this->dAssignments, this->inputData.size() * sizeof(std::size_t)));

        // TODO: We also only need scores for node data
        CUCH(cudaMalloc(&this->dNScores, this->nodeDataSize * sizeof(FLOAT)));
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
        if (this->rank == 0) {
            generateRandomPermutation(this->medoids, this->limit, this->seed);
        }

        this->medoids.resize(this->numClusters);

        // DEBUG
        std::sort(this->medoids.begin(), this->medoids.end());
        // END DEBUG
        // Broadcast the initial medoids
        MPICH(MPI_Bcast(this->medoids.data(), this->numClusters, this->assign_mpi_type, 0, MPI_COMM_WORLD));

        // NOTE: Try to allocate a larger arrays nodeDataSize * numRanks
        // so that we can use simple MPI_Allgather instead of MPI_Allgatherv
        this->scores.resize(this->inputData.size());
        this->assignments.resize(this->inputData.size());

        CUCH(cudaMemcpy(this->dData, this->inputData.data(), this->inputData.dataSize() * sizeof(FLOAT), cudaMemcpyHostToDevice));

        // Set the leading 0
        CUCH(cudaMemset(this->dIndexes, 0, sizeof(db_offset_t)));
        CUCH(cudaMemcpy(this->dIndexes + 1, this->inputData.indexData(), this->inputData.size() * sizeof(db_offset_t), cudaMemcpyHostToDevice));
    }

    bool runIteration() override {
        runAssignment(this->medoids, this->assignments);

        MPICH(MPI_Allgatherv(this->assignments.data() + this->nodeDataOffset, this->nodeDataSize, this->assign_mpi_type, this->assignments.data(), this->recvCounts.data(), this->displacements.data(), this->assign_mpi_type, MPI_COMM_WORLD));

        runScores(this->assignments, this->scores);

        MPICH(MPI_Allgatherv(this->scores.data() + this->nodeDataOffset, this->nodeDataSize, this->score_mpi_type, scores.data(), recvCounts.data(), displacements.data(), this->score_mpi_type, MPI_COMM_WORLD));

        return computeMedoids(this->assignments, this->scores, this->medoids);
    }

    void runAssignment(const std::vector<std::size_t> &medoids, std::vector<std::size_t> &assignments) {
        consolidateMedoids(medoids);

        auto sigToProcess = this->nodeDataSize;
        for (std::size_t kernelID = 0; kernelID < this->numKernels; ++kernelID, sigToProcess -= this->maxSigPerKernel) {
            auto kernelSig = std::min(sigToProcess, this->maxSigPerKernel);
            auto stream = this->streams[kernelID % this->streams.size()];

            auto kernelBlocks = std::min(
                this->blocksPerKernel,
                (kernelSig / this->sigPerBlock) + (kernelSig % this->sigPerBlock != 0 ? 1 : 0)
            );
            // TODO: This is most likely wrong, does not work with nodeDataOffset
            // It works because the full size of dAssignments is only used for getScores kernel
            // so this just uses the fist part for all nodes
            auto kerDAssignments = this->dAssignments + kernelID * this->maxSigPerKernel;

            db_offset_t *kerDIndexes = this->dIndexes + this->nodeDataOffset + kernelID * this->maxSigPerKernel;
            FLOAT * kerDData = this->dData + this->inputData.signatureDataStartOffset(this->nodeDataOffset + kernelID * this->maxSigPerKernel) * (DIM + 1);
            runComputeAssignmentsConsolidated<DIM, FLOAT>(
                kerDIndexes,
                kerDData,
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
            CUCH(cudaMemcpyAsync(assignments.data() + this->nodeDataOffset + kernelID * this->maxSigPerKernel, kerDAssignments, kernelSig * sizeof(std::size_t), cudaMemcpyDeviceToHost, stream));
        }
        CUCH(cudaDeviceSynchronize());
    }

    /** Runs the GetScores kernel to calculate scores of the data block assigned to this node
     *
     * Expects all data to be present on the GPU, does no data transfer apart from the assignments in and results out
     */
    void runScores(const std::vector<std::size_t> &assignments, std::vector<FLOAT> &scores) {
        auto sigToProcess = this->nodeDataSize;
        // TODO: Split and overlap the copy
        runZeroOut<FLOAT>(this->dNScores, this->nodeDataSize, 10);
        CUCH(cudaMemcpy(this->dAssignments, assignments.data(), assignments.size() * sizeof(std::size_t), cudaMemcpyHostToDevice));
        for (std::size_t kernelID = 0; kernelID < this->numKernels; ++kernelID, sigToProcess -= this->maxSigPerKernel) {
            auto kernelSig = std::min(sigToProcess, this->maxSigPerKernel);
            auto stream = this->streams[kernelID % this->streams.size()];

            db_offset_t *kerDIndexes = this->dIndexes + this->nodeDataOffset + kernelID * this->maxSigPerKernel;
            FLOAT * kerDData = this->dData + this->inputData.signatureDataStartOffset(this->nodeDataOffset + kernelID * this->maxSigPerKernel) * (DIM + 1);
            auto kerDAssignments = this->dAssignments + this->nodeDataOffset + kernelID * this->maxSigPerKernel;
            auto kerDScores = this->dNScores + kernelID * this->maxSigPerKernel;
            runGetScores<DIM, FLOAT>(
                kerDIndexes,
                kerDData,
                kerDAssignments,
                kernelSig,
                this->dIndexes,
                this->dData,
                this->dAssignments,
                this->inputData.size(),
                this->alpha,
                this->inputData.getMaxSignatureLength(),
                kerDScores,
                this->blockSize,
                this->numBlocks,
                stream
            );
            CUCH(cudaMemcpyAsync(scores.data() + this->nodeDataOffset + kernelID * this->maxSigPerKernel, kerDScores, kernelSig * sizeof(FLOAT), cudaMemcpyDeviceToHost, stream));
        }
        CUCH(cudaDeviceSynchronize());
    }

    // TODO: Try restricted modifier
    bool computeMedoids(const std::vector<std::size_t> &assignments, const std::vector<FLOAT> &scores, std::vector<std::size_t> &medoids) {
        // TODO: Preallocate in constructor
        std::vector<FLOAT> minScores(medoids.size(), std::numeric_limits<FLOAT>::infinity());
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

        medoids = std::move(newMedoids);
        return changed;
    }

    void fillResults(KMedoidsResults &results) const override {
        results.mAssignment = this->assignments;
        results.mMedoids = this->medoids;
    }

    void moveResults(KMedoidsResults &results) override {
        results.mAssignment = std::move(this->assignments);
        results.mMedoids = std::move(this->medoids);
    }
private:

    // In number of signatures
    std::size_t nodeDataOffset;
    std::size_t nodeDataSize;

    std::vector<int> recvCounts;
    std::vector<int> displacements;

    std::vector<FLOAT> scores;
    std::vector<std::size_t> assignments;
    std::vector<std::size_t> medoids;

    // All data
    FLOAT *dData;
    db_offset_t *dIndexes;

    // For consolidated medoids
    FLOAT *dMedData;
    db_offset_t *dMedIndexes;

    std::size_t *dAssignments;
    FLOAT *dNScores;

    std::size_t numBlocks;
    std::size_t numKernels;
    std::size_t maxSigPerKernel;

    void consolidateMedoids(const std::vector<std::size_t> &medoids) {
        // TODO: Preallocate these arrays
        std::vector<db_offset_t> medIndexes;
        std::vector<FLOAT> medData;

        medIndexes.reserve(this->numClusters + 1);
        medData.reserve(this->numClusters * this->inputData.getMaxSignatureLength() * (DIM + 1));

        const FLOAT *data = this->inputData.data();

        medIndexes.push_back(0);
        for (auto&& medIdx: medoids) {
            auto medStart = this->inputData.signatureDataStartOffset(medIdx);
            auto medEnd = this->inputData.signatureDataEndOffset(medIdx);
            auto medSize = medEnd - medStart;
            medIndexes.push_back(medIndexes.back() + medSize);
            medData.insert(medData.end(), data + medStart * (DIM + 1), data + medEnd * (DIM + 1));
        }

        CUCH(cudaMemcpy(this->dMedIndexes, medIndexes.data(), medIndexes.size() * sizeof(db_offset_t), cudaMemcpyHostToDevice));
        CUCH(cudaMemcpy(this->dMedData, medData.data(), medData.size() * sizeof(FLOAT), cudaMemcpyHostToDevice));
    }

};

#endif