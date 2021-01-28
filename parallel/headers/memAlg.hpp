#ifndef K_MEDOIDS_MEMALG_HPP
#define K_MEDOIDS_MEMALG_HPP

#include "commons.hpp"

#include <memory>
#include <vector>
#include <array>
#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <limits>
#include <atomic>

#include "alg.hpp"
#include "helpers.hpp"
#include "kernels.cuh"

#include "mpi.h"

#include "cuda/cuda.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

template<int DIM, typename FLOAT = float>
class MemCudaAlg : public CudaAlg<DIM, FLOAT>
{
public:
    MemCudaAlg(
        int rank,
        int numRanks,
        std::size_t limit,
        std::size_t seed,
        // Make sure that the db instance lives longer than the instance of this class
        const DBSignatureList<DIM, FLOAT> &inputData,
        FLOAT alpha,
        std::size_t numClusters,
        std::size_t asgnBlockSize,
        std::size_t asgnSigPerBlock,
        std::size_t asgnBlocksPerKernel,
        std::size_t scoreSourcesPerBlock,
        std::size_t scoreTargetsPerBlock,
        std::size_t scoreSourceBlocksPerKernel,
        std::size_t scoreTargetBlocksPerKernel,
        std::size_t cudaStreams
    )
        :   CudaAlg<DIM, FLOAT>(rank, numRanks, limit, seed, inputData, alpha, numClusters, asgnBlockSize, asgnSigPerBlock, asgnBlocksPerKernel, cudaStreams),
            mins(numClusters),
            // +1 for leading zero, to make all access symetric
            clusterComplexities(numClusters + 1),
            eventPool(16),
            rankScoreSlices(numRanks),
            // At most numClusters, maybe less
            rankFragments(numClusters),
            // 1 for each cluster + 2 * numRanks to account for
            // clusters spanning rank boundaries
            allFragments(numClusters + 2*numRanks),
            asgnProps(rank, numRanks, inputData.size(), asgnSigPerBlock, asgnBlocksPerKernel, this->sharedMemSize, inputData.getMaxSignatureLength(), asgnBlockSize),
            scoreProps(scoreSourcesPerBlock, scoreTargetsPerBlock, scoreSourceBlocksPerKernel, scoreTargetBlocksPerKernel)
    {
        CUCH(cudaHostAlloc(&this->hMeds, numClusters * sizeof(std::size_t), cudaHostAllocPortable));
        CUCH(cudaHostAlloc(&this->hMedsOld, numClusters * sizeof(std::size_t), cudaHostAllocPortable));
        CUCH(cudaMalloc(&this->dMeds, numClusters * sizeof(std::size_t)));

        CUCH(cudaHostAlloc(&this->hAssignments, inputData.size() * sizeof(std::size_t), cudaHostAllocPortable));
        CUCH(cudaMalloc(&this->dAssignments, inputData.size() * sizeof(std::size_t)));

        CUCH(cudaHostAlloc(&this->hScores, inputData.size() * sizeof(FLOAT), cudaHostAllocPortable));
        CUCH(cudaMalloc(&this->dScores, inputData.size() * sizeof(FLOAT)));

        CUCH(cudaHostAlloc(&this->hClusterList, inputData.size() * sizeof(std::size_t), cudaHostAllocPortable));
        CUCH(cudaMalloc(&this->dClusterList, inputData.size() * sizeof(std::size_t)));

        // Need +2 as we need +1 for the starting 0 and another +1 so that we can easily use it
        // to fill clusterList, which will shift it one to the left
        // Due to this, we will first start with two leading zeroes, where the second one will be removed
        // by the shift left, making the index correct with one unused value at the end
        CUCH(cudaHostAlloc(&this->hClusterListIndex, (numClusters + 2) * sizeof(std::size_t), cudaHostAllocPortable));
        // On the device, we only need the leading zero, no shifting, so +1
        CUCH(cudaMalloc(&this->dClusterListIndex, (numClusters + 1) * sizeof(std::size_t)));

        // TODO: Try allocating as managed memory
        // then just do a RAM copy, which the driver does anyway and let the CUDA runtime basically do overlapping for us
        CUCH(cudaMalloc(&this->dData, inputData.dataSize() * sizeof(FLOAT)));
        CUCH(cudaMalloc(&this->dIndexes, (inputData.size() + 1) * sizeof(db_offset_t)));

        CUCH(cudaStreamCreate(&this->dataStream));
    }

    ~MemCudaAlg() override {
        CUCH(cudaStreamDestroy(this->dataStream));

        CUCH(cudaFree(this->dIndexes));
        CUCH(cudaFree(this->dData));

        CUCH(cudaFree(this->dClusterListIndex));
        CUCH(cudaFreeHost(this->hClusterListIndex));

        CUCH(cudaFree(this->dClusterList));
        CUCH(cudaFreeHost(this->hClusterList));

        CUCH(cudaFree(this->dScores));
        CUCH(cudaFreeHost(this->hScores));

        CUCH(cudaFree(this->dAssignments));
        CUCH(cudaFreeHost(this->hAssignments));

        CUCH(cudaFree(this->dMeds));
        CUCH(cudaFreeHost(this->hMedsOld));
        CUCH(cudaFreeHost(this->hMeds));

    }

    void initialize() override {
        CUCH(cudaMemcpyAsync(this->dData, this->inputData.data(), this->inputData.dataSize() * sizeof(FLOAT), cudaMemcpyHostToDevice));

        // Set the leading 0
        CUCH(cudaMemsetAsync(this->dIndexes, 0, sizeof(db_offset_t)));
        CUCH(cudaMemcpyAsync(this->dIndexes + 1, this->inputData.indexData(), this->inputData.size() * sizeof(db_offset_t), cudaMemcpyHostToDevice));

        if (this->rank == 0) {
            std::vector<std::size_t> permutation;
            // TODO: Change random permutation generation
            generateRandomPermutation(permutation, this->limit, this->seed);
            std::copy(permutation.begin(), permutation.begin() + this->numClusters, this->hMeds);
        }

        // DEBUG
        std::sort(this->hMeds, this->hMeds + this->numClusters);
        // END DEBUG
        // Broadcast the initial medoids
        MPI_Request bcast;
        MPICH(MPI_Ibcast(this->hMeds, this->numClusters, this->assign_mpi_type, 0, MPI_COMM_WORLD, &bcast));

        RankScoreSlice::initializeMPIType();
        MedFragment::initializeMPIType(this->score_mpi_type);

        MPICH(MPI_Wait(&bcast, MPI_STATUS_IGNORE));
        CUCH(cudaDeviceSynchronize());
    }

    bool runIteration() override {

        // std::cerr << std::endl << "--------Iteration start--------" << std::endl << std::endl;

        runAssignment(this->asgnProps);

        // std::cout << "Assignments: " << std::endl;
        // for (std::size_t i = 0; i < this->inputData.size(); ++i) {
        //     std::cout << this->hAssignments[i] << ", ";
        // }
        // std::cout << std::endl << std::endl;

        preprocessClusters();

        // std::cout << "Cluster list index: " << std::endl;
        // for (std::size_t i = 0; i < this->numClusters + 1; ++i) {
        //     std::cout << this->hClusterListIndex[i] << ", ";
        // }
        // std::cout << std::endl << std::endl;

        // std::cout << "Cluster complexities: " << std::endl;
        // for (std::size_t i = 0; i < this->numClusters + 1; ++i) {
        //     std::cout << this->clusterComplexities[i] << ", ";
        // }
        // std::cout << std::endl << std::endl;

        // std::cout << "Cluster list: " << std::endl;
        // for (std::size_t i = 0; i < this->inputData.size(); ++i) {
        //     std::cout << this->hClusterList[i] << ", ";
        // }
        // std::cout << std::endl << std::endl;

        return runScores(this->scoreProps);
    }

    void fillResults(KMedoidsResults &results) const override {
        results.mMedoids.resize(this->numClusters);
        results.mAssignment.resize(this->inputData.size());
        std::copy(this->hMeds, this->hMeds + this->numClusters, results.mMedoids.begin());
        std::copy(this->hAssignments, this->hAssignments + this->inputData.size(), results.mAssignment.begin());
    }

    void moveResults(KMedoidsResults &results) override {
        fillResults(results);
    }
private:

    class CudaEventPool {
    public:
        CudaEventPool(std::size_t initialSize)
            : events(initialSize), next(0), mark(0)
        {
            for (std::size_t i = 0; i < events.size(); ++i) {
                CUCH(cudaEventCreate(&events[i]));
            }
        }

        ~CudaEventPool() {
            for (auto&& event: events) {
                CUCH(cudaEventDestroy(event));
            }
        }

        cudaEvent_t getEvent() {
            if (next < events.size()) {
                return events[next++];
            }

            events.resize(events.size() * 2);
            for (std::size_t i = next; i < events.size(); ++i) {
                CUCH(cudaEventCreate(&events[i]));
            }
            return events[next++];
        }

        void setMark() {
            mark = next;
        }

        std::vector<cudaEvent_t>::iterator getMarkBegin() {
            return events.begin() + mark;
        }

        std::vector<cudaEvent_t>::iterator getMarkEnd() {
            return events.begin() + next;
        }

        void reset() {
            next = 0;
            mark = 0;
        }
    private:
        std::vector<cudaEvent_t> events;
        std::size_t next;
        std::size_t mark;
    };


    struct AssignmentProperties {
        std::size_t blocksPerKernel;
        std::size_t sigPerBlock;
        std::size_t blockSize;

        std::size_t numKernels;
        std::size_t sigPerKernel;
        std::size_t rankDataSize;
        std::size_t rankDataOffset;


        std::vector<int> recvCounts;
        std::vector<int> displacements;

        AssignmentProperties(
            int rank,
            int numRanks,
            std::size_t inputSize,
            std::size_t sigPerBlock,
            std::size_t blocksPerKernel,
            std::size_t sharedMemSize,
            std::size_t maxSignatureLength,
            std::size_t blockSize
        ) : blocksPerKernel(blocksPerKernel), sigPerBlock(sigPerBlock), blockSize(blockSize)
        {
            std::size_t normRankDataSize = (inputSize / numRanks) + (inputSize % numRanks != 0 ? 1 : 0);
            this->rankDataOffset = rank * normRankDataSize;

            std::size_t lastRankDataSize = inputSize - normRankDataSize * (numRanks - 1);
            this->rankDataSize = rank != numRanks - 1 ? normRankDataSize : lastRankDataSize;

            auto maxSigPerBlock = std::max(
                getMaxSigPerBlockAssignment<DIM, FLOAT>(sharedMemSize, maxSignatureLength, this->blockSize),
                getMaxSigPerBlockScores<DIM, FLOAT>(sharedMemSize, maxSignatureLength, this->blockSize)
            );

            if (this->sigPerBlock > maxSigPerBlock) {
                throw std::invalid_argument("SigPerBlock too large.");
            }

            if (this->sigPerBlock == 0) {
                this->sigPerBlock = maxSigPerBlock;
            }

            std::size_t numBlocks = this->rankDataSize / this->sigPerBlock + (this->rankDataSize % this->sigPerBlock != 0 ? 1 : 0);
            this->numKernels = numBlocks / this->blocksPerKernel + (numBlocks % this->blocksPerKernel != 0 ? 1 : 0);
            this->sigPerKernel = this->sigPerBlock * this->blocksPerKernel;

            this->recvCounts = std::vector<int>(numRanks, this->rankDataSize);
            this->recvCounts.back() = lastRankDataSize;
            this->displacements.resize(numRanks);
            for (std::size_t i = 0; i < this->displacements.size(); ++i) {
                this->displacements[i] = i * this->rankDataSize;
            }
        }

        void print(std::ostream &out) {
            out << "Sig per block: " << this->sigPerBlock << std::endl;
            out << "Blocks per kernel: " << this->blocksPerKernel << std::endl;
            out << "Block size: " << this->blockSize << std::endl;
            out << "Num kernels: " << this->numKernels << std::endl;
            out << "Sig per kernel: " << this->sigPerKernel << std::endl;
            out << "Rank data size: " << this->rankDataSize << std::endl;
            out << "Rank data offset: " << this->rankDataOffset << std::endl;
            out << "Recv counts: [";
            for (auto&& cnt: this->recvCounts) {
                out << cnt << ", ";
            }
            out << "]" << std::endl;
            out << "Displacements: [";
            for (auto&& disp: this->displacements) {
                out << disp << ", ";
            }
            out << "]" << std::endl;
        }
    };

    struct ScoreProperties {
        std::size_t sourcesPerBlock;
        std::size_t targetsPerBlock;
        std::size_t sourceBlocksPerKernel;
        std::size_t targetBlocksPerKernel;

        ScoreProperties(
            std::size_t sourcesPerBlock,
            std::size_t targetsPerBlock,
            std::size_t sourceBlocksPerKernel,
            std::size_t targetBlocksPerKernel
        ) : sourcesPerBlock(sourcesPerBlock),
            targetsPerBlock(targetsPerBlock),
            sourceBlocksPerKernel(sourceBlocksPerKernel),
            targetBlocksPerKernel(targetBlocksPerKernel) {

        }

        void print(std::ostream &out) {
            out << "Sources per block: " << this->sourcesPerBlock << std::endl;
            out << "Targets per block: " << this->targetsPerBlock << std::endl;
            out << "Source blocks per kernel: " << this->sourceBlocksPerKernel << std::endl;
            out << "Target blocks per kernel: " << this->targetBlocksPerKernel << std::endl;
        }
    };

    struct RankScoreSlice {
        static MPI_Datatype MPI_Type;

        static void initializeMPIType() {
            int numItems = 2;
            int blockLengths[] = {1, 1};
            MPI_Aint offsets[2] = {offsetof(RankScoreSlice, startCluster), offsetof(RankScoreSlice, numClusters)};
            MPI_Datatype types[] = {my_MPI_SIZE_T, my_MPI_SIZE_T};

            MPICH(MPI_Type_create_struct(numItems, blockLengths, offsets, types, &RankScoreSlice::MPI_Type));
            MPICH(MPI_Type_commit(&RankScoreSlice::MPI_Type));
        }

        std::size_t startCluster;
        std::size_t numClusters;
    };

    struct MedFragment {
        static MPI_Datatype MPI_Type;

        static void initializeMPIType(MPI_Datatype scoreType) {
            int numItems = 2;
            int blockLengths[] = {1, 1};
            MPI_Aint offsets[2] = {offsetof(MedFragment, idx), offsetof(MedFragment, score)};
            MPI_Datatype types[] = {my_MPI_SIZE_T, scoreType};


            MPICH(MPI_Type_create_struct(numItems, blockLengths, offsets, types, &MedFragment::MPI_Type));
            MPICH(MPI_Type_commit(&MedFragment::MPI_Type));
        }

        std::size_t idx;
        FLOAT score;
    };

    // All data
    FLOAT *dData;
    db_offset_t *dIndexes;

    std::vector<FLOAT> mins;
    std::size_t *hMeds;
    std::size_t *hMedsOld;
    std::size_t *dMeds;

    std::size_t *hAssignments;
    std::size_t *dAssignments;

    FLOAT *hScores;
    FLOAT *dScores;

    std::size_t *hClusterList;
    std::size_t *dClusterList;

    std::size_t *hClusterListIndex;
    std::size_t *dClusterListIndex;

    std::vector<std::size_t> clusterComplexities;

    CudaEventPool eventPool;
    // CUDA stream dedicated for background data transfer
    cudaStream_t dataStream;

    // 1 for each rank
    std::vector<RankScoreSlice> rankScoreSlices;

    std::vector<MedFragment> rankFragments;

    // 1 for each cluster + 2 * numRanks to account for
    // clusters spanning rank boundaries
    std::vector<MedFragment> allFragments;

    AssignmentProperties asgnProps;
    ScoreProperties scoreProps;

    void runAssignment(const AssignmentProperties &props) {
        CUCH(cudaMemcpy(this->dMeds, this->hMeds, this->numClusters * sizeof(std::size_t), cudaMemcpyHostToDevice));
        auto sigToProcess = props.rankDataSize;
        for (std::size_t kernelID = 0; kernelID < props.numKernels; ++kernelID, sigToProcess -= props.sigPerKernel){
            auto kernelSig = std::min(sigToProcess, props.sigPerKernel);
            auto stream = this->streams[kernelID % this->streams.size()];

            auto kernelBlocks = std::min(
                props.blocksPerKernel,
                (kernelSig / props.sigPerBlock) + (kernelSig % props.sigPerBlock != 0 ? 1 : 0)
            );

            auto kerDAssignments = this->dAssignments + props.rankDataOffset + kernelID * props.sigPerKernel;
            db_offset_t *kerDIndexes = this->dIndexes + props.rankDataOffset + kernelID * props.sigPerKernel;
            FLOAT *kerDData = this->dData + this->inputData.signatureDataStartOffset(props.rankDataOffset + kernelID * props.sigPerKernel) * (DIM + 1);
            runComputeAssignments<DIM, FLOAT>(
                kerDIndexes,
                kerDData,
                kernelSig,
                this->dIndexes,
                this->dData,
                this->inputData.size(),
                this->dMeds,
                this->numClusters,
                this->alpha,
                this->inputData.getMaxSignatureLength(),
                kerDAssignments,
                props.blockSize,
                kernelBlocks,
                stream
            );

            CUCH(cudaMemcpyAsync(
                this->hAssignments + props.rankDataOffset + kernelID * props.sigPerKernel,
                kerDAssignments,
                kernelSig * sizeof(std::size_t),
                cudaMemcpyDeviceToHost,
                stream)
            );
        }

        CUCH(cudaDeviceSynchronize());

        MPICH(MPI_Allgatherv(MPI_IN_PLACE, props.rankDataSize, this->assign_mpi_type, this->hAssignments, props.recvCounts.data(), props.displacements.data(), this->assign_mpi_type, MPI_COMM_WORLD));
        // Start assignment upload to GPU
        CUCH(cudaMemcpyAsync(
            this->dAssignments,
            this->hAssignments,
            this->inputData.size() * sizeof(std::size_t),
            cudaMemcpyHostToDevice,
            dataStream)
        );
    }

    void preprocessClusters() {
        // TODO: Make this faster
        for (std::size_t i = 0; i < this->numClusters + 2; ++i) {
            this->hClusterListIndex[i] = 0;
        }

        // TODO: Parallelize this
        for (std::size_t i = 0; i < this->inputData.size(); ++i) {
            // +2 so that we can shift it left by one during clusterList filling
            // and still have leading 0
            this->hClusterListIndex[this->hAssignments[i] + 2]++;
        }

        // std::cout << "Cluster list index: " << std::endl;
        // for (std::size_t i = 0; i < this->numClusters + 2; ++i) {
        //     std::cout << this->hClusterListIndex[i] << ", ";
        // }
        // std::cout << std::endl << std::endl;

        // TODO: Parallelize this
        for (std::size_t i = 1; i < this->numClusters + 1; ++i) {
            // +1 due to the shift above
            // clusterIndex is filled shifted one to the right so that we can then shift it
            // one to the left during clusterList fillup
            this->clusterComplexities[i] = this->hClusterListIndex[i + 1] * this->hClusterListIndex[i + 1];
        }

        // std::cout << "Cluster complexities: " << std::endl;
        // for (std::size_t i = 0; i < this->numClusters + 1; ++i) {
        //     std::cout << this->clusterComplexities[i] << ", ";
        // }
        // std::cout << std::endl << std::endl;


        // TODO: Parallelize this prefix sum
        this->hClusterListIndex[0] = 0;
        this->hClusterListIndex[1] = 0;
        this->clusterComplexities[0] = 0;
        for (std::size_t i = 1; i < this->numClusters + 1; ++i) {
            // Again, clusterListIndex is shifted one to the right
            this->hClusterListIndex[i + 1] = this->hClusterListIndex[i] + this->hClusterListIndex[i + 1];
            this->clusterComplexities[i] = this->clusterComplexities[i - 1] + this->clusterComplexities[i];
        }

        // TODO: Parallelize this
        // This shifts the clusterListIndex one place to the left
        for (std::size_t i = 0; i < this->inputData.size(); ++i) {
            auto asgn = this->hAssignments[i];
            // Move the end of the cluster, i.e start of the next cluster
            // We already had the start of the next cluster in asgn + 2,
            // but that is being used to fill up the next cluster
            auto idx = this->hClusterListIndex[asgn + 1]++;
            this->hClusterList[idx] = i;
        }

        // Start cluster list and cluster list index uploads
        CUCH(cudaMemcpyAsync(
            this->dClusterListIndex,
            this->hClusterListIndex,
            (this->numClusters + 1) * sizeof(std::size_t),
            cudaMemcpyHostToDevice,
            dataStream)
        );
        CUCH(cudaMemcpyAsync(
            this->dClusterList,
            this->hClusterList,
            this->inputData.size() * sizeof(std::size_t),
            cudaMemcpyHostToDevice,
            dataStream)
        );
    }

    bool runScores(const ScoreProperties &props) {
        // Wait for data uploads to finish
        runZeroOut<FLOAT>(this->dScores, this->inputData.size(), 10, this->streams[0]);
        CUCH(cudaDeviceSynchronize());
        //CUCH(cudaStreamSynchronize(dataStream));

        this->eventPool.reset();
        std::size_t streamIdx = 0;
        std::size_t totalComplexity = this->clusterComplexities.back();

        std::size_t complexityPerRank = totalComplexity / this->numRanks + (totalComplexity % this->numRanks != 0 ? 1 : 0);
        std::size_t startComplexity = complexityPerRank * this->rank;
        std::size_t endComplexity = std::min(complexityPerRank * (this->rank + 1), this->clusterComplexities.back());
        auto firstBigger = std::upper_bound(this->clusterComplexities.begin(), this->clusterComplexities.end(), startComplexity);
        std::size_t curCluster = firstBigger - this->clusterComplexities.begin();
        // Make it first smaller than startComplexity, so that we can check
        // for partially processed big clusters and small clusters going accross rank boundaries
        if (curCluster != 0) {
            curCluster -= 1;
        }


        std::size_t rankStartSource;
        std::size_t rankEndSource;
        std::size_t rankStartCluster;
        std::size_t rankEndCluster;
        // If the first cluster is big cluster, the previous rank
        // will have processed part of the cluster up to endComplexity
        if (isBigCluster(curCluster)) {
            // If the startComplexity is exactly the cluster start complexity,
            // start at the cluster beginning
            // This will always happen if cluster 0 is big
            rankStartCluster = curCluster;
            rankStartSource = getBigClusterStartSource(curCluster, clusterComplexities, startComplexity);
            rankEndSource = getBigClusterEndSource(curCluster, clusterComplexities, endComplexity);
            processBigCluster(rankStartSource, rankEndSource, this->hClusterListIndex[curCluster], this->hClusterListIndex[curCluster + 1], props.sourcesPerBlock, props.targetsPerBlock, props.sourceBlocksPerKernel, props.targetBlocksPerKernel, streamIdx);
            curCluster++;
            rankEndCluster = curCluster;
        }
        else if (startComplexity != clusterComplexities[curCluster]) {
            // Small cluster was processed on the previous node if the end complexity lied inside the cluster
            // If it was exactly at the beggining of the cluster, as it will be for the rank 0 and small cluster 0
            // then we need to process it, otherwise start on the one after it
            curCluster++;
            rankStartCluster = curCluster;
            rankEndCluster = rankStartCluster;
            rankStartSource = this->hClusterListIndex[curCluster];
            rankEndSource = rankStartSource;
        }
        else {
            // Small cluster with cluster start exactly at startComplexity
            rankStartCluster = curCluster;
            rankEndCluster = rankStartCluster;
            rankStartSource = this->hClusterListIndex[curCluster];
            rankEndSource = rankStartSource;
        }

        while (this->clusterComplexities[curCluster] < endComplexity) {
            if (isBigCluster(curCluster)) {
                // Will most definitely start processing at the start of the cluster
                // May need to cut it off before the end
                rankEndSource = getBigClusterEndSource(curCluster, clusterComplexities, endComplexity);
                processBigCluster(
                    this->hClusterListIndex[curCluster],
                    rankEndSource,
                    this->hClusterListIndex[curCluster],
                    this->hClusterListIndex[curCluster + 1],
                    props.sourcesPerBlock,
                    props.targetsPerBlock,
                    props.sourceBlocksPerKernel,
                    props.targetBlocksPerKernel,
                    streamIdx);
                curCluster++;
                rankEndCluster = curCluster;
            }
            else {
                curCluster += processSmallClusters(curCluster, clusterComplexities, endComplexity, props.sourcesPerBlock, props.sourceBlocksPerKernel, streamIdx);
                rankEndSource = this->hClusterListIndex[curCluster];
                rankEndCluster = curCluster;
            }
        }

        // We are using Iallgatherv as Iallgather just did not work, don't know why
        std::vector<int> recvCounts(this->numRanks);
        std::vector<int> displacements(this->numRanks);

        for (std::size_t i = 0; i < this->numRanks; ++i) {
            recvCounts[i] = 1;
        }
        for (std::size_t i = 0; i < this->numRanks; ++i) {
            displacements[i] = i;
        }

        this->rankScoreSlices[this->rank] = RankScoreSlice{rankStartCluster, rankEndCluster - rankStartCluster};
        MPI_Request allGather;
        // Exchange numbers of clusters processed by each rank
        // Can be done while processing the cluster processing is running
        // For some reason MPI_Iallgather did not work properly, it just transmited the contents of rank 0
        MPICH(MPI_Iallgatherv(MPI_IN_PLACE, 1, RankScoreSlice::MPI_Type, this->rankScoreSlices.data(), recvCounts.data(), displacements.data(), RankScoreSlice::MPI_Type, MPI_COMM_WORLD, &allGather));

        // Wait for all clusters processed by this rank to be finished
        CUCH(cudaDeviceSynchronize());

        // std::cout << "Rank start cluster: " << rankStartCluster << std::endl;
        // std::cout << "Rank end cluster: " << rankEndCluster << std::endl;
        // std::cout << "Rank start source: " << rankStartSource << std::endl;
        // std::cout << "Rank end source: " << rankEndSource << std::endl;

        // std::vector<FLOAT> orderedScores(this->inputData.size());
        // for (std::size_t i = 0; i < this->inputData.size(); ++i) {
        //     orderedScores[this->hClusterList[i]] = this->hScores[i];
        // }
        // std::cout << "Scores: " << std::endl;
        // for (std::size_t i = 0; i < this->inputData.size(); ++i) {
        //     std::cout << orderedScores[i] << ", ";
        // }
        // std::cout << std::endl << std::endl;

        // std::cout << "Scores: " << std::endl;
        // for (std::size_t i = this->hClusterListIndex[8]; i < this->hClusterListIndex[9]; ++i) {
        //     std::cout << this->hScores[i] << ", ";
        // }
        // std::cout << std::endl << std::endl;

        computeRankFragments(rankStartCluster, rankEndCluster, rankStartSource, rankEndSource);

        MPICH(MPI_Wait(&allGather, MPI_STATUS_IGNORE));

        // for (std::size_t rnk = 0; rnk < this->numRanks; ++rnk) {
        //     std::cout << "Rank: " << rnk << std::endl;
        //     std::cout << "Start cluster: " << this->rankScoreSlices[rnk].startCluster << std::endl;
        //     std::cout << "Num clusters: " << this->rankScoreSlices[rnk].numClusters << std::endl;
        //     std::cout << std::endl;
        // }

        for (std::size_t i = 0; i < this->numRanks; ++i) {
            recvCounts[i] = this->rankScoreSlices[i].numClusters;
        }
        displacements[0] = 0;
        for (std::size_t i = 1; i < this->numRanks; ++i) {
            displacements[i] = displacements[i - 1] + recvCounts[i - 1];
        }

        // std::cout << "Displacements: [";
        // for (auto&& disp: displacements) {
        //     std::cout << disp << ", ";
        // }
        // std::cout << "]" << std::endl;

        MPICH(MPI_Allgatherv(this->rankFragments.data(), this->rankFragments.size(), MedFragment::MPI_Type, allFragments.data(), recvCounts.data(), displacements.data(), MedFragment::MPI_Type, MPI_COMM_WORLD));


        return computeMedoids(displacements);
    }

    /**
     *
     * If we want to run it in parallel with rankScoreSlice communication, we must
     * allocate new fragment vector each time, as the number of
     */
    void computeRankFragments(std::size_t rankStartCluster, std::size_t rankEndCluster, std::size_t rankStartSource, std::size_t rankEndSource) {
        std::size_t rankNumClusters = rankEndCluster - rankStartCluster;
        this->rankFragments.resize(rankNumClusters);
        // TODO: Paralellize
        for (std::size_t cluster = 0; cluster < rankNumClusters; ++cluster) {
            std::size_t clusterStartSource = std::max(rankStartSource, this->hClusterListIndex[rankStartCluster + cluster]);
            std::size_t clusterEndSource = std::min(rankEndSource, this->hClusterListIndex[rankStartCluster + cluster + 1]);

            // std::cout << "Cluster: " << rankStartCluster + cluster << std::endl;
            // std::cout << "Cluster start source: " << clusterStartSource << std::endl;
            // std::cout << "Cluster end source: " << clusterEndSource << std::endl;

            auto clusterStartScore = this->hScores + clusterStartSource;
            // TODO: Add execution policy
            auto min = std::min_element(clusterStartScore, this->hScores + clusterEndSource);
            this->rankFragments[cluster] = MedFragment{this->hClusterList[clusterStartSource + (min - clusterStartScore)], *min};
        }
    }

    bool computeMedoids(const std::vector<int> &displacements) {
        for (std::size_t cluster = 0; cluster < this->numClusters; ++cluster) {
            this->mins[cluster] = std::numeric_limits<FLOAT>::infinity();
        }
        std::swap(this->hMeds, this->hMedsOld);
        for (std::size_t rnk = 0; rnk < this->numRanks; ++rnk) {
            for (std::size_t i = 0; i < this->rankScoreSlices[rnk].numClusters; ++i) {
                std::size_t cluster = this->rankScoreSlices[rnk].startCluster + i;
                auto medFragment = this->allFragments[displacements[rnk] + i];
                if (medFragment.score < this->mins[cluster]) {
                    this->mins[cluster] = medFragment.score;
                    this->hMeds[cluster] = medFragment.idx;
                }
            }
        }

        for (std::size_t i = 0; i < this->numClusters; ++i) {
            if (this->hMeds[i] != this->hMedsOld[i]) {
                return true;
            }
        }
        return false;
    }

    std::size_t getClusterComplexity(std::size_t cluster) {
        return this->clusterComplexities[cluster + 1] - this->clusterComplexities[cluster];
    }

    bool isBigCluster(std::size_t cluster) {
        // TODO: Make this a parameter
        constexpr int clusterComplexityCutoff = 1000;

        return getClusterComplexity(cluster) >= clusterComplexityCutoff;
    }

    std::size_t getBigClusterStartSource(std::size_t curCluster, const std::vector<std::size_t> &clusterComplexities, std::size_t startComplexity) {
        std::size_t clusterStart = this->hClusterListIndex[curCluster];
        std::size_t clusterEnd = this->hClusterListIndex[curCluster + 1];
        std::size_t clusterSize = clusterEnd - clusterStart;

        // [0,1] number telling us in which part of the big cluster to start
        // Because for every source we must process the whole size of the cluster
        // the split of complexities between ranks is just division of source image range
        // This is to prevent multiple ranks computing partial scores for the same image
        // This will also allow us to just share results for the clusters we computed scores for
        // so another Allgatherv
        double partToStartAt = static_cast<double>(startComplexity - clusterComplexities[curCluster]) / getClusterComplexity(curCluster);
        return startComplexity == clusterComplexities[curCluster] ?
            clusterStart :
            clusterStart + partToStartAt * clusterSize;
    }

    std::size_t getBigClusterEndSource(std::size_t curCluster, const std::vector<std::size_t> &clusterComplexities, std::size_t endComplexity) {
        std::size_t clusterStart = this->hClusterListIndex[curCluster];
        std::size_t clusterEnd = this->hClusterListIndex[curCluster + 1];
        std::size_t clusterSize = clusterEnd - clusterStart;

        double partToEndAt = static_cast<double>(endComplexity - clusterComplexities[curCluster]) / getClusterComplexity(curCluster);

        // WARNING: The rounding of partToEndAt * clusterSize MUST match with the rounding in getBigCLusterStartIndex
        //  otherwise we risk missing an item between endSource of rank i and startSource of rank i + 1
        return endComplexity >= clusterComplexities[curCluster + 1] ?
            clusterEnd :
            clusterStart + partToEndAt * clusterSize;
    }

    // TODO: Share big cluster processing between neighbouring nodes, to split up really large clusters
    // Leave each rank to process only so many source images to fill complexity per rank
    // So really big clusters can be shared between many many ranks
    // Then each rank will do a simple min of all clusters it worked on and send the minimum with it's score to all other clusters
    void processBigCluster(std::size_t sourcesStart, std::size_t sourcesEnd, std::size_t clusterStart, std::size_t clusterEnd, std::size_t sourcesPerBlock, std::size_t targetsPerBlock, std::size_t sourceBlocksPerKernel, std::size_t targetBlocksPerKernel, std::size_t &streamIdx) {
        std::size_t kernelSourcesStart = sourcesStart;
        std::size_t kernelSourcesEnd = std::min(kernelSourcesStart + sourcesPerBlock * sourceBlocksPerKernel, sourcesEnd);


        // std::cerr << "BIG CLUSTER" << std::endl;
        // std::cerr << "Sources start: " << sourcesStart << std::endl;
        // std::cerr << "Sources end: " << sourcesEnd << std::endl;
        // std::cerr << "Cluster start: " << clusterStart << std::endl;
        // std::cerr << "Cluster end: " << clusterEnd << std::endl;
        // std::cerr << std::endl;


        while (kernelSourcesStart < sourcesEnd) {
            std::size_t kernelSourcesNum = kernelSourcesEnd - kernelSourcesStart;
            std::size_t sourceBlocks = kernelSourcesNum / sourcesPerBlock + (kernelSourcesNum % sourcesPerBlock != 0 ? 1 : 0);

            // std::cerr << "Kernel sources start: " << kernelSourcesStart << std::endl;
            // std::cerr << "Kernel sources end: " << kernelSourcesEnd << std::endl;
            // std::cerr << "Source blocks: " << sourceBlocks << std::endl;
            // std::cerr << "Stream Idx: " << streamIdx << std::endl;
            // std::cerr << std::endl;

            this->eventPool.setMark();

            std::size_t targetsStart = clusterStart;
            std::size_t targetsEnd = std::min(clusterStart + targetsPerBlock * targetBlocksPerKernel, clusterEnd);
            std::size_t targetsNum = targetsEnd - targetsStart;
            while (targetsStart < clusterEnd) {
                streamIdx = (streamIdx + 1) % this->streams.size();
                std::size_t targetBlocks = targetsNum / targetsPerBlock + (targetsNum % targetsPerBlock != 0 ? 1 : 0);
                runGetScoresPreprocessedLarge<DIM, FLOAT>(
                    this->dIndexes,
                    this->dData,
                    this->dAssignments,
                    this->dClusterList,
                    kernelSourcesStart,
                    kernelSourcesNum,
                    targetsStart,
                    targetsNum,
                    this->alpha,
                    this->inputData.getMaxSignatureLength(),
                    this->dScores,
                    this->blockSize,
                    sourceBlocks,
                    targetBlocks,
                    this->streams[streamIdx]
                );
                CUCH(cudaEventRecord(this->eventPool.getEvent(), this->streams[streamIdx]));

                targetsStart = targetsEnd;
                targetsEnd = std::min(targetsStart + targetsPerBlock * targetBlocksPerKernel, clusterEnd);
            };

            // Wait for all kernels computing the current source range
            for (auto it = this->eventPool.getMarkBegin(); it != this->eventPool.getMarkEnd(); ++it) {
                CUCH(cudaStreamWaitEvent(this->streams[streamIdx], *it));
            }
            CUCH(cudaMemcpyAsync(this->hScores + kernelSourcesStart, this->dScores + kernelSourcesStart, kernelSourcesNum * sizeof(FLOAT), cudaMemcpyDeviceToHost, this->streams[streamIdx]));
            kernelSourcesStart = kernelSourcesEnd;
            kernelSourcesEnd = std::min(kernelSourcesStart + sourcesPerBlock * sourceBlocksPerKernel, sourcesEnd);
        };
    }

    std::size_t processSmallClusters(std::size_t firstCluster, const std::vector<std::size_t> &clusterComplexities, std::size_t endComplexity, std::size_t sourcesPerBlock, std::size_t blocksPerKernel, std::size_t &streamIdx) {
        // The first big cluster after the small cluster range
        std::size_t endCluster = firstCluster + 1;

        // std::cerr << "SMALL CLUSTER" << std::endl;
        // std::cerr << "First cluster: " << firstCluster << std::endl;
        // std::cerr << "First cluster complexity: " << getClusterComplexity(firstCluster) << std::endl;
        // std::cerr << "End complexity: " << endComplexity << std::endl;
        // std::cerr << std::endl;

        // Add clusters to range while they are big
        while (!isBigCluster(endCluster) && clusterComplexities[endCluster] < endComplexity) {

            // std::cerr << "Additional cluster complexity: " << getClusterComplexity(endCluster) << std::endl;
            endCluster += 1;
        }

        std::size_t startIndex = this->hClusterListIndex[firstCluster];
        std::size_t leftToProcess = this->hClusterListIndex[endCluster] - startIndex;
        while (leftToProcess != 0) {
            streamIdx = (streamIdx + 1) % this->streams.size();
            std::size_t kernelSources = std::min(sourcesPerBlock * blocksPerKernel, leftToProcess);
            std::size_t numBlocks = kernelSources / sourcesPerBlock + (kernelSources % sourcesPerBlock != 0 ? 1 : 0);

            // std::cerr << "Start index: " << startIndex << std::endl;
            // std::cerr << "Kernel sources: " << kernelSources << std::endl;
            // std::cerr << "Num blocks: " << numBlocks << std::endl;
            // std::cerr << "Stream Idx: " << streamIdx << std::endl;

            runGetScoresPreprocessedSmall<DIM, FLOAT>(
                this->dIndexes,
                this->dData,
                this->dAssignments,
                this->dClusterList,
                this->dClusterListIndex,
                startIndex,
                kernelSources,
                this->alpha,
                this->inputData.getMaxSignatureLength(),
                this->dScores,
                this->blockSize,
                numBlocks,
                this->streams[streamIdx]
            );
            CUCH(cudaMemcpyAsync(this->hScores + startIndex, this->dScores + startIndex, kernelSources * sizeof(FLOAT), cudaMemcpyDeviceToHost, this->streams[streamIdx]));

            startIndex += kernelSources;
            leftToProcess -= kernelSources;
        }

        return endCluster - firstCluster;
    }
};

template<int DIM, typename FLOAT> MPI_Datatype MemCudaAlg<DIM,FLOAT>::RankScoreSlice::MPI_Type;
template<int DIM, typename FLOAT> MPI_Datatype MemCudaAlg<DIM,FLOAT>::MedFragment::MPI_Type;

#endif