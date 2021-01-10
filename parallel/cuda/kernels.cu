#include "kernels.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdint>

template<int DIM, typename FLOAT = float>
struct Signature {
    const std::size_t numCentroids;
    const FLOAT *weights;
    const FLOAT *centroids;


    __device__ Signature(const db_offset_t startIdx, const db_offset_t endIdx, const FLOAT *data)
        :   numCentroids(endIdx - startIdx),
            weights(data + startIdx * (DIM + 1)),
            centroids(this->weights + this->numCentroids)
    {

    }
};

template<int DIM, typename FLOAT = float>
__device__ FLOAT dist2(const FLOAT *a, const FLOAT *b) {
    FLOAT res = 0;
    for (int i = 0; i < DIM; ++i) {
        FLOAT diff = b[i] - a[i];
        res += diff * diff;
    }
    return res;
}

template<int DIM, typename FLOAT = float>
__device__ FLOAT similarity(const FLOAT *centr1, const FLOAT *centr2, FLOAT alpha) {
    return expf(-alpha * dist2<DIM, FLOAT>(centr1, centr2));
}

/**
 * Transforms the data index from the whole dataset into an index into the part of the dataset given
 * to the current kernel
 *
 * See the description of globalToSharedIndexes for more info about indexes
 *
 */
__device__ db_offset_t getLocalIndex(const db_offset_t globalValue, const db_offset_t baseIndex) {
    return globalValue - baseIndex;
}

/**
 * Transforms index, which tells us offset in number of centroids, into offset in
 * number of values, whatever the value type or size is.
 *
 * Data contains DIM values + weight for each centroid, so DIM + 1 values per centroid
 *
 */
__device__ db_offset_t idxToOffset(const db_offset_t idx, const int DIM) {
    return idx * (DIM + 1);
}

template<typename FLOAT>
__device__ FLOAT mergeSums(FLOAT *partSums) {
    FLOAT total = 0;

    __syncthreads();

    if (threadIdx.x == 0) {
        // TODO: Use warp reduce functions
        for (int i = 0; i < blockDim.x; ++i) {
            total += partSums[i];
        }
    }

    __syncthreads();

    return total;
}

/** TODO: Use that the resulting matrix is symmetric
* Computes the self similarity of a signature
*
* The computation is divided as follows:
* t1.1 t2.1 t3.1 t4.1 t1.2
* t2.2 t3.2 t4.2 t1.3 t2.3
* t3.3 etc.
*
* The value at row i, column j is computed as
* w_i * sim(c_i, c_j) * w_j
*
* The result is all the matrix values summed up
* So we sum all results of given thread and then
* we sum the results of all the threads.
*/
template<int DIM, typename FLOAT = float>
__device__ FLOAT getSelfSim(Signature<DIM, FLOAT> sig, FLOAT alpha, FLOAT *partSums) {
    FLOAT sum = 0;

    // TODO: Can be rewritten into a for cycle
    int row = threadIdx.x / sig.numCentroids;
    int col = threadIdx.x % sig.numCentroids;

    while (row < sig.numCentroids) {
        sum += sig.weights[row] * similarity<DIM, FLOAT>(sig.centroids + row * DIM, sig.centroids + col * DIM, alpha) * sig.weights[col];
        col += blockDim.x;
        row += col / sig.numCentroids;
        col = col % sig.numCentroids;
    }

    partSums[threadIdx.x] = sum;
    return mergeSums(partSums);
}

/** TODO: Check that the result matrix really is symmetric
* \brief Computes the top left and bottom right corner parts of the similarity metrix
*       multiplied by weights
*
* \note Results are added to the partSums array so that they are kept thread local, no synchronization needed
*
*/
template<int DIM, typename FLOAT = float>
__device__ FLOAT getOtherSim(Signature<DIM, FLOAT> sig1, Signature<DIM, FLOAT> sig2, FLOAT alpha, FLOAT *partSums) {
    FLOAT sum = 0;

    // TODO: While can be rewritten into a for cycle
    int row = threadIdx.x / sig2.numCentroids;
    int col = threadIdx.x % sig2.numCentroids;

    while (row < sig1.numCentroids) {
        // 2 * to use the fact the matrix is symmetric
        // so that we don't have to compute it twice
        sum += 2 * sig1.weights[row] * similarity<DIM, FLOAT>(sig1.centroids + row * DIM, sig2.centroids + col * DIM, alpha) * -sig2.weights[col];
        col += blockDim.x;
        row += col / sig2.numCentroids;
        col = col % sig2.numCentroids;
    }

    partSums[threadIdx.x] = sum;
    return mergeSums(partSums);
}

template<int DIM, typename FLOAT = float>
__device__ void copySigData(const db_offset_t *inIdx, const FLOAT *inAllData, int numSig, db_offset_t baseIndex, FLOAT *outData) {
    auto startOffset = idxToOffset(getLocalIndex(inIdx[0], baseIndex), DIM);
    auto endOffset = idxToOffset(getLocalIndex(inIdx[numSig], baseIndex), DIM);

    // Just copy continuous data to shared memory
    for (auto i = threadIdx.x; i < endOffset - startOffset; i += blockDim.x) {
        outData[i] = inAllData[startOffset + i];
    }
}

/**
 * Transforms global indexes into shared indexes for data procesed by the current block
 *
 * There are three index "types" used in the kernel code.
 * - global index, which is the value passed to the kernel __global__ function, this index represents
 *   offset of the signature in the whole dataset in unit of number of centroids
 * - local index, or kernel local index, which represents offset in number of centroids in the subset of data processed by
 *   the current kernel
 * - shared mem index, or block local index, which represents offset in number of centroids in the subset of data loaded into
 *   shared memory, which is a subset of data processed by the whole kernel
 */
__device__ void globalToSharedIndexes(const db_offset_t *inIdx, const int numSig, const db_offset_t baseIndex, db_offset_t *outIdx) {
    // Recalculate indexes based on the shared memory contents and
    // copy them to shared memory
    auto startIndex = getLocalIndex(inIdx[0], baseIndex);
    // + 1 so that we also have the start index of the following signature
    // to allow us to calculate the size of the last signature
    for (auto i = threadIdx.x; i < numSig + 1; i += blockDim.x) {
        outIdx[i] = getLocalIndex(inIdx[i], baseIndex) - startIndex;
    }
}

/**
 * Partitions shared memory to separate parts to store data like current medoid, block data, block data indexes, partial sums etc.
 *
 * The amount of shared memory needed is the sum of:
 *  sizeof(FLOAT) * maxSignatureSize                                // For current medoid
 *  sizeof(FLOAT) * numDataSignaturesPerBlock                       // For current minimal distance to medoid of each data signature
 *  sizeof(FLOAT) * blockDim.x                                      // For partial sums during SQDF computation
 *  sizeof(FLOAT) * numDataSignaturesPerBlock                       // For precomputed selfSimilarities of each data signature
 *  sizeof(db_offset_t) * (numDataSignaturesPerBlock + 1)           // For index table of data signatures
 *  sizeof(FLOAT) * maxSignatureSize * numDataSIgnaturesPerBlock    // For the data signatures themselfs
 */
template<typename FLOAT = float>
__device__ void partitionSharedMemAssign(void *shared, std::size_t maxSignatureSize, int sigPerBlock, FLOAT *&med, FLOAT *&mins, FLOAT *&shSums, FLOAT *&dSelfSims, db_offset_t *&shIndexes, FLOAT *&shData) {
    // TODO: Check for bank conflicts
    // Split up the allocated shared memory
    med = (FLOAT *)shared;
    mins = med + maxSignatureSize;
    // TODO: Use warp reduce and reduce the size of shSum to one element per warp
    // We need one sum entry for each thread
    shSums = mins + sigPerBlock;
    dSelfSims = shSums + blockDim.x;
    shIndexes = (db_offset_t *)(dSelfSims + sigPerBlock);

    // + 1 so that we allocate one more index space for the start index of
    // the following signature to allow us to calculate the size of the last signature
    shData = (FLOAT *)(shIndexes + sigPerBlock + 1);
}



/**
 * \brief
 * \note The expected usage of this kernel is that it will be called multiple times, each time with a small chunk of data that fits
 *          into shared memory next to one medoid signature and few helper datastructures. It will go through all medoids sequentially,
 *          reading each into shared memory and then calculating the distance of each data signature to the medoid signature.
 *          Each block computes the SQFD for the pair of medoid signature/data signature, but only the thread 0 works with global
 *          things like min and current assignment. This way we get around synchronization.
 *
            For amount of shared memory required, see partitionSharedMemAssign function
 * \param data All signatures in data must fit into shared memory, togehter with
 * \warning The index arrays are extended compared to the CPU code and serial implementation and additionally contain the first 0 entry,
 *      so that we have consistently both start and end index for every signature, which simplifies kernel code
 */
template<int DIM, typename FLOAT = float>
__global__ void computeAssignmentsConsolidated(
    const db_offset_t *indexes,
    const FLOAT *data,
    const std::size_t numSig,
    const db_offset_t *medIndexes,
    const FLOAT *medData,
    const std::size_t numMed,
    const FLOAT alpha,
    const std::size_t maxSignatureSize,
    std::size_t * assignments
) {
    extern __shared__  FLOAT shared[];

    // Number of blocks will always divide the number of Signatures without remainder
    // APART from the last kernel call, which will just have to compute the rest of the data
    const int sigPerBlock = (numSig / gridDim.x) + numSig % gridDim.x != 0 ? 1 : 0;

    // SHARED MEMORY PARTITIONING

    FLOAT *med, *mins, *shSums, *dSelfSims, *shData;
    db_offset_t *shIndexes;
    partitionSharedMemAssign(shared, maxSignatureSize, sigPerBlock, med, mins, shSums, dSelfSims, shIndexes, shData);

    // TODO: Initialize mins

    // TODO: Off by one errors
    int blockStartSig = blockIdx.x * sigPerBlock;
    // As the processed signatures have to fit into shared memory, int is enough
    int blockEndSig = min((blockIdx.x + 1) * sigPerBlock, (int)numSig);
    int blockNumSig = blockEndSig - blockStartSig;

    const db_offset_t dataBaseIndex = indexes[0];
    copySigData<DIM, FLOAT>(indexes + blockStartSig, data, blockNumSig, dataBaseIndex, shData);
    globalToSharedIndexes(indexes + blockStartSig, blockNumSig, dataBaseIndex, shIndexes);

    __syncthreads();

    for (int dataSig = 0; dataSig < blockNumSig; ++dataSig) {
        FLOAT selfSim = getSelfSim(Signature<DIM, FLOAT>(shIndexes[dataSig], shIndexes[dataSig + 1], shData), alpha, shSums);
        if (threadIdx.x == 0) {
            dSelfSims[dataSig] = selfSim;
        }
    }

    for (int medIdx = 0; medIdx < numMed; ++medIdx) {
        // Load medoid to shared mem
        // All medoids are passed to each kernel, so the base index is 0
        copySigData<DIM, FLOAT>(medIndexes + medIdx, medData, 1, 0, med);
        auto medNumCentroids = medIndexes[medIdx + 1] - medIndexes[medIdx];
        __syncthreads();

        FLOAT medSelfSim = getSelfSim(Signature<DIM, FLOAT>(0, medNumCentroids, med), alpha, shSums);

        for (int i = 0; i < blockNumSig; ++i) {

            auto otherSim = getOtherSim(Signature<DIM, FLOAT>(shIndexes[i], shIndexes[i + 1], shData), Signature<DIM, FLOAT>(0, medNumCentroids, med), alpha, shSums);
            if (threadIdx.x == 0) {
                FLOAT sqfd2 = dSelfSims[i] + medSelfSim + otherSim;
                if (sqfd2 < mins[i]) {
                    mins[i] = sqfd2;
                    // TODO: Map this local med index to DB Signature index
                    // OR pass the mapping to CUDA kernel so that we can do it here
                    assignments[blockStartSig + i] = medIdx;
                }
            }
        }
    }
}

// TODO: Assignments without consolidating medoids, just use that the whole data fits into memory

template<int DIM, typename FLOAT = float>
__global__ void computeAssignments(const db_offset_t *indexes, const FLOAT *data, const std::size_t totNumSig,const db_offset_t *kernIndexes, const FLOAT *kernData, const std::size_t kernNumSig, const std::size_t *medSeqIdxs, const std::size_t numMed, const FLOAT alpha, std::size_t * assignments) {

}

__device__ void partitionSharedMemScore() {

}

/**
 * \brief
 * \note This function computes the scores of <source> group of signatures (described by sIndexes and sData)
 *      as medoids when comparing with signatures in the <target> group (described by tIndexes and tData)
 *      The functions is designed so that it can be used to process the whole dataset with overlapped data transfer,
 *      as it needs only the <source> and <target> groups in global memory. Of course if the whole dataset fits
 *      into global memory, it can be used to process that too.
 */
template<int DIM, typename FLOAT = float>
__global__ void getScores(
    const db_offset_t *sIndexes,
    const FLOAT *sData,
    const std::size_t * sAssignments,
    const std::size_t sNumSig,
    const db_offset_t *tIndexes,
    const FLOAT *tData,
    const std::size_t * tAssignments,
    const std::size_t tNumSig,
    const FLOAT alpha,
    const std::size_t maxSignatureSize,
    const std::size_t kernelID,
    FLOAT *sScores) {
    extern __shared__  FLOAT shared[];

    // Number of blocks will always divide the number of Signatures without remainder
    // APART from the last kernel call, which will just have to compute the rest of the data
    const int sSigPerBlock = (sNumSig / gridDim.x) + sNumSig % gridDim.x != 0 ? 1 : 0;

    // TODO: Check for bank conflicts
    db_offset_t *shsIndexes = (db_offset_t *)shared;

    // + 1 so that we allocate one more index space for the start index of
    // the following signature to allow us to calculate the size of the last signature
    FLOAT *shsData = (FLOAT *)(shsIndexes + sSigPerBlock + 1);
    FLOAT *shtSig = shsData + sSigPerBlock * maxSignatureSize;
    FLOAT *shSums = shtSig + maxSignatureSize;
    FLOAT *shsSelfSims = shSums + sSigPerBlock;


    // TODO: Off by one errors
    int blockStartSig = blockIdx.x * sSigPerBlock + kernelID;
    int blockEndSig = min((blockIdx.x + 1) * sSigPerBlock, (int)sNumSig);
    int blockNumSig = blockEndSig - blockStartSig;

    const db_offset_t sDataBaseIndex = sIndexes[0];
    copySigData<DIM, FLOAT>(sIndexes + blockStartSig, sData, blockNumSig, sDataBaseIndex, shsData);
    globalToSharedIndexes(sIndexes + blockStartSig, blockNumSig, sDataBaseIndex, shsIndexes);

    __syncthreads();

    for (int sSig = 0; sSig < blockNumSig; ++sSig) {
        FLOAT selfSim = getSelfSim(Signature<DIM, FLOAT>(shsIndexes[sSig], shsIndexes[sSig + 1], shsData), alpha, shSums);
        if (threadIdx.x == 0) {
            shsSelfSims[sSig] = selfSim;
        }
    }

    for (int tSigIdx = 0; tSigIdx < tNumSig; ++tSigIdx) {
        // TODO: Fix the 0 if we want to process anything but the whole dataset as tSig
        copySigData<DIM, FLOAT>(tIndexes + tSigIdx, tData, 1, 0, shtSig);
        auto tSigNumCentroids = tIndexes[tSigIdx + 1] - tIndexes[tSigIdx];
        __syncthreads();

        FLOAT tSigSelfSim = getSelfSim(Signature<DIM, FLOAT>(0, tSigNumCentroids, shtSig), alpha, shSums);

        for (int i = 0; i < blockNumSig; ++i) {
            auto otherSim = getOtherSim(
                Signature<DIM, FLOAT>(shsIndexes[i], shsIndexes[i + 1], shsData),
                Signature<DIM, FLOAT>(0, tSigNumCentroids, shtSig),
                alpha,
                shSums);

            if (threadIdx.x == 0) {
                sScores[blockStartSig + i] += shsSelfSims[i] + tSigSelfSim + otherSim;
            }
        }
    }
}

template<int DIM, typename FLOAT>
void runComputeAssignmentsConsolidated(
    const db_offset_t *indexes,
    const FLOAT *data,
    const std::size_t numSig,
    const db_offset_t *medIndexes,
    const FLOAT *medData,
    const std::size_t numMed,
    const FLOAT alpha,
    const std::size_t maxSignatureSize,
    std::size_t * assignments,
    int blockSize,
    int numBlocks,
    cudaStream_t stream
) {
    std::size_t dataSigPerBlock = (numSig / numBlocks + (numSig % numBlocks != 0 ? 1 : 0));
    std::size_t shMemSize = getShMemSizeAssignment(maxSignatureSize, blockSize, dataSigPerBlock);
    computeAssignmentsConsolidated<DIM, FLOAT><<<numBlocks, blockSize, shMemSize, stream>>>(indexes, data, numSig, medIndexes, medData, numMed, alpha, maxSignatureSize, assignments);
}

template<int DIM, typename FLOAT>
void runGetScores(
    const db_offset_t *sIndexes,
    const FLOAT *sData,
    const std::size_t * sAssignments,
    const std::size_t sNumSig,
    const db_offset_t *tIndexes,
    const FLOAT *tData,
    const std::size_t tNumSig,
    const std::size_t * tAssignments,
    const FLOAT alpha,
    const std::size_t maxSignatureSize,
    const std::size_t kernelID,
    FLOAT *sScores,
    int blockSize,
    int numBlocks,
    cudaStream_t stream
) {
    std::size_t sSigPerBlock = (sNumSig / numBlocks + (sNumSig % numBlocks != 0 ? 1 : 0));
    std::size_t shMemSize = getShMemSizeScores(maxSignatureSize, blockSize, sSigPerBlock);
    getScores<DIM, FLOAT><<<numBlocks, blockSize, shMemSize, stream>>>(
        sIndexes,
        sData,
        sAssignments,
        sNumSig,
        tIndexes,
        tData,
        tAssignments,
        tNumSig,
        alpha,
        maxSignatureSize,
        kernelID,
        sScores
    );
}

template void runComputeAssignmentsConsolidated<7, float>(
    const db_offset_t *indexes,
    const float *data,
    const std::size_t numSig,
    const db_offset_t *medIndexes,
    const float *medData,
    const std::size_t numMed,
    const float alpha,
    const std::size_t maxSignatureSize,
    std::size_t * assignments,
    int blockSize,
    int numBlocks,
    cudaStream_t stream
);

template void runGetScores<7, float>(
    const db_offset_t *sIndexes,
    const float *sData,
    const std::size_t * sAssignments,
    const std::size_t sNumSig,
    const db_offset_t *tIndexes,
    const float *tData,
    const std::size_t tNumSig,
    const std::size_t * tAssignments,
    const float alpha,
    const std::size_t maxSignatureSize,
    const std::size_t kernelID,
    float *sScores,
    int blockSize,
    int numBlocks,
    cudaStream_t stream
);