#include "kernels.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_constants.h"

#include <cstdint>


template<int DIM, typename FLOAT = float>
struct Signature {
    const std::size_t numCentroids;
    FLOAT *weights;
    const FLOAT *centroids;


    __device__ Signature(const db_offset_t startIdx, const db_offset_t endIdx, FLOAT *data)
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
    unsigned int tid = threadIdx.x;
    __syncthreads();
    // Adapted from http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
    for (unsigned int stride = blockDim.x / 2; stride > warpSize; stride /= 2) {
        if (tid < stride) {
            partSums[tid] += partSums[tid + stride];
        }
        __syncthreads();
    }


    if (tid < warpSize) {
        total = partSums[tid] + partSums[tid + warpSize];
        for (int stride = warpSize / 2; stride > 0; stride /= 2) {
            total += __shfl_down_sync(0xFFFFFFFF, total, stride);
        }
    }

    if (tid == 0) {
        partSums[0] = total;
    }

    __syncthreads();
    // Broadcast the sum to all threads of the block
    total = partSums[0];

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

template<int DIM, typename FLOAT, bool sync>
__device__ void normalizeWeights(Signature<DIM, FLOAT> sig, FLOAT *shSums) {
    FLOAT localSum = 0;
    for (int i = threadIdx.x; i < sig.numCentroids; i += blockDim.x) {
        localSum += sig.weights[i];
    }
    shSums[threadIdx.x] = localSum;
    FLOAT normalizer = mergeSums(shSums);

    for (int i = threadIdx.x; i < sig.numCentroids; i += blockDim.x) {
        sig.weights[i] = sig.weights[i] / normalizer;
    }

    if (sync) {
        __syncthreads();
    }
}

template<int DIM, typename FLOAT = float>
__device__ std::size_t copySigData(const db_offset_t *inIdx, const FLOAT *inAllData, int numSig, db_offset_t baseIndex, FLOAT *outData) {
    auto startOffset = idxToOffset(getLocalIndex(inIdx[0], baseIndex), DIM);
    auto endOffset = idxToOffset(getLocalIndex(inIdx[numSig], baseIndex), DIM);

    // Just copy continuous data to shared memory
    for (auto i = threadIdx.x; i < endOffset - startOffset; i += blockDim.x) {
        outData[i] = inAllData[startOffset + i];
    }

    // Return number of copied centroids
    return inIdx[numSig] - inIdx[0];
}

/**
 * Copy non-continuous signatures from data to continuous memory in outData.
 * Also creates outIndexes to access the outData.
 */
template<int DIM, typename FLOAT = float>
__device__ void consolidateSigData(const db_offset_t *indexes, const FLOAT *data, const std::size_t *clusterList, std::size_t startIndex, std::size_t num, db_offset_t *outIndexes, FLOAT *outData) {

    outIndexes[0] = 0;
    for (int i = 0; i < num; ++i) {
        auto sigId = clusterList[startIndex + i];
        auto cpCentroids = copySigData<DIM, FLOAT>(indexes + sigId, data, 1, 0, outData);
        outData += cpCentroids * (DIM + 1);
        if (threadIdx.x == 0) {
            outIndexes[i + 1] = outIndexes[i] + cpCentroids;
        }
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
 * All data muste be aligned to the size of its type. So for example db_offset_t, which is 8 byte, MUST be aligned
 * to 8 bytes. This is why we allocatate shared array as db_offset_t and put everything using db_offset_t first
 *
 * The FLOAT may be 4 byte float or 8 byte double, which will both work when put after db_offset_t
 *
 * The amount of shared memory needed is the sum of:
 *  sizeof(db_offset_t) * (numDataSignaturesPerBlock + 1)           // For index table of data signatures
 *  sizeof(std::size_t) / (numDataSignaturesPerBlock)               // For intermediate assignments
 *  sizeof(FLOAT) * maxSignatureSize * (DIM + 1)                    // For current medoid
 *  sizeof(FLOAT) * numDataSignaturesPerBlock                       // For current minimal distance to medoid of each data signature
 *  sizeof(FLOAT) * blockDim.x                                      // For partial sums during SQDF computation
 *  sizeof(FLOAT) * numDataSignaturesPerBlock                       // For precomputed selfSimilarities of each data signature
 *
 *  sizeof(FLOAT) * maxSignatureSize * numDataSIgnaturesPerBlock * (DIM + 1)    // For the data signatures themselfs
 */
template<int DIM, typename FLOAT = float>
__device__ void partitionSharedMemAssign(
    db_offset_t *shared,
    std::size_t maxSignatureSize,
    int sigPerBlock,
    db_offset_t *&shIndexes,
    std::size_t *&shAssignments,
    FLOAT *&med,
    FLOAT *&mins,
    FLOAT *&shSums,
    FLOAT *&dSelfSims,
    FLOAT *&shData
) {
    // TODO: Check for bank conflicts
    // Split up the allocated shared memory

    // 8 byte types
    shIndexes = shared;

    shAssignments = (std::size_t*)(shIndexes + sigPerBlock + 1);
    // FLOAT (4 or 8 byte)

    // + 1 so that we allocate one more index space for the start index of
    // the following signature to allow us to calculate the size of the last signature
    med = (FLOAT *)(shAssignments + sigPerBlock);
    mins = med + maxSignatureSize * (DIM + 1);
    // TODO: Use warp reduce and reduce the size of shSum to one element per warp
    // We need one sum entry for each thread
    shSums = mins + sigPerBlock;
    dSelfSims = shSums + blockDim.x;

    shData = dSelfSims + sigPerBlock;

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
    std::size_t * kernAssignments
) {
    // Must be declared using the largest type
    // All access to memory in CUDA MUST be aligned
    // So accessing 4 byte type must be aligned to 4 bytes
    extern __shared__  db_offset_t shared[];

    // Number of blocks will always divide the number of Signatures without remainder
    // APART from the last kernel call, which will just have to compute the rest of the data
    const int sigPerBlock = (numSig / gridDim.x) + (numSig % gridDim.x != 0 ? 1 : 0);

    // SHARED MEMORY PARTITIONING

    db_offset_t *shIndexes;
    std::size_t *shAssignments;
    FLOAT *med, *mins, *shSums, *dSelfSims, *shData;
    partitionSharedMemAssign<DIM, FLOAT>(
        shared,
        maxSignatureSize,
        sigPerBlock,
        shIndexes,
        shAssignments,
        med,
        mins,
        shSums,
        dSelfSims,
        shData
    );

    // Initialize mins
    for (int i = threadIdx.x; i < sigPerBlock; i += blockDim.x) {
        mins[i] = CUDART_INF_F;
    }

    int blockStartSig = blockIdx.x * sigPerBlock;
    // As the processed signatures have to fit into shared memory, int is enough
    int blockEndSig = min((blockIdx.x + 1) * sigPerBlock, (int)numSig);
    int blockNumSig = blockEndSig - blockStartSig;


    const db_offset_t dataBaseIndex = indexes[0];
    copySigData<DIM, FLOAT>(indexes + blockStartSig, data, blockNumSig, dataBaseIndex, shData);
    globalToSharedIndexes(indexes + blockStartSig, blockNumSig, dataBaseIndex, shIndexes);

    __syncthreads();
    for (unsigned int dataSig = 0; dataSig < blockNumSig; ++dataSig) {
        normalizeWeights<DIM, FLOAT, true>(Signature<DIM, FLOAT>(shIndexes[dataSig], shIndexes[dataSig + 1], shData), shSums);
    }

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
        auto medSig = Signature<DIM, FLOAT>(0, medIndexes[medIdx + 1] - medIndexes[medIdx], med);

        __syncthreads();

        normalizeWeights<DIM, FLOAT, true>(medSig, shSums);

        FLOAT medSelfSim = getSelfSim(medSig, alpha, shSums);

        for (int i = 0; i < blockNumSig; ++i) {

            auto otherSim = getOtherSim(Signature<DIM, FLOAT>(shIndexes[i], shIndexes[i + 1], shData), medSig, alpha, shSums);
            if (threadIdx.x == 0) {
                FLOAT sqfd2 = dSelfSims[i] + medSelfSim + otherSim;
                if (sqfd2 < mins[i]) {
                    mins[i] = sqfd2;
                    shAssignments[i] = medIdx;
                }
            }
        }
    }

    for (int dataSig = threadIdx.x; dataSig < blockNumSig; dataSig += blockDim.x) {
        kernAssignments[blockStartSig + dataSig] = shAssignments[dataSig];
    }

}

/**
 * Compute assignments without consolidating medoids, take advantage of the fact that the whole data fits into memory
 */
template<int DIM, typename FLOAT = float>
__global__ void computeAssignments(
    const db_offset_t *kernIndexes,
    const FLOAT *kernData,
    const std::size_t kernNumSig,
    const db_offset_t *allIndexes,
    const FLOAT *allData,
    const std::size_t allNumSig,
    const std::size_t *medSeqIdxs,
    const std::size_t numMed,
    const FLOAT alpha,
    const std::size_t maxSignatureSize,
    std::size_t * kernAssignments
) {
    // Must be declared using the largest type
    // All access to memory in CUDA MUST be aligned
    // So accessing 4 byte type must be aligned to 4 bytes
    extern __shared__  db_offset_t shared[];

    // Number of blocks will always divide the number of Signatures without remainder
    // APART from the last kernel call, which will just have to compute the rest of the data
    const int sigPerBlock = (kernNumSig / gridDim.x) + (kernNumSig % gridDim.x != 0 ? 1 : 0);

    // SHARED MEMORY PARTITIONING

    db_offset_t *shIndexes;
    std::size_t *shAssignments;
    FLOAT *med, *mins, *shSums, *dSelfSims, *shData;
    partitionSharedMemAssign<DIM, FLOAT>(
        shared,
        maxSignatureSize,
        sigPerBlock,
        shIndexes,
        shAssignments,
        med,
        mins,
        shSums,
        dSelfSims,
        shData
    );

    // Initialize mins
    for (int i = threadIdx.x; i < sigPerBlock; i += blockDim.x) {
        mins[i] = CUDART_INF_F;
    }

    int blockStartSig = blockIdx.x * sigPerBlock;
    // As the processed signatures have to fit into shared memory, int is enough
    int blockEndSig = min(blockStartSig + sigPerBlock, (int)kernNumSig);
    int blockNumSig = blockEndSig - blockStartSig;


    const db_offset_t dataBaseIndex = kernIndexes[0];
    copySigData<DIM, FLOAT>(kernIndexes + blockStartSig, kernData, blockNumSig, dataBaseIndex, shData);
    globalToSharedIndexes(kernIndexes + blockStartSig, blockNumSig, dataBaseIndex, shIndexes);

    __syncthreads();
    for (unsigned int dataSig = 0; dataSig < blockNumSig; ++dataSig) {
        normalizeWeights<DIM, FLOAT, true>(Signature<DIM, FLOAT>(shIndexes[dataSig], shIndexes[dataSig + 1], shData), shSums);
    }

    for (int dataSig = 0; dataSig < blockNumSig; ++dataSig) {
        FLOAT selfSim = getSelfSim(Signature<DIM, FLOAT>(shIndexes[dataSig], shIndexes[dataSig + 1], shData), alpha, shSums);
        if (threadIdx.x == 0) {
            dSelfSims[dataSig] = selfSim;
        }
    }

    for (int medIdx = 0; medIdx < numMed; ++medIdx) {
        // Load medoid to shared mem
        // Reading from all so the base index is 0
        // THE ONLY THING DIFFERENT FROM CONSOLIDATED
        auto medSeqId = medSeqIdxs[medIdx];
        copySigData<DIM, FLOAT>(allIndexes + medSeqId, allData, 1, 0, med);
        auto medSig = Signature<DIM, FLOAT>(0, allIndexes[medSeqId + 1] - allIndexes[medSeqId], med);

        __syncthreads();

        normalizeWeights<DIM, FLOAT, true>(medSig, shSums);

        FLOAT medSelfSim = getSelfSim(medSig, alpha, shSums);

        for (int i = 0; i < blockNumSig; ++i) {

            auto otherSim = getOtherSim(Signature<DIM, FLOAT>(shIndexes[i], shIndexes[i + 1], shData), medSig, alpha, shSums);
            if (threadIdx.x == 0) {
                FLOAT sqfd2 = dSelfSims[i] + medSelfSim + otherSim;
                if (sqfd2 < mins[i]) {
                    mins[i] = sqfd2;
                    shAssignments[i] = medIdx;
                }
            }
        }
    }

    __syncthreads();

    for (int dataSig = threadIdx.x; dataSig < blockNumSig; dataSig += blockDim.x) {
        kernAssignments[blockStartSig + dataSig] = shAssignments[dataSig];
    }
}

template<int DIM, typename FLOAT = float>
__device__ void partitionSharedMemScore(
    db_offset_t *shared,
    std::size_t maxSignatureSize,
    int sSigPerBlock,
    db_offset_t *&shsIndexes,
    std::size_t *&shsAssignments,
    FLOAT *&shsData,
    FLOAT *&shtSig,
    FLOAT *&shSums,
    FLOAT *&shsSelfSims,
    bool *&sAnyMatch
) {
    // TODO: Check for bank conflicts
    // 8 byte types
    shsIndexes = shared;
    shsAssignments = (std::size_t *)(shsIndexes + sSigPerBlock + 1);
    // FLOAT (4 or 8 bytes)
    // + 1 so that we allocate one more index space for the start index of
    // the following signature to allow us to calculate the size of the last signature
    shsData = (FLOAT *)(shsAssignments + sSigPerBlock);
    shtSig = shsData + sSigPerBlock * maxSignatureSize * (DIM + 1);
    shSums = shtSig + maxSignatureSize * (DIM + 1);
    shsSelfSims = shSums + blockDim.x;
    sAnyMatch = (bool *)(shsSelfSims + sSigPerBlock);
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
    FLOAT *sScores) {

    extern __shared__  db_offset_t shared[];

    // Number of blocks will always divide the number of Signatures without remainder
    // APART from the last kernel call, which will just have to compute the rest of the data
    const int sSigPerBlock = (sNumSig / gridDim.x) + (sNumSig % gridDim.x != 0 ? 1 : 0);

    db_offset_t *shsIndexes;
    std::size_t *shsAssignments;
    FLOAT *shsData, *shtSig, *shSums, *shsSelfSims;
    bool *sAnyMatch;
    partitionSharedMemScore<DIM, FLOAT>(shared, maxSignatureSize, sSigPerBlock, shsIndexes, shsAssignments, shsData, shtSig, shSums, shsSelfSims, sAnyMatch);

    int blockStartSig = blockIdx.x * sSigPerBlock;
    int blockEndSig = min((blockIdx.x + 1) * sSigPerBlock, (int)sNumSig);
    int blockNumSig = blockEndSig - blockStartSig;

    const db_offset_t sDataBaseIndex = sIndexes[0];
    copySigData<DIM, FLOAT>(sIndexes + blockStartSig, sData, blockNumSig, sDataBaseIndex, shsData);
    globalToSharedIndexes(sIndexes + blockStartSig, blockNumSig, sDataBaseIndex, shsIndexes);

    for (int sSig = threadIdx.x; sSig < blockNumSig; sSig += blockDim.x) {
        shsAssignments[sSig] = sAssignments[blockStartSig + sSig];
    }

    __syncthreads();

    for (unsigned int sSig = 0; sSig < blockNumSig; ++sSig) {
        normalizeWeights<DIM, FLOAT, true>(Signature<DIM, FLOAT>(shsIndexes[sSig], shsIndexes[sSig + 1], shsData), shSums);
    }

    for (int sSig = 0; sSig < blockNumSig; ++sSig) {
        FLOAT selfSim = getSelfSim(Signature<DIM, FLOAT>(shsIndexes[sSig], shsIndexes[sSig + 1], shsData), alpha, shSums);
        if (threadIdx.x == 0) {
            shsSelfSims[sSig] = selfSim;
        }
    }

    for (int tSigIdx = 0; tSigIdx < tNumSig; ++tSigIdx) {
        auto tAssignment = tAssignments[tSigIdx];

        // Done just by the first warp, so that we can use warp level primitives
        if (threadIdx.x < 32) {
            bool clusterMatch = false;
            for (int sSig = threadIdx.x; sSig < blockNumSig; sSig += 32) {
                clusterMatch |= tAssignment == shsAssignments[sSig];
            }
            clusterMatch = __any_sync(0xFFFFFFFF, clusterMatch);
            if (threadIdx.x == 0) {
                *sAnyMatch = clusterMatch;
            }
        }
        __syncthreads();

        if (!*sAnyMatch) {
            // No source signature is in the same cluster as target signature
            continue;
        }

        // TODO: Fix the 0 if we want to process anything but the whole dataset as tSig
        copySigData<DIM, FLOAT>(tIndexes + tSigIdx, tData, 1, 0, shtSig);
        auto tSig = Signature<DIM, FLOAT>(0, tIndexes[tSigIdx + 1] - tIndexes[tSigIdx], shtSig);
        __syncthreads();

        normalizeWeights<DIM, FLOAT, true>(tSig, shSums);

        FLOAT tSigSelfSim = getSelfSim(tSig, alpha, shSums);

        for (int sSig = 0; sSig < blockNumSig; ++sSig) {
            if ((tAssignment != shsAssignments[sSig]) || (sDataBaseIndex + shsIndexes[sSig] == tIndexes[tSigIdx])) {
                continue;
            }

            auto otherSim = getOtherSim(
                Signature<DIM, FLOAT>(shsIndexes[sSig], shsIndexes[sSig + 1], shsData),
                tSig,
                alpha,
                shSums);

            if (threadIdx.x == 0) {
                sScores[blockStartSig + sSig] += shsSelfSims[sSig] + tSigSelfSim + otherSim;
            }
        }
    }
}

/**
 * Calculate partial scores of signatures from a single cluster.
 * This function is designed so that we can split computation of
 * scores for signatures from large clusters into multiple parallel
 * computations.
 *
 * This is an n^2 problem, where we need to compute distance for every pair of source image
 * and target image. This lends itself nicely to using two dimensional kernel.
 *
 * Each block of this kernel loads different block of source signatures, based on its
 * blockId in x axis, and processes different block of target signatures based on its blockId in y axis
 *
 * \param indexes Index of the whole image database
 * \param data Image database data
 * \param assignment Cluster assignment of each image
 * \param clusterList List of image indexes, based on their assigned cluster in increasing order.
 *       First there are all images from cluster 0, then all from cluster 1, etc.
 * \param clusterListIndex Index of clusterList, where clusterList[clusterListIndex[i] is the
 *      first image of the cluster i
 * \param sourceStartIndex Index in the clusterList where the range of source images starts for this kernel
 * \param numSource Number of source images to process by this kernel
 * \param targetStartIndex Index in the clusterList where the range of target images starts for this kernel
 * \param numTargets Number of targets to process by this kernel
 * \param alpha
 * \param maxSignatureSize
 * \param sScores Source image scores which will be added to by this kernel
 *
 * We load the source signatures into shared memory, and then go through the target signatures one by one,
 * computing score of each source signature. After we go through the whole target range,
 * we add the score of each source signature atomically to the score in sScores.
 * This is to allow other kernels to compute partial scores for the same source nodes,
 * but with different target range, all in parallel.
 */
template<int DIM, typename FLOAT = float>
__global__ void getScoresPreprocessedLarge(
    const db_offset_t *indexes,
    const FLOAT *data,
    const std::size_t *assignment,
    const std::size_t *clusterList,
    const std::size_t sourceStartIndex,
    const std::size_t numSources,
    const std::size_t targetStartIndex,
    const std::size_t numTargets,
    const FLOAT alpha,
    const std::size_t maxSignatureSize,
    FLOAT *sScores) {

    // Must be declared using the largest type
    // All access to memory in CUDA MUST be aligned
    // So accessing 4 byte type must be aligned to 4 bytes
    extern __shared__  db_offset_t shared[];

    const int sourcesPerBlock = (numSources / gridDim.x) + (numSources % gridDim.x != 0 ? 1 : 0);
    const int targetsPerBlock = (numTargets / gridDim.y) + (numTargets % gridDim.y != 0 ? 1 : 0);

    db_offset_t *shIndexes;
    FLOAT *shData, *shtData, *shSums, *shSelfSims, *shScores;

    // TODO: Check for bank conflicts
    // 8 byte types
    shIndexes = shared;

    // FLOAT (4 or 8 bytes)
    // + 1 so that we allocate one more index space for the start index of
    // the following signature to allow us to calculate the size of the last signature
    shData = (FLOAT *)(shIndexes + sourcesPerBlock + 1);
    shtData = shData + sourcesPerBlock * maxSignatureSize * (DIM + 1);
    shSums = shtData + maxSignatureSize * (DIM + 1);
    shSelfSims = shSums + blockDim.x;
    shScores = shSelfSims + sourcesPerBlock;

    int blockStartSource = sourceStartIndex + blockIdx.x * sourcesPerBlock;
    int blockEndSource = min(blockStartSource + sourcesPerBlock, (int)(sourceStartIndex + numSources));

    if (blockStartSource >= blockEndSource) {
        return;
    }

    int blockNumSource = blockEndSource - blockStartSource;


    for (int i = threadIdx.x; i < sourcesPerBlock; i += blockDim.x) {
        shScores[i] = 0;
    }

    consolidateSigData<DIM, FLOAT>(indexes, data, clusterList, blockStartSource, blockNumSource, shIndexes, shData);

    __syncthreads();

    for (unsigned int i = 0; i < blockNumSource; ++i) {
        normalizeWeights<DIM, FLOAT, true>(Signature<DIM, FLOAT>(shIndexes[i], shIndexes[i + 1], shData), shSums);
    }

    for (unsigned int i = 0; i < blockNumSource; ++i) {
        FLOAT selfSim = getSelfSim(Signature<DIM, FLOAT>(shIndexes[i], shIndexes[i + 1], shData), alpha, shSums);
        if (threadIdx.x == 0) {
            shSelfSims[i] = selfSim;
        }
    }


    int blockStartTarget = targetStartIndex + blockIdx.y * targetsPerBlock;
    int blockEndTarget = min(blockStartTarget + targetsPerBlock, (int)(targetStartIndex + numTargets));

    for (unsigned int t = blockStartTarget; t < blockEndTarget; ++t) {
        const db_offset_t *tIndex = indexes + clusterList[t];
        copySigData<DIM, FLOAT>(tIndex, data, 1, 0, shtData);
        auto tSig = Signature<DIM, FLOAT>(0, *(tIndex + 1) - *tIndex, shtData);
        __syncthreads();

        normalizeWeights<DIM, FLOAT, true>(tSig, shSums);
        FLOAT tSelfSim = getSelfSim(tSig, alpha, shSums);

        for (unsigned int s = 0; s < blockNumSource; ++s) {
            auto otherSim = getOtherSim(
                Signature<DIM, FLOAT>(shIndexes[s], shIndexes[s + 1], shData),
                tSig,
                alpha,
                shSums);

            if (threadIdx.x == 0) {
                shScores[s] += shSelfSims[s] + tSelfSim + otherSim;
            }
        }
    }

    __syncthreads();
    for (unsigned int s = threadIdx.x; s < blockNumSource; s += blockDim.x) {
        atomicAdd(sScores + blockStartSource + s, shScores[s]);
        //atomicAdd(sScores + blockStartSource + s, 1);
    }
}

/**
 * Calculate scores for signatures from multiple small clusters
 * This function expects that it is the only one
 * computing score for each of the source signatures it processes
 * as these are only small clusters which don't need to be split
 *
 * This function takes sourcesPerBlock signatures starting from clusterList at startIndex.
 * clusterList contains image sequential ids ordered according to the cluster they belong to.
 * So first are all images from cluster 0, then all from cluster 1, then all from cluster 2 etc.
 * clusterListIndex[i] tells us index in clusterList where cluster i starts
 *
 * We load the source signatures to shared memory in order of increasing cluster number
 * Then we get the cluster number of the first source signature and we go through
 * the whole cluster in clusterList, computing distances of all source signatures from
 * this cluster to all targets in the cluster.
 *
 * Once we go through all the targets in the first cluster, we get to the second cluster,
 * and we start computing distances of all source signatures from the second cluster
 * to all targets in the second cluster.
 *
 * We go like this through all the clusters the source signatures are from.
 * As we compute the whole score for each source signature, we expect that we are
 * the only ones writing to the given value in sScores.
 */
template<int DIM, typename FLOAT = float>
__global__ void getScoresPreprocessedSmall(
    const db_offset_t *indexes,
    const FLOAT *data,
    const std::size_t *assignment,
    const std::size_t *clusterList,
    const std::size_t *clusterListIndex,
    const std::size_t startIndex,
    const std::size_t numSources,
    const FLOAT alpha,
    const std::size_t maxSignatureSize,
    FLOAT *sScores) {

    // Must be declared using the largest type
    // All access to memory in CUDA MUST be aligned
    // So accessing 4 byte type must be aligned to 4 bytes
    extern __shared__  db_offset_t shared[];

    const int sourcesPerBlock = (numSources / gridDim.x) + (numSources % gridDim.x != 0 ? 1 : 0);

    db_offset_t *shIndexes;
    FLOAT *shData, *shtData, *shSums, *shSelfSims, *shScores;

    // TODO: Check for bank conflicts
    // 8 byte types
    shIndexes = shared;

    // FLOAT (4 or 8 bytes)
    // + 1 so that we allocate one more index space for the start index of
    // the following signature to allow us to calculate the size of the last signature
    shData = (FLOAT *)(shIndexes + sourcesPerBlock + 1);
    shtData = shData + sourcesPerBlock * maxSignatureSize * (DIM + 1);
    shSums = shtData + maxSignatureSize * (DIM + 1);
    shSelfSims = shSums + blockDim.x;
    shScores = shSelfSims + sourcesPerBlock;

    // TODO: These ints may be a problem for very large datasets
    int blockStartSource = startIndex + blockIdx.x * sourcesPerBlock;
    int blockEndSource = min(blockStartSource + sourcesPerBlock, (int)(startIndex + numSources));

    if (blockStartSource >= blockEndSource) {
        return;
    }

    int blockNumSource = blockEndSource - blockStartSource;


    // Initialize scores
    for (int i = threadIdx.x; i < sourcesPerBlock; i += blockDim.x) {
        shScores[i] = 0;
    }

    consolidateSigData<DIM, FLOAT>(indexes, data, clusterList, blockStartSource, blockNumSource, shIndexes, shData);

    __syncthreads();
    for (unsigned int i = 0; i < blockNumSource; ++i) {
        normalizeWeights<DIM, FLOAT, true>(Signature<DIM, FLOAT>(shIndexes[i], shIndexes[i + 1], shData), shSums);
    }

    for (unsigned int i = 0; i < blockNumSource; ++i) {
        FLOAT selfSim = getSelfSim(Signature<DIM, FLOAT>(shIndexes[i], shIndexes[i + 1], shData), alpha, shSums);
        if (threadIdx.x == 0) {
            shSelfSims[i] = selfSim;
        }
    }

    auto curCluster = assignment[clusterList[blockStartSource]];
    auto curClusterStart = clusterListIndex[curCluster];
    auto curClusterEnd = clusterListIndex[curCluster + 1];
    // BlockEndSource is the first image after the last processed source image
    // So -1 is the last processed image, and its cluster + 1 is the first cluster
    // after the clusters spanned by this blocks sources
    auto lastClusterEnd = clusterListIndex[assignment[clusterList[blockEndSource - 1]] + 1];

    auto sourceStart = 0;
    // Process either blockNumSource if all sources of this block are from the same cluster
    auto sourceEnd = min(blockNumSource, (int)(curClusterEnd - blockStartSource));
    for (unsigned int t = curClusterStart; t < lastClusterEnd; ++t) {

        const db_offset_t *tIndex = indexes + clusterList[t];
        copySigData<DIM, FLOAT>(tIndex, data, 1, 0, shtData);
        auto tSig = Signature<DIM, FLOAT>(0, *(tIndex + 1) - *tIndex, shtData);
        __syncthreads();

        normalizeWeights<DIM, FLOAT, true>(tSig, shSums);
        FLOAT tSelfSim = getSelfSim(tSig, alpha, shSums);

        for (unsigned int s = sourceStart; s < sourceEnd; ++s) {
            auto otherSim = getOtherSim(
                Signature<DIM, FLOAT>(shIndexes[s], shIndexes[s + 1], shData),
                tSig,
                alpha,
                shSums);

            if (threadIdx.x == 0) {
                shScores[s] += shSelfSims[s] + tSelfSim + otherSim;
            }
        }

        if (t == curClusterEnd - 1) {
            sourceStart = sourceEnd;
            curCluster += 1;
            curClusterEnd = clusterListIndex[curCluster + 1];
            sourceEnd = min(blockNumSource, (int)(curClusterEnd - blockStartSource));
        }
    }

    __syncthreads();
    for (unsigned int s = threadIdx.x; s < blockNumSource; s += blockDim.x) {
        // This function expects that it is the only one
        // computing score for each of the source signatures it processes
        // as these are only small clusters which don't need to be split
        sScores[blockStartSource + s] = shScores[s];
    }
}

template<typename T>
__global__ void zeroOut(T *data, std::size_t size) {
    for (std::size_t i = threadIdx.x; i < size; i += blockDim.x) {
        data[i] = 0;
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
    std::size_t * kernAssignments,
    int blockSize,
    int numBlocks,
    cudaStream_t stream
) {
    std::size_t dataSigPerBlock = (numSig / numBlocks + (numSig % numBlocks != 0 ? 1 : 0));
    std::size_t shMemSize = getShMemSizeAssignment<DIM, FLOAT>(maxSignatureSize, blockSize, dataSigPerBlock);

    computeAssignmentsConsolidated<DIM, FLOAT><<<numBlocks, blockSize, shMemSize, stream>>>(indexes, data, numSig, medIndexes, medData, numMed, alpha, maxSignatureSize, kernAssignments);
}

template<int DIM, typename FLOAT>
void runComputeAssignments(
    const db_offset_t *kernIndexes,
    const FLOAT *kernData,
    const std::size_t kernNumSig,
    const db_offset_t *allIndexes,
    const FLOAT *allData,
    const std::size_t allNumSig,
    const std::size_t *medSeqIdxs,
    const std::size_t numMed,
    const FLOAT alpha,
    const std::size_t maxSignatureSize,
    std::size_t * kernAssignments,
    int blockSize,
    int numBlocks,
    cudaStream_t stream
) {
    std::size_t dataSigPerBlock = (kernNumSig / numBlocks + (kernNumSig % numBlocks != 0 ? 1 : 0));
    std::size_t shMemSize = getShMemSizeAssignment<DIM, FLOAT>(maxSignatureSize, blockSize, dataSigPerBlock);

    computeAssignments<DIM, FLOAT><<<numBlocks, blockSize, shMemSize, stream>>>(
        kernIndexes,
        kernData,
        kernNumSig,
        allIndexes,
        allData,
        allNumSig,
        medSeqIdxs,
        numMed,
        alpha,
        maxSignatureSize,
        kernAssignments
    );
}

template<int DIM, typename FLOAT>
void runGetScores(
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
    FLOAT *sScores,
    int blockSize,
    int numBlocks,
    cudaStream_t stream
) {
    std::size_t sSigPerBlock = (sNumSig / numBlocks + (sNumSig % numBlocks != 0 ? 1 : 0));
    std::size_t shMemSize = getShMemSizeScores<DIM, FLOAT>(maxSignatureSize, blockSize, sSigPerBlock);

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
        sScores
    );
}

template<int DIM, typename FLOAT>
void runGetScoresPreprocessedLarge(
    const db_offset_t *indexes,
    const FLOAT *data,
    const std::size_t *assignment,
    const std::size_t *clusterList,
    const std::size_t sourceStartIndex,
    const std::size_t numSources,
    const std::size_t targetStartIndex,
    const std::size_t numTargets,
    const FLOAT alpha,
    const std::size_t maxSignatureSize,
    FLOAT *sScores,
    int blockSize,
    int sourceBlocks,
    int targetBlocks,
    cudaStream_t stream
) {

    std::size_t sourcesPerBlock = (numSources / sourceBlocks + (numSources % sourceBlocks != 0 ? 1 : 0));
    std::size_t shMemSize = getShMemSizeScoresPreprocessed<DIM, FLOAT>(maxSignatureSize, blockSize, sourcesPerBlock);

    dim3 gridDim(sourceBlocks, targetBlocks);
    getScoresPreprocessedLarge<DIM, FLOAT><<<gridDim, blockSize, shMemSize, stream>>>(
        indexes,
        data,
        assignment,
        clusterList,
        sourceStartIndex,
        numSources,
        targetStartIndex,
        numTargets,
        alpha,
        maxSignatureSize,
        sScores
    );
}

template<int DIM, typename FLOAT>
void runGetScoresPreprocessedSmall(
    const db_offset_t *indexes,
    const FLOAT *data,
    const std::size_t *assignment,
    const std::size_t *clusterList,
    const std::size_t *clusterListIndex,
    const std::size_t startIndex,
    const std::size_t numSources,
    const FLOAT alpha,
    const std::size_t maxSignatureSize,
    FLOAT *sScores,
    int blockSize,
    int numBlocks,
    cudaStream_t stream) {

    std::size_t sSigPerBlock = (numSources / numBlocks + (numSources % numBlocks != 0 ? 1 : 0));
    std::size_t shMemSize = getShMemSizeScoresPreprocessed<DIM, FLOAT>(maxSignatureSize, blockSize, sSigPerBlock);

    getScoresPreprocessedSmall<DIM, FLOAT><<<numBlocks, blockSize, shMemSize, stream>>>(
        indexes,
        data,
        assignment,
        clusterList,
        clusterListIndex,
        startIndex,
        numSources,
        alpha,
        maxSignatureSize,
        sScores
    );
}

template<typename T>
void runZeroOut(T *data, std::size_t size, std::size_t itemsPerThread, cudaStream_t stream) {
    constexpr int threadsPerBlock = 256;
    std::size_t numBlocks = ((size / itemsPerThread) / threadsPerBlock) + 1;
    zeroOut<T><<<numBlocks, threadsPerBlock, 0, stream>>>(data, size);
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

template void runComputeAssignments<7, float>(
    const db_offset_t *kernIndexes,
    const float *kernData,
    const std::size_t kernNumSig,
    const db_offset_t *allIndexes,
    const float *allData,
    const std::size_t allNumSig,
    const std::size_t *medSeqIdxs,
    const std::size_t numMed,
    const float alpha,
    const std::size_t maxSignatureSize,
    std::size_t * kernAssignments,
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
    const std::size_t * tAssignments,
    const std::size_t tNumSig,
    const float alpha,
    const std::size_t maxSignatureSize,
    float *sScores,
    int blockSize,
    int numBlocks,
    cudaStream_t stream
);

template void runGetScoresPreprocessedLarge<7, float>(
    const db_offset_t *indexes,
    const float *data,
    const std::size_t *assignment,
    const std::size_t *clusterList,
    const std::size_t sourceStartIndex,
    const std::size_t numSources,
    const std::size_t targetStartIndex,
    const std::size_t numTargets,
    const float alpha,
    const std::size_t maxSignatureSize,
    float *sScores,
    int blockSize,
    int sourceBlocks,
    int targetBlocks,
    cudaStream_t stream
);

template void runGetScoresPreprocessedSmall<7, float>(
    const db_offset_t *indexes,
    const float *data,
    const std::size_t *assignment,
    const std::size_t *clusterList,
    const std::size_t *clusterListIndex,
    const std::size_t startIndex,
    const std::size_t numSources,
    const float alpha,
    const std::size_t maxSignatureSize,
    float *sScores,
    int blockSize,
    int numBlocks,
    cudaStream_t stream
);

template void runZeroOut<float>(float *data, std::size_t size, std::size_t itemsPerThread, cudaStream_t stream);