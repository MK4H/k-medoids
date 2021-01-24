#ifndef K_MEDOIDS_KERNELS_CUH
#define K_MEDOIDS_KERNELS_CUH

#include "commons.hpp"

template<int DIM, typename FLOAT = float>
std::size_t getShMemSizeAssignment(std::size_t maxSignatureSize, int blockSize, std::size_t sigPerBlock) {
    return sizeof(db_offset_t) * (sigPerBlock + 1) +
        sizeof(std::size_t) * sigPerBlock +
        sizeof(FLOAT) * maxSignatureSize * (DIM + 1) +
        sizeof(FLOAT) * sigPerBlock +
        sizeof(FLOAT) * blockSize +
        sizeof(FLOAT) * sigPerBlock +
        sizeof(FLOAT) * maxSignatureSize * sigPerBlock * (DIM + 1);
}

template<int DIM, typename FLOAT = float>
std::size_t getMaxSigPerBlockAssignment(std::size_t shMemSize, std::size_t maxSignatureSize, int blockSize) {
    return (
        shMemSize -
        sizeof(FLOAT) * maxSignatureSize * (DIM + 1) -
        sizeof(FLOAT) * blockSize -
        sizeof(db_offset_t)
    ) / (
        sizeof(db_offset_t) + sizeof(std::size_t) + sizeof(FLOAT) + sizeof(FLOAT) + sizeof(FLOAT) * maxSignatureSize * (DIM + 1)
    );
}

template<int DIM, typename FLOAT = float>
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
);

template<int DIM, typename FLOAT = float>
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
);

template<int DIM, typename FLOAT = float>
std::size_t getShMemSizeScores(std::size_t maxSignatureSize, int blockSize, std::size_t sigPerBlock) {
    return sizeof(FLOAT) * maxSignatureSize * (DIM + 1) +
        sizeof(FLOAT) * sigPerBlock +
        sizeof(FLOAT) * blockSize +
        sizeof(std::size_t) * sigPerBlock +
        sizeof(db_offset_t) * (sigPerBlock + 1) +
        sizeof(FLOAT) * maxSignatureSize * sigPerBlock * (DIM + 1) +
        sizeof(bool);
}

template<int DIM, typename FLOAT = float>
std::size_t getMaxSigPerBlockScores(std::size_t shMemSize, std::size_t maxSignatureSize, int blockSize) {
    return (
        shMemSize -
        sizeof(FLOAT) * maxSignatureSize * (DIM + 1) -
        sizeof(FLOAT) * blockSize -
        sizeof(db_offset_t) -
        sizeof(bool)
    ) / (
        sizeof(FLOAT) + sizeof(std::size_t) + sizeof(db_offset_t) + sizeof(FLOAT) * maxSignatureSize * (DIM + 1)
    );
}

template<int DIM, typename FLOAT = float>
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
);

template<int DIM, typename FLOAT = float>
std::size_t getShMemSizeScoresPreprocessed(std::size_t maxSignatureSize, int blockSize, std::size_t sigPerBlock) {
    return sizeof(db_offset_t) * (sigPerBlock + 1) +
        sizeof(FLOAT) * sigPerBlock * maxSignatureSize * (DIM + 1) +
        sizeof(FLOAT) * maxSignatureSize * (DIM + 1) +
        sizeof(FLOAT) * blockSize +
        sizeof(FLOAT) * sigPerBlock +
        sizeof(FLOAT) * sigPerBlock;
}

template<int DIM, typename FLOAT = float>
std::size_t getMaxSigPerBlockScoresPreprocessed(std::size_t shMemSize, std::size_t maxSignatureSize, int blockSize) {
    return (
        shMemSize -
        sizeof(db_offset_t) -
        sizeof(FLOAT) * maxSignatureSize * (DIM + 1) -
        sizeof(FLOAT) * blockSize
    ) / (
        sizeof(db_offset_t) + sizeof(FLOAT) * maxSignatureSize * (DIM + 1) + sizeof(FLOAT) + sizeof(FLOAT)
    );
}

template<int DIM, typename FLOAT = float>
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
);

template<int DIM, typename FLOAT = float>
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
    cudaStream_t stream
);

template<typename T>
void runZeroOut(T *data, std::size_t size, std::size_t itemsPerThread, cudaStream_t stream = (cudaStream_t)0);

#endif