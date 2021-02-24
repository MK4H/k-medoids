# Final k-medoids

The objective is to implement k-medoids algorithm on image feature signatures where the measure (distance) is SQFD.
The problem was explained in detail in recordings (see the web of NPRG058).
Datasets are not in the repository, but you can found them on parlab NFS.

# Solution

My solution implements two algorithms.

First is consAlg, short for consolidating algorithm, which is named after the way it consolidates medoids to a continuous piece of memory before uploading them to GPUs for assignment computation. This algorithm was my initial implementation where are discovered what is really needed and what will work, so it is not very good.

The second algorithm is memAlg, named after the data property it takes advantage of. Because the whole dataset fits into GPU memory, that is where we upload it and leave it there for the whole duration of the algorithm.

The only things we move around are image assignments to clusters and image scores.

As for used technologies, I used MPI and CUDA. The general idea is to split the data into chunks of similiar size for each rank and compute data of all ranks in parallel. Inside each rank, we use CUDA to again process all data, now inside the rank, in parallel. What chunk of data is processed by each rank changes based on the step.

The two main steps are Assignment computation and Score computation, with few additional steps like class construction, initialization, cluster preprocessing and medoid computation.

## Class construction
First step is alg class construction. We allocate all the memory required to run the algorithm here, so that it is reused in each iteration.  It is then deallocated in class destructor.

## Initialization step

Then we run initialization step. In this step, we upload the whole dataset into the GPU memory. We could allocate the dataset buffers as unified memory and use just standard std::copy to get the data into this unified memory. This would automatically create interleaving between GPU kernel execution and memory trasnfers, while also making use of the fact that during Assignment computation, each rank accesses only the data belonging to the rank. Size of this data is dataset.size()/num_ranks. From the rest of the dataset, we only need the medoids, which there are only few. Unfortuanetly, during Score computation the access pattern is pretty much random access, so most likely the whole dataset would have to be transfered anyway.

We use the dataset as it is stored in the file, using the index structure and data basically unchanged. Only change is that we add the leading 0 to the index so that every access to the index is equivalent, regardless of the position of the data.

The dataset upload is done asynchronously. While the data transfer is running, we compute the initial medoids on rank 0, sort them as serial implementation does and then broadcast them to all other ranks.
We then wait for both the broadcast and the data upload to finish on each rank.

## Iterations
After the initalization, each rank has the dataset uploaded and has the intial medoids in host memory. We then go to run iterations of the algorithm.

### Assignments
First we compute assignments. We copy the current medoids from host memory to device memory and then start the GPU computation.

Assignment computation is perfectly data parallel, so assignment of each image can be computed in parallel with assignment of all the other images.

Our kernel, named `computeAssignments`, works as follows. We use whole blocks to compute the assignment of images. Each blocks loads `sigPerBlock` signatures to shared memory. We call these *source* signatures. These are continuous in memory, so it is very easy and fast. We then create the same index structure as we have for the whole dataset just for the `sigPerBlock` signatures in shared memory. Then we normalize weights of each signature. We use the whole block to normalize weights of each signature, so signatures are basically processed sequentially.

Then we precompute self similarity of the source signatures, so that it does not have to be computed again and again. We store this precomputed self similarity in shared memory.


Then we go through all the medoids one by one, copy the medoid data into shared memory, normalize weights, compute self similarity.

Then for each of these medoids, we go through all the source signatures and compute *other* similarity, which is when you imagine the whole SQFD matrix the upper right and lower left matrix. As SQFD is symmetric and the *other* similarity matrix does not have any items on the main diagonal, we compute only one of the halfs and then multiply it by 2.

We then add both self similarities with the *other* similarity, which gives us the distance squared from the given source signature to the current medoid. We compare it with minimal distance squared we have seen so far, cached in shared memory, and update the assignment, also cached in shared memory, if the new distance is smaller.

Once we go through all the medoids, we propagate the assignments to global memory from shared memory.

As we can only process a small number of signatures per block, limited by the size of shared memory and the signatures, and the number of blocks per kernel is limited, we have to split the processing into multiple kernels. This also helps with overlapping of the assignment transfer back to host memory.


After the rank computes assignments of all it's images, we use MPI Allgatherv to gather the assignments of data assigned to each rank to all ranks, so that we can use it to compute the scores.

After this exchange is finished, we start asynchronous transfer of the assignments to device memory.
If we used something like NCCL or MPI with CUDA support, we could have just exchanged the data straight from device memory.

## Cluster preprocessing

While the assignments are getting uploaded to device memory, we use the assignments to get some information about clusters.

This is a place that could use some work, as everything is computed sequentially with no parallelization, and Amdahl is going to kill us for that. This part is also why my solution scales badly with the number of clusters.

We create a simple index into the dataset. It consists of two arrays:

- clusterList, which holds image indexes ordered by their assignment. This means that first we have the index of images which belong to cluster 0, followed by indexes of images which belong to cluster 1 etc. Here index means an integer which tells us that the image is i-th in the dataset.

- clusterListIndex, which holds offset in the clusterList where the given cluster starts, + where the last cluster ends.

There is a third array, clusterComplexities, which is used to distribute work between ranks. It is basically cluster size squared, as each cluster takes this many operations to compute scores for, and then prefix sum of that.

We first compute size of each cluster into the clusterListIndex array. Then we use these sizes to compute clusterComplexitites.

Then we do prefix sum of both clusterListIndex and clusterComplexities, so that we get the values we need.

Lastly we compute the clusterList.

After that, we start uploads of clusterList and clusterListIndex to device memory.

## Scores

We first zero out scores from previous run, in parallel with data transfer of clusterList and clusterListIndex.

After that we start computing the scores. We try to give each rank roughly the same complexity of clusters to compute, so that the work is balanced.

We just take the total complexity and divide it equally between ranks and then search which cluster is the first that the current rank should compute.

THere are two types of clusters, small, where multiple clusters may be processed by the same kernel to lower the amount of kernels we need to start if there are many continuous small clusters. It would be even better if we ordered clusters by size, but that is not currently implemented.

Big clusters is a cluster that may be computed by multiple kernels. We need to compute distance of each image in the cluster to each other image in the cluster. Each image can be processed in parallel, but even computation of a single image can be split into multiple parallel tasks, where each task takes part of the whole cluster and computes the distance to the given image.

If first clsuter of for a rank falls into a small cluster, then it starts on the following cluster. If it falls into a big cluster, it computes the part of the big cluster.

The cutoff between small and big cluster is given in their complexity and is given as a command line parameter.

Runnig small cluster kernels is easy, basically the same as assignments. For big cluster, we have a problem that multiple clusters can be computing a value of a single image, which they synchronize using aotmic operations. We use events inserted after the kernels computing the same cluster and before the memory transfer to postpone the transfer after the cluster computation is finished.

During the computation, for each rank we track the start cluster, end cluster, start source, end source (where source is source image). We use these values later to exchange results between ranks.

After the score computation is finished, we use two Allgather calls to exchange data. First distributes which clusters were computed by each rank. We us asynchronous communication here, as we are computing the lowest score of the clusters that were processed by the current rank.

Once we have minimal scores for each of the clusters or parts of cluster which were processed by the current rank and we know the number of clusters processed by each rank, we can exchange this information with all other ranks using another Allgather call. This allows us to minimize the transfered information, as we only share two numbers for each cluster + at most 2 numbers for each rank. From this data, each rank computes the new medoids.

We have finished an iteration.

### Scores CUDA code

There are two CUDA functions for computing scores. One is for big clusters, one for small. They are very similiar, even with the Assignment function they are derived from.

They also copy number of source signatures to shared memory, precompute their self similarities and then go through a list of signatures, here rest of the cluster/part of the cluster, where they load each signature to shared memory, precompute self similarity and then go through source signatures in sequence, computing *other* similarity and adding them all to compute the score or partial score of the given source signature.

After we go through the whole target list, we atomicly update the scores in global memory.

For small clusters, the only difference is that we have some additional logic to go only through the source nodes which are from the same cluster as the target node. As the clusters are in increasing order, this is very easy, just moving two indexes along the source node array.

In both of these, the range of source signatures and target signatures they go through is given in terms of clusterList, where each cluster is represented as continuous sequence of indexes.

## Allocated memory

All host memory that is ever used for transfering data between host and device memory is allocated as pinned. This includes the following:

- two arrays for medoids (two allow us to compare if the current iteration changed anything)
- assignments
- scores
- clusterList
- clusterListIndex

the total size is (3\*number of images in dataset) + (3\* number of clusters), which is not that bad.