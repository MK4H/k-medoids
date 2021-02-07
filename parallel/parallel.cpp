#define _CRT_SECURE_NO_WARNINGS

#include "results.hpp"

#include "signatures.hpp"
#include "helpers.hpp"

#include "cli/args.hpp"

#include <vector>
#include <iostream>
#include <random>

#include <mpi.h>

#include "consAlg.hpp"
#include "memAlg.hpp"

void printResultStats(const KMedoidsResults& results)
{
	std::vector<std::size_t> sizes(results.mMedoids.size());
	for (auto&& a : results.mAssignment) {
		++sizes[a];
	}

	for (std::size_t i = 0; i < results.mMedoids.size(); ++i) {
		std::cout << results.mMedoids[i] << " (" << sizes[i] << "), ";
	}
	std::cout << std::endl << std::endl;

	// for (auto&& a : results.mAssignment) {
	// 	std::cout << a << ", ";
	// }
	// std::cout << std::endl << std::endl;
}



template<int DIM, typename FLOAT = float>
void run(const bpp::ProgramArguments& args, int rank, int comm_size)
{
	static_assert(!std::is_same_v<FLOAT, float> || !std::is_same_v<FLOAT, double>, "Invalid FLOAT type");

	using db_t = DBSignatureListMapped<DIM, FLOAT>;

	FLOAT alpha = (FLOAT)args.getArgFloat("alpha").getValue();
	std::size_t k = (std::size_t)args.getArgInt("k").getValue();
	std::size_t maxIters = (std::size_t)args.getArgInt("iterations").getValue();
	std::size_t seed = (std::size_t)args.getArgInt("seed").getValue();
	std::size_t asgnBlockSize = (std::size_t)args.getArgInt("asgnBlockSize").getValue();
	std::size_t asgnSigPerBlock = (std::size_t)args.getArgInt("asgnSigPerBlock").getValue();
	std::size_t asgnBlocksPerKernel = (std::size_t)args.getArgInt("asgnBlocksPerKernel").getValue();
	std::size_t scoreBlockSize = (std::size_t)args.getArgInt("scoreBlockSize").getValue();
	std::size_t scoreSourcesPerBlock = (std::size_t)args.getArgInt("scoreSourcesPerBlock").getValue();
	std::size_t scoreTargetsPerBlock = (std::size_t)args.getArgInt("scoreTargetsPerBlock").getValue();
	std::size_t scoreSourceBlocksPerKernel = (std::size_t)args.getArgInt("scoreSourceBlocksPerKernel").getValue();
	std::size_t scoreTargetBlocksPerKernel = (std::size_t)args.getArgInt("scoreTargetBlocksPerKernel").getValue();
	std::size_t smallClusterComplexity = (std::size_t)args.getArgInt("smallClusterComplexity").getValue();
	std::size_t streams = (std::size_t)args.getArgInt("cudaStreams").getValue();

	std::cout << "Opening database file " << args[0] << " ... ";
	db_t db(args[0]);
	std::size_t limit = db.size();
	std::cout << limit << " signatures found." << std::endl;

	if (args.getArgInt("images").isPresent()) {
		limit = std::min(limit, (std::size_t)args.getArgInt("images").getValue());
		db.setCount(limit);
	}
	std::cout << "Total " << limit << " image signatures will be used." << std::endl;

	std::cout << "Initializing k-medoids (k = " << k << ", alpha = " << alpha << ", iterations limit = " << maxIters << ") ..." << std::endl;
	KMedoidsResults results;

	// TODO: Other algs
	MemCudaAlg<DIM, FLOAT> alg(
		rank,
		comm_size,
		limit,
		seed,
		db,
		alpha,
		k,
		asgnBlockSize,
		asgnSigPerBlock,
		asgnBlocksPerKernel,
		scoreBlockSize,
		scoreSourcesPerBlock,
		scoreTargetsPerBlock,
		scoreSourceBlocksPerKernel,
		scoreTargetBlocksPerKernel,
		smallClusterComplexity,
		streams
	);

	alg.initialize();

	for (std::size_t iter = 0; iter < maxIters; ++iter) {
		if (!alg.runIteration()) {
			break;
		}
	}

	alg.fillResults(results);

	//std::cout << "Computed avg. distance: " << kMedoids.getLastAvgDistance() << std::endl;
	//std::cout << "Computed avg. in-cluster distance: " << kMedoids.getLastAvgClusterDistance() << std::endl;
	printResultStats(results);

	// Save only on master
	if (rank == 0 && args.getArg("save").isPresent()) {
		std::string fileName = args.getArgString("save").getValue();
		std::cout << "Saving results in " << fileName << " ..." << std::endl;
		results.save(fileName);
	}

}

int main(int argc, char* argv[])
{
	int ret = 0;
	int provided;
	MPICH(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));

	int rank, size;
	MPICH(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
	MPICH(MPI_Comm_size(MPI_COMM_WORLD, &size));

	fdRedirecter redirect(std::string("rank") + std::to_string(rank));

	/*
	 * Arguments
	 */
	bpp::ProgramArguments args(1, 1);
	args.setNamelessCaption(0, "Input .bsf file");

	try {
		args.registerArg<bpp::ProgramArguments::ArgString>("save", "Path to a file to which the results are saved.", false);

		args.registerArg<bpp::ProgramArguments::ArgInt>("iterations", "Maximal number of iterations", false, 16, 0, 4096);
		args.registerArg<bpp::ProgramArguments::ArgInt>("k", "Number of clusters", false, 32, 0, 1024 * 1024);
		args.registerArg<bpp::ProgramArguments::ArgInt>("images", "Limit for number of images (only first N images from the input file are taken)", false, 1024, 0);
		args.registerArg<bpp::ProgramArguments::ArgFloat>("alpha", "Alpha tuning parameter for SQFD", false, 0.2, 0.0001, 100);

		args.registerArg<bpp::ProgramArguments::ArgInt>("asgnBlockSize", "Number of threads in CUDA block when computing assignment", false, 256, 128, 1024);
		args.registerArg<bpp::ProgramArguments::ArgInt>("asgnSigPerBlock", "Number of signatures processed by a single block during assignment computation, must fit into shared memory", false, 5, 1, 64);
		args.registerArg<bpp::ProgramArguments::ArgInt>("asgnBlocksPerKernel", "Number of blocks per kernel during assignment computation, determines data transfer overlapping and kernel execution parallelism", false, 65535, 1, 65535);
		// TODO: Calculate max number of signatures
		args.registerArg<bpp::ProgramArguments::ArgInt>("scoreBlockSize", "Number of threads in CUDA block when computing scores", false, 256, 128, 1024);
		args.registerArg<bpp::ProgramArguments::ArgInt>("scoreSourcesPerBlock", "Number of source signatures processed by a single block during score computation, must fit into shared memory", false, 5, 1, 64);
		args.registerArg<bpp::ProgramArguments::ArgInt>("scoreTargetsPerBlock", "Number of target signatures processed by a single block during score computation, determines how many blocks will be computing a single source block in parallel", false, 100, 1);
		args.registerArg<bpp::ProgramArguments::ArgInt>("scoreSourceBlocksPerKernel", "Number of blocks in the source signature dimension per kernel during score computation, determines data transfer overlapping and kernel execution parallelism", false, 65535, 1, 65535);
		args.registerArg<bpp::ProgramArguments::ArgInt>("scoreTargetBlocksPerKernel", "Number of blocks in the target signature dimension per kernel during score computation, largely determines global memory access pressure", false, 65535, 1, 65535);
		args.registerArg<bpp::ProgramArguments::ArgInt>("smallClusterComplexity", "Complexity under which block is considered small. Complexity is size of the block squared.", false, 1000000, 100);
		args.registerArg<bpp::ProgramArguments::ArgInt>("cudaStreams", "Number of cuda streams, determines maximum parallelism in GPU.", false, 4, 1);
		args.registerArg<bpp::ProgramArguments::ArgInt>("seed", "Seed for random generator (to make results deterministic)", false, 42);

		// Process the arguments ...
		args.process(argc, argv);
	}
	catch (bpp::ArgumentException& e) {
		std::cout << "Invalid arguments: " << e.what() << std::endl << std::endl;
		args.printUsage(std::cout);
		ret = 1;
		goto end;
	}

	try {
		run<7>(args, rank, size);
	}
	catch (std::exception& e) {
		std::cout << "Error: " << e.what() << std::endl << std::endl;
		ret = 2;
		goto end;
	}

end:
	MPICH(MPI_Finalize());
	return ret;
}
