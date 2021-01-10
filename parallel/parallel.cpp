#define _CRT_SECURE_NO_WARNINGS

#include "results.hpp"

#include "signatures.hpp"
#include "helpers.hpp"

#include "cli/args.hpp"

#include <vector>
#include <iostream>
#include <random>

#include <mpi.h>

#include "work.hpp"

void generateRandomPermutation(std::vector<std::size_t> &res, std::size_t count, std::size_t seed) {
	res.resize(count);
	for (std::size_t i = 0; i < count; ++i) {
		res[i] = i;
	}

	std::mt19937 gen((unsigned int)seed);
	std::shuffle(res.begin(), res.end(), gen);
}

void printResultStats(const KMedoidsResults& results)
{
	std::vector<std::size_t> sizes(results.mMedoids.size());
	for (auto&& a : results.mAssignment) {
		++sizes[a];
	}

	for (std::size_t i = 0; i < results.mMedoids.size(); ++i) {
		std::cout << results.mMedoids[i] << " (" << sizes[i] << "), ";
	}
	std::cout << std::endl;
}

template<int DIM, typename FLOAT = float>
void run(const bpp::ProgramArguments& args, int rank, int comm_size)
{
	using db_t = DBSignatureListMapped<DIM, FLOAT>;

	FLOAT alpha = (FLOAT)args.getArgFloat("alpha").getValue();
	std::size_t k = (std::size_t)args.getArgInt("k").getValue();
	std::size_t maxIters = (std::size_t)args.getArgInt("iterations").getValue();
	std::size_t seed = (std::size_t)args.getArgInt("seed").getValue();
	std::size_t blockSize = (std::size_t)args.getArgInt("blockSize").getValue();
	std::size_t sigPerBlock = (std::size_t)args.getArgInt("sigPerBlock").getValue();
	std::size_t blocksPerKernel = (std::size_t)args.getArgInt("blocksPerKernel").getValue();
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

	if (rank == 0) {
		generateRandomPermutation(results.mMedoids, limit, seed);
	}

	results.mMedoids.resize(k);
	// Broadcast the initial medoids
	MPICH(MPI_Bcast(results.mMedoids.data(), k, MPI_UINT64_T, 0, MPI_COMM_WORLD));

	work(rank, comm_size, alpha, k, maxIters, limit, blockSize, sigPerBlock, blocksPerKernel, streams, db, results);

	//std::cout << "Computed avg. distance: " << kMedoids.getLastAvgDistance() << std::endl;
	//std::cout << "Computed avg. in-cluster distance: " << kMedoids.getLastAvgClusterDistance() << std::endl;

	printResultStats(results);

	if (args.getArg("save").isPresent()) {
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

		args.registerArg<bpp::ProgramArguments::ArgInt>("blockSize", "Number of threads in CUDA block", false, 256, 128, 1024);
		// TODO: Calculate max number of signatures
		args.registerArg<bpp::ProgramArguments::ArgInt>("sigPerBlock", "Number of signatures processed by a single block, must fit into shared memory", false, 10, 1, 64);
		args.registerArg<bpp::ProgramArguments::ArgInt>("blocksPerKernel", "Number of blocks per kernel, determines data transfer overlapping and kernel execution parallelism", false, 1024, 1, 65535);
		args.registerArg<bpp::ProgramArguments::ArgInt>("cudaStreams", "Number of cuda streams, determines maximum parallelism in GPU.", false, 4, 1, std::numeric_limits<bpp::ProgramArguments::ArgInt::value_t>::max());
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
		int rank, size;
		MPICH(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
		MPICH(MPI_Comm_size(MPI_COMM_WORLD, &size));

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
