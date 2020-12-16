#define _CRT_SECURE_NO_WARNINGS

#include "kmedoids.hpp"
#include "results.hpp"

#include "signatures.hpp"
#include "sqfd.hpp"

#include "cli/args.hpp"
#include "system/stopwatch.hpp"

#include <vector>
#include <iostream>

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
void run(const bpp::ProgramArguments& args)
{
	using sqfd_t = BinaryDistanceFunctorSQFD<DIM, FLOAT, FLOAT>;
	using db_t = DBSignatureListMapped<DIM, FLOAT>;

	FLOAT alpha = (FLOAT)args.getArgFloat("alpha").getValue();
	std::size_t k = (std::size_t)args.getArgInt("k").getValue();
	std::size_t maxIters = (std::size_t)args.getArgInt("iterations").getValue();
	std::size_t seed = (std::size_t)args.getArgInt("seed").getValue();

	std::cout << "Openning database file " << args[0] << " ... ";
	db_t db(args[0]);
	std::size_t limit = db.size();
	std::cout << limit << " signatures found." << std::endl;

	if (args.getArgInt("images").isPresent()) {
		limit = std::min(limit, (std::size_t)args.getArgInt("images").getValue());
		db.setCount(limit);
	}
	std::cout << "Total " << limit << " image signatures will be used." << std::endl;

	std::cout << "Initializing k-medoids (k = " << k << ", alpha = " << alpha << ", iterations limit = " << maxIters << ") ..." << std::endl;
	sqfd_t sqfd(alpha);
	KMedoids<db_t, sqfd_t, FLOAT> kMedoids(sqfd, k, maxIters);
	KMedoidsResults results;

	std::cout << "Generating initial medoids set using seed " << seed << std::endl;
	generateRandomPermutation(results.mMedoids, limit, seed);
	results.mMedoids.resize(k);

	std::cout << "Executing k-medoids ... "; std::cout.flush();
	bpp::Stopwatch stopwatch(true);
	std::size_t iters = kMedoids.run(db, results.mMedoids, results.mAssignment);
	stopwatch.stop();
	std::cout << iters << " iterations" << std::endl;
	std::cout << "Execution wall time: " << stopwatch.getSeconds() << " seconds" << std::endl;

	std::cout << "Computed avg. distance: " << kMedoids.getLastAvgDistance() << std::endl;
	std::cout << "Computed avg. in-cluster distance: " << kMedoids.getLastAvgClusterDistance() << std::endl;

	printResultStats(results);

	if (args.getArg("save").isPresent()) {
		std::string fileName = args.getArgString("save").getValue();
		std::cout << "Saving results in " << fileName << " ..." << std::endl;
		results.save(fileName);
	}
}

int main(int argc, char* argv[])
{
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

		args.registerArg<bpp::ProgramArguments::ArgInt>("seed", "Seed for random generator (to make results deterministic)", false, 42);

		// Process the arguments ...
		args.process(argc, argv);
	}
	catch (bpp::ArgumentException& e) {
		std::cout << "Invalid arguments: " << e.what() << std::endl << std::endl;
		args.printUsage(std::cout);
		return 1;
	}

	try {
		run<7>(args);
	}
	catch (std::exception& e) {
		std::cout << "Error: " << e.what() << std::endl << std::endl;
		return 2;
	}

	return 0;
}
