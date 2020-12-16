#ifndef KMEDOIDS_SHARED_RESULTS_HPP
#define KMEDOIDS_SHARED_RESULTS_HPP

#include "system/file.hpp"
#include "misc/exception.hpp"

#include <vector>
#include <cctype>

class KMedoidsResults
{
private:
	struct Header {
		std::uint64_t magic;
		std::uint64_t itemSize;
		std::uint64_t medoidsCount;
		std::uint64_t assignmentsCount;

		Header(std::size_t _medoidsCount = 0, std::size_t _assignmentsCount = 0)
			: magic(0xbee3bee3b001b001), itemSize(sizeof(std::size_t)), medoidsCount(_medoidsCount), assignmentsCount(_assignmentsCount) {}
	};

public:
	std::vector<std::size_t> mMedoids;
	std::vector<std::size_t> mAssignment;

	void load(const std::string& fileName)
	{
		bpp::File file(fileName);
		file.open("rb");

		Header header;
		std::uint64_t magic = header.magic;
		file.read(&header);
		if (header.magic != magic) {
			throw (bpp::RuntimeError() << "Given file " << fileName << " does not contain correct header.");
		}

		if (header.itemSize != sizeof(std::size_t)) {
			throw (bpp::RuntimeError() << "Given file " << fileName << " is saved on " << (header.itemSize*8)
				<< "bit machine but we are running " << (sizeof(std::size_t)*8) << "bit code.");
		}

		file.read(mMedoids, header.medoidsCount);
		file.read(mAssignment, header.assignmentsCount);

		file.close();
	}


	void save(const std::string &fileName) const
	{
		bpp::File file(fileName);
		file.open("wb");

		Header header(mMedoids.size(), mAssignment.size());
		file.write(&header);
		file.write(mMedoids);
		file.write(mAssignment);

		file.close();
	}
};

#endif