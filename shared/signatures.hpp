#ifndef KMEDOIDS_SHARED_SIGNATURES_HPP
#define KMEDOIDS_SHARED_SIGNATURES_HPP

#include <points.hpp>
#include <commons.hpp>
#include <system/mmap_file.hpp>

#include <algorithm>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <stdexcept>

#include <cassert>
#include <cstdint>
#include <cstdio>



// Forward Declaration
template<int DIM, typename NUM_TYPE> class DBSignatureList;


/**
 * \brief Wrapper class for internal signature representation.
 * \tparam DIM Signature coordinates dimension.
 * \tparam NUM_TYPE Data type of the numbers stored in signatures (float or double).
 *
 * This class works like a smart read-only pointer to a signature.
 * It is quite lighweight and it does not allocate any resources.
 */
template<int DIM, typename NUM_TYPE = float> class DBSignature
{
friend class DBSignatureList<DIM, NUM_TYPE>;
private:
	const NUM_TYPE *mData;	///< Block of #length weights + #length*DIM coordinates
	std::size_t mLength;		///< Number of centroids.

	std::size_t mId;			///< Signature ID within the database.

	/**
	 * \brief The constructor is private so it can be invoked only from friend classes.
	 * \param data Initialization value for mData member.
	 * \param length Initialization value for mLength member.
	 */
	DBSignature(const NUM_TYPE *data, std::size_t length, std::size_t id)
		: mData(data), mLength(length), mId(id) {}

public:
	// Enforce virtual destructor for descendants...
	virtual ~DBSignature() {}


	/**
	 * \brief Empty constructor creates invalid signature.
	 */
	DBSignature() : mData(NULL), mLength(0), mId(0) {}

	/**
	 * \brief Copy constructor provides the only way how to work with signatures
	 *		outside the database classes.
	 */
	DBSignature(const DBSignature &sig) : mData(sig.mData), mLength(sig.mLength), mId(sig.mId) {}


	/**
	 * \brief Return the ID of the signature.
	 */
	std::size_t getId() const
	{
		return mId;
	}


	/**
	 * \brief Return number of centroids in the signature.
	 */
	std::size_t getCentroidCount() const
	{
		return mLength;
	}


	/**
	 * \brief Return total number of NUM_TYPE entries in the raw data block (weights+coordinates).
	 */
	std::size_t getRawLength() const
	{
		return (DIM+1) * mLength;
	}


	/**
	 * \brief Pointer to raw data block. The block starts with weights,
	 *		so its also pointer to weights block.
	 */
	const NUM_TYPE* getRawData() const		{ return mData; }

	/**
	 * \brief Alias for getRawData().
	 */
	const NUM_TYPE* getWeights() const		{ return getRawData(); }
	
	/**
	 * \brief Return pointer to selected centroid coordinates.
	 * \param i Index of the centroid (from 0 to length-1).
	 * \note Coordinates are in a continous range. Address of the first coordinates
	 *		(index 0) is also address of the whole coordinates block.
	 * \return Pointer to the continous array of DIM coodinates.
	 */
	const NUM_TYPE* getCoordinates(std::size_t i = 0) const
	{
		assert(i < mLength);
		return mData + mLength + i*DIM;
	}
};





/**
 * \brief Core implementation of signature database.
 * \tparam DIM Signature coordinates dimension.
 * \tparam NUM_TYPE Data type of the numbers stored in signatures (float or double).
 */
template<int DIM, typename NUM_TYPE = float> class DBSignatureList
{
protected:
	const NUM_TYPE *mData;	///< Pointer to a block, where raw data are stored.

	/** 
	 * \brief Index over the data. It holds prefix scan of signature lengths.
	 * \note The lengts are in fact numbers of centroids.
	 */
	const db_offset_t *mIndex;

	std::size_t mCount;	///< Number of signatures in the database.

	std::size_t mMinSignatureLength;	///< Length of the smallest signature in db.
	std::size_t mMaxSignatureLength;	///< Length of the largest signature in db.


	/**
	 * \brief Constructor is declared protected, so it can be published by descendant classes.
	 */
	DBSignatureList() : mData(NULL), mIndex(NULL), mCount(0),
		mMinSignatureLength(~(std::size_t)0), mMaxSignatureLength(0) {}

	/**
	 * \brief Trivial shallow copy constructor.
	 */
	DBSignatureList(const DBSignatureList &db) : mData(db.mData), mIndex(db.mIndex), mCount(db.mCount),
		mMinSignatureLength(db.mMinSignatureLength), mMaxSignatureLength(db.mMaxSignatureLength) {}

public:
	// Enforce virtual destructor for descendants.
	virtual ~DBSignatureList() {}

	/**
	 * \brief Return the number of signatures in the database.
	 */
	std::size_t size() const { return mCount; }


	/**
	 * \brief Fetch selected signature.
	 * \param i Index of the signature (from 0 to count-1).
	 * \return The DBSignature wrapper object.
	 */
	DBSignature<DIM, NUM_TYPE> operator[](std::size_t i) const
	{
		assert(i < mCount);
		db_offset_t offset = (i > 0) ? mIndex[i-1] : 0;
		return DBSignature<DIM, NUM_TYPE>(mData + offset*(DIM+1), static_cast<std::size_t>(mIndex[i]-offset), i);
	}


	/**
	 * \brief Return the number of centroids of the smallest signature in the list.
	 */
	std::size_t getMinSignatureLength() const { return mMinSignatureLength; }

	/**
	 * \brief Return the number of centroids of the largest signature in the list.
	 */
	std::size_t getMaxSignatureLength() const { return mMaxSignatureLength; }


	/**
	 * \brief Attempt to populate memory pages containing signatures from selected range.
	 * \param formIdx Index of the first signature in the range to be populated.
	 * \param toIdx Index of the first signature after the range to be populated.
	 */
	void populate(std::size_t fromIdx = 0, std::size_t toIdx = ~((std::size_t)0))
	{
		if (toIdx > mCount) toIdx = mCount;
		if (fromIdx >= toIdx) return;
		NUM_TYPE x = (NUM_TYPE)0.0;
		while (fromIdx < toIdx) {
			DBSignature<DIM, NUM_TYPE> sig = (*this)[fromIdx];
			x += sig.getWeights()[0];
			++fromIdx;
		}
		if (x <= 0.0f)
			throw (DBException() << "Something is odd!");
	}
};





/**
 * \brief DBSignatureList that saves signatures in self-allocated memory.
 *		It can load signatures from text file or build it by adding signatures.
 * \tparam DIM Signature coordinates dimension.
 * \tparam NUM_TYPE Data type of the numbers stored in signatures (float or double).
 */
template<int DIM, typename NUM_TYPE = float> class DBSignatureListAlloc : public DBSignatureList<DIM, NUM_TYPE>
{
private:
	std::vector<NUM_TYPE> mDataVec;		///< Tmp vector used for loading. The mData pointer points inside this vector.
	std::vector<db_offset_t> mIndexVec;	///< Tmp vector used for loading. The mIndex pointer points inside this vector.


	/**
	 * \brief Loads one signature data from text file into internal vectors.
	 * \param file Opened stream from which the data are read.
	 * \param size Length of the signature (number of centroids).
	 */
	void loadSignature(std::ifstream &file, std::size_t size)
	{
		// Update known signature length limits.
		this->mMinSignatureLength = std::min<std::size_t>(this->mMinSignatureLength, size);
		this->mMaxSignatureLength = std::max<std::size_t>(this->mMaxSignatureLength, size);

		// Update index (incremental computation of prefix scan).
		mIndexVec.push_back(static_cast<db_offset_t>(size) + (!mIndexVec.empty() ? mIndexVec.back() : 0));

		// Resize data (weights) and prepare coordinates vectors.
		std::size_t weightIdx = mDataVec.size();
		std::size_t coordsIdx = weightIdx + size;
		mDataVec.resize(mDataVec.size() + size*(DIM+1));

		// Load numbers and scatter them to proper places.
		while (size > 0) {
			file >> mDataVec[weightIdx++];
			for (std::size_t i = 0; i < DIM; ++i)
				file >> mDataVec[coordsIdx++];
			--size;
		}
	}

public:
	/**
	 * \brief Create empty signature list.
	 * \param reserve Size of the preallocated (reserved) memory (in number of signatures).
	 */
	DBSignatureListAlloc(std::size_t reserve = 0)
		: DBSignatureList<DIM, NUM_TYPE>()
	{}


	/**
	 * \brief Load signature list from a text file.
	 * \param fileName Path to database SVF file.
	 * \param limitSize Maximal number of centroids being loaded.
	 */
	DBSignatureListAlloc(const std::string &fileName, std::size_t limitSize = ~(std::size_t)0)
		: DBSignatureList<DIM, NUM_TYPE>()
	{
		// Try to open the file...
		std::ifstream file;
		file.open(fileName, std::ios_base::in);
		if (file.fail())
			throw DBException("Cannot open database file.");
		
		// Read the file contents into raw vectors.
		std::string buf;
		while (!file.eof() && !file.fail()) {
			// Skip irrelevant lines ...
			if (file.peek() < '0' || file.peek() > '9') {
				std::getline(file, buf);
				continue;
			}

			// Read leading object parameters.
			std::size_t size, dim;
			file >> size >> dim;
			if (dim != DIM)
				throw DBException("Signature dimesion mismatch!");

			// Load one signature.
			loadSignature(file, size);
			++this->mCount;

			if (this->mCount % 5000 == 0)	// raw progress reporting
				std::cerr << this->mCount << std::endl;

			// Terminate if the row limit has been reached.
			if (this->mCount >= limitSize) break;

			// Skip rest of the line if necessary.
			std::getline(file, buf);
		}

		file.close();

		// Update main pointers.
		this->mData = &(mDataVec[0]);
		this->mIndex = &(mIndexVec[0]);
	}


	/**
	 * \brief Convert weighted point into signature and add it into the list.
	 * \tparam WEIGHT_TYPE Type of the weights used in the weights points structure.
	 * \param points Weighted points being converted into signature.
	 */
	template<typename WEIGHT_TYPE>
	void addSignature(const PointsWithWeights<DIM, NUM_TYPE, WEIGHT_TYPE> &points)
	{
		if (points.size() == 0)
			throw (DBException() << "Unable to add empty signature at position " << this->mCount+1 << ".");

		// Update known signature length limits.
		this->mMinSignatureLength = std::min<std::size_t>(this->mMinSignatureLength, static_cast<std::size_t>(points.size()));
		this->mMaxSignatureLength = std::max<std::size_t>(this->mMaxSignatureLength, static_cast<std::size_t>(points.size()));

		mIndexVec.push_back(static_cast<db_offset_t>(points.size()) + (!mIndexVec.empty() ? mIndexVec.back() : 0));

		// Resize data (weights) and prepare coordinates vectors.
		std::size_t weightIdx = mDataVec.size();
		std::size_t coordsIdx = weightIdx + points.size();
		mDataVec.resize(mDataVec.size() + points.size()*(DIM+1));

		// Load numbers and scatter them to proper places.
		for (std::size_t i = 0; i < points.size(); ++i) {
			mDataVec[weightIdx + i] = static_cast<NUM_TYPE>(points.weight(i));
			for (std::size_t d = 0; d < DIM; ++d)
				mDataVec[coordsIdx++] = points.get(i, d);
		}

		// The pointers must be updates since the vectors may have relocated.
		this->mData = &(mDataVec[0]);
		this->mIndex = &(mIndexVec[0]);
		
		++this->mCount;
	}
};





/**
 * \brief DBSignatureList mapped to memory from binary file. This list is immutable.
 * \tparam DIM Signature coordinates dimension.
 * \tparam NUM_TYPE Data type of the numbers stored in signatures (float or double).
 * \tparam NUM_TYPE Data type of the numbers stored in signatures (float or double).
 */
template<int DIM, typename NUM_TYPE = float> class DBSignatureListMapped : public DBSignatureList<DIM, NUM_TYPE>
{
private:
	bpp::MMapFile mFile;		///< Memory mapped file representing the database.

	std::size_t mExactCount;	///< Actual number of signatures in the file.

	/**
	 * \brief Internal descriptor structure used for binary database file format.
	 *
	 * This structure represents the first 16B of binary database file.
	 */
	struct DBFileDescriptor
	{
	private:
		std::uint32_t	mMagic;			///< Magic value for verification.
		std::uint16_t	mDim;			///< Centroid coordinates dimension.
		std::uint16_t	mTypeLength;	///< Size of NUM_TYPE (in bytes).
		std::uint32_t	mSignatures;	///< Number of signatures in database.
		std::uint16_t	mMinSigLen;		///< Length of the smallest signature in db.
		std::uint16_t	mMaxSigLen;		///< Length of the largest signature in db.

		/**
		 * \brief Provide hardwired magic number for the header verification.
		 */
		static std::uint32_t getValidMagic()
		{
			return 0x1beddeed;	// figure that one out :-)
		}


	public:
		/**
		 * \brief Create and fill DB file descriptor.
		 *
		 * This method is used only when database is being created/modified.
		 * When the database is loaded, the constructor is not called.
		 */
		DBFileDescriptor(std::size_t signatures, std::size_t minSigLen, std::size_t maxSigLen)
			: mMagic(getValidMagic()), mDim(DIM), mTypeLength(sizeof(NUM_TYPE)),
			mSignatures((std::uint32_t)signatures), mMinSigLen((std::uint16_t)minSigLen), mMaxSigLen((std::uint16_t)maxSigLen)
		{}

		/**
		 * \brief Verify that the header has correct magic number, dimension, and type length.
		 * \return True if the header is ok, false otherwise.
		 */
		bool validate() const
		{
			return (this->mMagic == getValidMagic())
				&& (this->mDim == DIM)
				&& (this->mTypeLength == sizeof(NUM_TYPE));
		}

		std::size_t getSignatures() const	{ return mSignatures; }
		std::size_t getMinSigLen() const	{ return mMinSigLen; }
		std::size_t getMaxSigLen() const	{ return mMaxSigLen; }
	};


public:
	/**
	 * \brief Load a binary file as a database. The file is in fact memory mapped.
	 */
	DBSignatureListMapped(const std::string &fileName)
	{
		mFile.open(fileName.c_str());

		const DBFileDescriptor *descriptor = (const DBFileDescriptor*)mFile.getData();
		if (!descriptor->validate())
			throw DBException("Provided file is not valid database binary file or it has wrong dimension/numeric type.");

		// Copy metadata from file headers.
		this->mExactCount = this->mCount = descriptor->getSignatures();
		this->mMinSignatureLength = descriptor->getMinSigLen();
		this->mMaxSignatureLength = descriptor->getMaxSigLen();

		// Initialize data and index pointers.
		const char *filePtr = (const char *)mFile.getData();
	   	this->mIndex = (const db_offset_t*)(filePtr + sizeof(DBFileDescriptor));
		this->mData = (const NUM_TYPE*)(this->mIndex + this->mCount);
	}

	/**
	 * Modify (limit) the actual number of signatures in this container.
	 * \param count new value returned by size()
	 */
	void setCount(std::size_t count)
	{
		if (count > mExactCount) {
			throw (bpp::LogicError() << "Given count " << count << " is greater than the number of signatures in the file " << this->mExactCount << ".");
		}
		this->mCount = count;
	}

	/**
	 * Reset the actual number of signatures to encompass the entire file.
	 */
	void resetCount()
	{
		this->mCount = this->mExactCount;
	}
};





/**
 * \brief Builder class that constructs binary signature list.
 *		The binary signature list can be used by DBSignatureListMapped.
 * \tparam DIM Signature coordinates dimension.
 * \tparam NUM_TYPE Data type of the numbers stored in signatures (float or double).
 */
template<int DIM, typename NUM_TYPE = float> class DBSignatureListBinBuilder
{
private:
	/**
	 * \brief Internal descriptor structure used for binary database file format.
	 *
	 * This structure represents the first 16B of binary database file.
	 */
	struct DBFileDescriptor
	{
	private:
		std::size_t		mMagic;			///< Magic value for verification.
		std::uint16_t	mDim;			///< Centroid coordinates dimension.
		std::uint16_t	mTypeLength;	///< Size of NUM_TYPE (in bytes).
		std::size_t		mSignatures;	///< Number of signatures in database.
		std::uint16_t	mMinSigLen;		///< Length of the smallest signature in db.
		std::uint16_t	mMaxSigLen;		///< Length of the largest signature in db.

		/**
		 * \brief Provide hardwired magic number for the header verification.
		 */
		static std::size_t getValidMagic()
		{
			return 0x1beddeed;	// figure that one out :-)
		}


	public:
		/**
		 * \brief Create and fill DB file descriptor.
		 *
		 * This method is used only when database is being created/modified.
		 * When the database is loaded, the constructor is not called.
		 */
		DBFileDescriptor(std::size_t signatures = 0, std::size_t minSigLen = 0, std::size_t maxSigLen = 0)
			: mMagic(getValidMagic()), mDim(DIM), mTypeLength(sizeof(NUM_TYPE)),
			mSignatures(signatures), mMinSigLen((std::uint16_t)minSigLen), mMaxSigLen((std::uint16_t)maxSigLen)
		{}

		/**
		 * \brief Verify that the header has correct magic number, dimension, and type length.
		 * \return True if the header is ok, false otherwise.
		 */
		bool validate() const
		{
			return (this->mMagic == getValidMagic())
				&& (this->mDim == DIM)
				&& (this->mTypeLength == sizeof(NUM_TYPE));
		}

		std::size_t getSignatures() const	{ return mSignatures; }
		std::size_t getMinSigLen() const	{ return mMinSigLen; }
		std::size_t getMaxSigLen() const	{ return mMaxSigLen; }

		/**
		 * \brief Update the header after new signature is registered in DB.
		 * \param length Number of centroids in the new signature.
		 */
		void registerNewSignature(std::size_t length)
		{
			if (mSignatures == 0) {
				mMinSigLen = length;
				mMaxSigLen = length;
			}
			else {
				mMinSigLen = std::min<std::size_t>(mMinSigLen, length);
				mMaxSigLen = std::max<std::size_t>(mMaxSigLen, length);
			}
			++mSignatures;
		}
	};


	/**
	 * \brief Finalize and close the file. 
	 */
	void close()
	{
		if (mFp == NULL) return;
		if (mIndex.size() != static_cast<std::size_t>(mFinalSize))
			throw DBException("Unable to close file, that has not been filled yet.");

		// Finalize the file.
		std::fseek(mFp, 0, SEEK_SET);	// Seek back to file beginning.
		if (std::fwrite(&mHeader, sizeof(DBFileDescriptor), 1, mFp) != 1)
			throw DBException("Cannot save db file descriptor into a file.");
		
		// Write the database index.
		if (std::fwrite(&(mIndex[0]), sizeof(db_offset_t), mIndex.size(), mFp) != mIndex.size())
			throw DBException("Cannot save index into a file.");

		// Close and release resources.
		fclose(mFp);
		mFp = NULL;
		mIndex.clear();
	}


	std::FILE *mFp;						///< Handle to the underlying binary file.
	std::size_t mFinalSize;				///< Expected number of signatures.
	DBFileDescriptor mHeader;			///< File header structure.
	std::vector<db_offset_t> mIndex;	///< Offset index for the file.

public:
	/**
	 * \brief Create the builder object and open underlying output file.
	 * \param fileName Path to a file where the data will be written.
	 * \param finalSize Number of signatures that will be written to the file.
	 * \note The exact number of signatures must be known in advance.
	 */
	DBSignatureListBinBuilder(const std::string &fileName, std::size_t finalSize)
		: mFp(NULL), mFinalSize(finalSize)
	{
		mFp = std::fopen(fileName.c_str(), "wb");
		if (mFp == NULL)
			throw (DBException() << "Unable to open new signature file '" << fileName << "'.");

		std::fseek(mFp, sizeof(DBFileDescriptor) + finalSize * sizeof(db_offset_t), SEEK_SET);
	}


	/**
	 * \brief Add another signature to the file.
	 * \param sig Signature to be written into the constructed file.
	 * \note The file is automatically closed when final size is reached.
	 *		Attempts to add signatures after the file has been closed causes
	 *		an exception to be thrown.
	 */
	void add(const DBSignature<DIM, NUM_TYPE> &sig)
	{
		if (mFp == NULL)
			throw (DBException() << "Unable to write into already closed file.");

		// Write the signature data directly to opened file.
		if (std::fwrite(sig.getRawData(), sizeof(NUM_TYPE), sig.getRawLength(), mFp) != sig.getRawLength())
			throw DBException("Cannot save signature data into a file.");

		// Add the meta-info into header and index structures.
		mHeader.registerNewSignature(sig.getCentroidCount());
		mIndex.push_back((mIndex.empty() ? 0 : mIndex.back()) + sig.getCentroidCount());

		if (mIndex.size() == static_cast<std::size_t>(mFinalSize))
			close();
	}


	/**
	 * \brief Add list of signatures.
	 * \param List of signatures being added to constructed file.
	 * \note The file is automatically closed when final size is reached.
	 *		Attempts to add signatures after the file has been closed causes
	 *		an exception to be thrown.
	 */
	void add(const DBSignatureList<DIM, NUM_TYPE> &signatures)
	{
		for (std::size_t i = 0; i < signatures.size(); ++i) {
			add(signatures[i]);
		}
	}
};




/**
 * \brief Builder class that constructs text (SVF) signature list file.
 * \tparam DIM Signature coordinates dimension.
 * \tparam NUM_TYPE Data type of the numbers stored in signatures (float or double).
 */
template<int DIM, typename NUM_TYPE = float> class DBSignatureListTextBuilder
{
private:
	std::FILE *mFp;						///< Handle to the underlying binary file.

public:
	/**
	 * \brief Create the builder object and open underlying output file.
	 * \param fileName Path to a file where the data will be written.
	 */
	DBSignatureListTextBuilder(const std::string &fileName)
		: mFp(NULL)
	{
		mFp = std::fopen(fileName.c_str(), "w");
		if (mFp == NULL)
			throw (DBException() << "Unable to open new signature file '" << fileName << "'.");
	}


	~DBSignatureListTextBuilder()
	{
		close();
	}


	/**
	 * \brief Add another signature to the file.
	 * \param sig Signature to be written into the constructed file.
	 */
	void add(const DBSignature<DIM, NUM_TYPE> &sig)
	{
		if (mFp == NULL)
			throw (DBException() << "Unable to write into already closed file.");

		std::fprintf(mFp, "%d %d", sig.getCentroidCount(), DIM);

		for (std::size_t i = 0; i < sig.getCentroidCount(); ++i) {
			fprintf(mFp, " %g", sig.getWeights()[i]);
			for (std::size_t d = 0; d < DIM; ++d)
				fprintf(mFp, " %g", sig.getCoordinates(i)[d]);
		}

		std::fprintf(mFp, "\n");
	}


	/**
	 * \brief Add list of signatures.
	 * \param List of signatures being added to constructed file.
	 */
	void add(const DBSignatureList<DIM, NUM_TYPE> &signatures)
	{
		for (std::size_t i = 0; i < signatures.size(); ++i) {
			add(signatures[i]);
		}
	}


	/**
	 * \brief Finalize and close the file. 
	 */
	void close()
	{
		if (mFp == NULL) return;
		fclose(mFp);
		mFp = NULL;
	}
};
#endif
