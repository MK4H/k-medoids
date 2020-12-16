#ifndef KMEDOIDS_SHARED_KMEDOIDS_HPP
#define KMEDOIDS_SHARED_KMEDOIDS_HPP

#include "misc/exception.hpp"

#include <vector>
#include <algorithm>
#include <random>

/**
 * \brief A K-medoids algorithm.
 * \tparam OBJ_CONTAINER Container holding all database objects. Must implement [] and size() like std::vector.
 * \tparam DIST Type of distance functor that computes/provides distances between objects.
 *		The functor must have operator() that takes two OBJs (items from OBJ_CONTAINER) and yields a FLOAT.
 * \tparam FLOAT Float data type with selected precision (used for distances and medoid scores).
 *		It should be set to float or double.
 */
template<class OBJ_CONTAINER, class DIST, typename FLOAT = float>
class KMedoids
{
protected:
	DIST& mDistFnc;					///< Functor used to compute object distances (OBJ, OBJ) to FLOAT
	std::size_t mK;					///< Number of desired clusters
	std::size_t mMaxIters;			///< Maximal number of algorithm iterations.
	std::vector<FLOAT> mBestScores;	///< Computed cluster scores from the last update.


	// Distance statistics
	FLOAT mLastAvgDistance;				///< Average distance between an object and its respective medoid (computed with last update).
	FLOAT mLastAvgClusterDistance;		///< Average distance of average distances within each cluster (computed with last update).


	/**
	 * \brief Verify algorithm parameters. If they do conflict, LogicError is thrown.
	 */
	virtual void checkParams(std::size_t objectsCount)
	{
		if (mK < 2)
			throw (bpp::LogicError() << "Too few clusters (" << mK << ") selected. The k value must be at least 2.");
		if (mMaxIters == 0)
			throw (bpp::LogicError() << "At least one iteration must be allowed in the algorithm.");
		if (mK > objectsCount)
			throw (bpp::LogicError() << "The algorithm is requested to create " << mK << " clusters of " << objectsCount << " objects.");
	}


	/**
	 * \brief Create the initial selection of medoids (uniform random), unless they are already set.
	 */
	void initRandMedoids(std::vector<std::size_t>& medoids, std::size_t objectCount) const
	{
		if (medoids.size() == mK) {
			// medoids are already set, just check them out
			std::sort(medoids.begin(), medoids.end());
			for (std::size_t i = 0; i < mK; ++i) {
				if (medoids[i] >= objectCount) {
					throw (bpp::RuntimeError() << "Initial medoid indices are out of range.");
				}
				if (i > 0 && medoids[i] == medoids[i - 1]) {
					throw (bpp::RuntimeError() << "Initial medoid indices contain duplicates.");
				}
			}
		}
		else {
			// initialize mediods with random seletion
			medoids.resize(objectCount);
			for (std::size_t i = 0; i < objectCount; ++i) {
				medoids[i] = i;
			}

			std::random_device rd;
			std::mt19937 gen(rd());
			std::shuffle(medoids.begin(), medoids.end(), gen);
		}
	}


	/**
	 * \brief Compute current assignment of objects to medoids according to distance functor.
	 * \param objects Input set of objects to be clustered.
	 * \param medoids The medoid objects (as indices to the objects vector).
	 * \param assignments The result assignment value for each object. The assignment vector
	 *		has the same size as objects vector and each value is an index to medoids vector.
	 * \param distances Pointer to an array, where distances to assigned medoids are kept.
	 *		If nullptr, the distances are not saved.
	 */
	void computeAssignments(const OBJ_CONTAINER& objects, std::size_t objectCount, std::vector<std::size_t>& medoids,
		std::vector<std::size_t>& assignments, FLOAT* distances = nullptr) const
	{
		assignments.resize(objectCount);

		// Compute assignment for each object individually.
		for (std::size_t i = 0; i < objectCount; ++i) {
			std::size_t asgn = 0;
			FLOAT minDist = mDistFnc(objects[i], objects[medoids[asgn]]);

			// Find nearest medoid...
			for (std::size_t m = 1; m < medoids.size(); ++m) {
				FLOAT dist = mDistFnc(objects[i], objects[medoids[m]]);
				if (dist < minDist) {
					minDist = dist;
					asgn = m;
					if (i == medoids[asgn]) break;	// Break if the object is the medoid.
				}
			}

			// Save the assignment.
			assignments[i] = asgn;
			if (distances != nullptr) {
				distances[i] = minDist;
			}
		}
	}


	/**
	 * \brief Compute the score of selected medoid within given cluster as a sum of distance squares.
	 * \param objects Input set of objects to be clustered.
	 * \param cluster List of object indices within the examined cluster.
	 * \param medoid Index of selected medoid in cluster vector.
	 * \return The score of the selected medoid.
	 */
	FLOAT computeMedoidScore(const OBJ_CONTAINER& objects, const std::vector<std::size_t>& cluster, std::size_t medoid) const
	{
		FLOAT score = 0.0;
		for (std::size_t i = 0; i < cluster.size(); ++i) {
			if (i == medoid) continue;
			FLOAT dist = mDistFnc(objects[cluster[medoid]], objects[cluster[i]]);
			score += dist * dist;
		}
		return score;
	}


	/**
	 * \brief Find the best medoid for selected cluster and return its object index.
	 * \param objects Input set of objects to be clustered.
	 * \param cluster List of object indices within the examined cluster.
	 * \param bestScore Output value where best distance sum of returned medoid is stored.
	 * \return Index of the best medoid found (in the objects collection).
	 */
	std::size_t getBestMedoid(const OBJ_CONTAINER& objects, const std::vector<std::size_t>& cluster, FLOAT& bestScore) const
	{
		// The cluster is never empty!
		if (cluster.empty()) {
			throw (bpp::RuntimeError() << "Unable to select the best medoid of an empty cluster.");
		}

		// One-medoid show and zwei-medoid buntes are easy to solve.
		if (cluster.size() < 3) {
			return cluster[0];
		}

		// Find a medoid with smallest score.
		std::size_t medoid = 0;
		bestScore = computeMedoidScore(objects, cluster, medoid);

		for (std::size_t i = 1; i < cluster.size(); ++i) {
			FLOAT score = computeMedoidScore(objects, cluster, i);
			if (score < bestScore) {
				bestScore = score;
				medoid = i;
			}
		}

		return cluster[medoid];
	}


	/**
	 * \brief Update the medoids according to the new assignments.
	 * \param objects Input set of objects to be clustered.
	 * \param medoids The medoid objects (as indices to the objects vector).
	 * \param assignments The assignment value for each object. The assignment vector
	 *		has the same size as objects vector and each value is an index to medoids vector.
	 * \return True if the medoids vector has been modified, false otherwise.
	 *		If the medoids have not been modified, the algorithm has reached a stable state.
	 */
	bool updateMedoids(const OBJ_CONTAINER& objects, std::vector<std::size_t>& medoids, const std::vector<std::size_t>& assignments)
	{
		// Construct a cluster index (vector of clusters, each cluster is a vector of object indices).
		std::vector< std::vector<std::size_t> > clusters;
		clusters.resize(medoids.size()); // one cluster per medoid

		// Spread objects into clusters based on the assignments vector.
		for (std::size_t i = 0; i < assignments.size(); ++i) {
			clusters[assignments[i]].push_back(i);
		}

		// Compute changes (new scores and medoids).
		mLastAvgDistance = mLastAvgClusterDistance = (FLOAT)0.0;

		std::size_t clusterCount = clusters.size();
		mBestScores.resize(clusterCount);
		std::vector<std::size_t> newMedoids(clusterCount);

		// Find the best medoid for each cluster.
		for (std::size_t m = 0; m < clusterCount; ++m) {
			mBestScores[m] = (FLOAT)0.0;
			newMedoids[m] = getBestMedoid(objects, clusters[m], mBestScores[m]);
		}

		bool changed = false;	// whether the medoids vector was modified
		for (std::size_t m = 0; m < clusters.size(); ++m) {
			mLastAvgDistance += mBestScores[m];
			if (clusters[m].size() > 1) {
				mLastAvgClusterDistance += mBestScores[m] / (FLOAT)(clusters[m].size() - 1);
			}

			changed = changed || (newMedoids[m] != medoids[m]);
			medoids[m] = newMedoids[m];
		}

		mLastAvgDistance /= (FLOAT)(assignments.size() - clusters.size());	// all objects except medoids
		mLastAvgClusterDistance /= (FLOAT)clusters.size();					// all clusters

		// Report whether the medoids vector has been modified (if not -> early termination).
		return changed;
	}


	/**
	 * \brief Internal run method called by public run() interface.
	 * \param objects Input set of objects to be clustered.
	 * \param medoids The result medoid objects (as indices to the objects vector).
	 * \param assignments The result assignment value for each object. The assignment vector
	 *		has the same size as objects vector and each value is an index to medoids vector.
	 * \param distances Pointer to an array, where distances to assigned medoids are kept.
	 *		If nullptr, the distances are not saved.
	 * \return Number of iterations performed.
	 */
	virtual std::size_t runPrivate(const OBJ_CONTAINER& objects, std::vector<std::size_t>& medoids,
		std::vector<std::size_t>& assignments, FLOAT* distances = nullptr)
	{
		std::size_t objectCount = objects.size();
		checkParams(objectCount);
		initRandMedoids(medoids, objectCount);
		medoids.resize(mK);

		std::size_t iter = 0;
		while (iter < mMaxIters) {
			++iter;

			// Compute new assignments of objects to medoids.
			computeAssignments(objects, objectCount, medoids, assignments, distances);

			// Update medoids (terminate if no update occured).
			if (!updateMedoids(objects, medoids, assignments)) break;
		}

		return iter;
	}


public:
	KMedoids(DIST& distFnc, std::size_t k, std::size_t maxIters)
		: mDistFnc(distFnc), mK(k), mMaxIters(maxIters), mLastAvgDistance(0), mLastAvgClusterDistance(0) {}

	std::size_t getK() const { return mK; }
	void setK(std::size_t k) { mK = k; }

	std::size_t getMaxIters() const { return mMaxIters; }
	void setMaxIters(std::size_t maxIters) { mMaxIters = maxIters; }

	FLOAT getLastAvgDistance() const { return mLastAvgDistance; }
	FLOAT getLastAvgClusterDistance() const { return mLastAvgClusterDistance; }

	const std::vector<FLOAT>& getBestScores() const { return mBestScores; }

	/**
	* \brief Run the k-medoids clustering on given data.
	* \param objects Pointer to a C array of objects.
	* \param objectCount Number of objects in the input array.
	* \param medoids The result medoid objects (as indices to the objects vector).
	* \param assignments The result assignment value for each object. The assignment vector
	*		has the same size as objects vector and each value is an index to medoids vector.
	 * \param distances Pointer to an array, where distances to assigned medoids are kept.
	 *		If nullptr, the distances are not saved.
	* \return Number of iterations performed.
	*/
	std::size_t run(const OBJ_CONTAINER& objects, std::vector<std::size_t>& medoids,
		std::vector<std::size_t>& assignments, FLOAT* distances = nullptr)
	{
		// Just recall private virtual method...
		return this->runPrivate(objects, medoids, assignments, distances);
	}
};

#endif
