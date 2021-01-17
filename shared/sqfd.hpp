#ifndef KMEDOIDS_SHARED_SQFD_HPP
#define KMEDOIDS_SHARED_SQFD_HPP

#include <signatures.hpp>

#include <memory>
#include <vector>
#include <cmath>


/**
 * \brief Functor that computes SQFD distance serially on CPU.
 * \tparam DIM Dimension of the feature space.
 * \tparam NUM_TYPE Numeric type in which the input/output values are (float or double).
 * \tparam IN_TYPE Numeric type for intermediate results (float or double).
 * \tparam LP The p value (x1000) for the Lp distance function (default is 2000 ~ L2 metric).
 */
template<int DIM, typename NUM_TYPE = float, typename IN_TYPE = float, int LP = 2000>
class DistanceFunctorSQFD
{
private:
	DBSignature<DIM, NUM_TYPE> mQuery;		///< Inteligent reference to query.
	IN_TYPE mAlpha;							///< Alpha parameter for the similarity function.
	std::vector<NUM_TYPE> mQueryWeights;	///< Normalized copy of query weights.
	NUM_TYPE mQuerySimilarity;				///< Self-similarity value of the query object.


	/**
	 * \brief Compute squared Lp distance according to templated LP value.
	 * \param first Pointer to array of #DIM numbers representing first coordinates.
	 * \param second Pointer to array of #DIM numbers representing second coordinates.
	 * \return (L_p)^2 value of the two given points, where p is the LP template value.
	 */
	static inline IN_TYPE distance(const NUM_TYPE *first, const NUM_TYPE *second)
	{
		IN_TYPE dist = (IN_TYPE)0.0;
		for (std::size_t d = 0; d < DIM; ++d) {
			IN_TYPE diff = (IN_TYPE)first[d] - (IN_TYPE)second[d];
			if (LP == 2000)
				dist += diff * diff;
			else if (LP == 1000)
				dist += std::fabs(diff);
			else if (LP == 5000)
				dist += std::fabs(diff) * diff * diff * diff * diff;
			else if (LP == 500)
				dist += std::sqrt(std::fabs(diff));
			else if (LP == 250)
				dist += std::sqrt(std::sqrt(std::fabs(diff)));
			else
				dist += std::pow(std::fabs(diff), (IN_TYPE)LP/(IN_TYPE)1000.0);
		}

		if (LP == 2000)
			return dist;
		else if (LP == 1000)
			;	// Nothing else to be done
		else if (LP == 500) {
			dist *= dist;
		}
		else if (LP == 250) {
			dist *= dist;
			dist *= dist;
		}
		else
			dist = std::pow(dist, (IN_TYPE)1000.0/(IN_TYPE)LP);
		return dist*dist;	// Return Lp squared.
	}


	/**
	 * \brief Compute similarity function for selected two coordinates.
	 * \param first Pointer to array of #DIM numbers representing first coordinates.
	 * \param second Pointer to array of #DIM numbers representing second coordinates.
	 * \param alpha Similarity function precision parameter.
	 */
	static inline IN_TYPE similarity(const NUM_TYPE *first, const NUM_TYPE *second, IN_TYPE alpha)
	{
		return std::exp(-alpha * distance(first, second));
	}


	/**
	 * \brief Compute similarity function for selected two coordinates.
	 * \param first Pointer to array of #DIM numbers representing first coordinates.
	 * \param second Pointer to array of #DIM numbers representing second coordinates.
	 */
	inline IN_TYPE similarity(const NUM_TYPE *first, const NUM_TYPE *second) const
	{
		return similarity(first, second, mAlpha);
	}


	/**
	 * \brief Copy object weights from given array to a vector and normalize them.
	 * \param weights Pointer to weights stream where the original data are.
	 * \param count Number of centroids (and the weights records).
	 * \param res Vector where normalized weights are stored.
	 */
	static inline void normalize(const NUM_TYPE *weights, std::size_t count, std::vector<NUM_TYPE> &res)
	{
		res.resize(count);
		IN_TYPE normalizer = (IN_TYPE)0;
		for (std::size_t i = 0; i < count; ++i) {
			normalizer += (IN_TYPE)weights[i];
		}
		for (std::size_t i = 0; i < count; ++i) {
			res[i] = (NUM_TYPE)((IN_TYPE)weights[i] / normalizer);
		}
	}


public:
	/**
	 * \brief Initialize the SQFD functor.
	 * \param query Query object signature referrence.
	 * \param alpha Tuning constant for similarity function.
	 */
	DistanceFunctorSQFD(const DBSignature<DIM, NUM_TYPE> &query, IN_TYPE alpha)
		: mQuery(query), mAlpha(alpha)
	{
		// The query normalization is done just once.
		normalize(query.getWeights(), query.getCentroidCount(), mQueryWeights);
		mQuerySimilarity = selfSimilarity(query, alpha);
	}


	/**
	 * \brief Main function of the functor.
	 * \param sig The object signature for which the distance (from query) is computed.
	 * \return Distance value (from 0 to 1 range).
	 */
	NUM_TYPE operator()(const DBSignature<DIM, NUM_TYPE> &sig) const
	{
		IN_TYPE res = (IN_TYPE)0.0;
		const NUM_TYPE *coords1 = mQuery.getCoordinates();
		const NUM_TYPE *coords2 = sig.getCoordinates();

		std::vector<NUM_TYPE> dbObjWeights;
		normalize(sig.getWeights(), sig.getCentroidCount(), dbObjWeights);

		for (std::size_t i = 0; i < mQuery.getCentroidCount(); ++i) {
			IN_TYPE intermediate = (IN_TYPE)0.0;
			for (std::size_t j = 0; j < mQuery.getCentroidCount(); ++j) {
				intermediate += (IN_TYPE)mQueryWeights[j] * similarity(coords1 + i*DIM, coords1 + j*DIM);
			}
			for (std::size_t j = 0; j < sig.getCentroidCount(); ++j) {
				intermediate += (IN_TYPE)-dbObjWeights[j] * similarity(coords1 + i*DIM, coords2 + j*DIM);
			}
			res += intermediate * (IN_TYPE)mQueryWeights[i];
		}

		for (std::size_t i = 0; i < sig.getCentroidCount(); ++i) {
			IN_TYPE intermediate = 0.0;
			for (std::size_t j = 0; j < mQuery.getCentroidCount(); ++j) {
				intermediate += (IN_TYPE)mQueryWeights[j] * similarity(coords2 + i*DIM, coords1 + j*DIM);
			}
			for (std::size_t j = 0; j < sig.getCentroidCount(); ++j) {
				intermediate += (IN_TYPE)-dbObjWeights[j] * similarity(coords2 + i*DIM, coords2 + j*DIM);
			}

			res += intermediate * (IN_TYPE)-dbObjWeights[i];
		}

		return (res > 0.000001) ? (NUM_TYPE)std::sqrt(res) : 0.0f;
	}


	/**
	 * \brief Main function of the functor.
	 * \param sig The object signature for which the distance (from query) is computed.
	 * \return Distance value (from 0 to 1 range).
	 */
	NUM_TYPE operator()(const DBSignature<DIM, NUM_TYPE> &sig, NUM_TYPE sigSelfSimilarity) const
	{
		IN_TYPE res = (IN_TYPE)0.0;
		const NUM_TYPE *coords1 = mQuery.getCoordinates();
		const NUM_TYPE *coords2 = sig.getCoordinates();

		std::vector<NUM_TYPE> dbObjWeights;
		normalize(sig.getWeights(), sig.getCentroidCount(), dbObjWeights);

		for (std::size_t i = 0; i < mQuery.getCentroidCount(); ++i) {
			IN_TYPE intermediate = (IN_TYPE)0.0;
			for (std::size_t j = 0; j < sig.getCentroidCount(); ++j) {
				intermediate += (IN_TYPE)dbObjWeights[j] * similarity(coords1 + i*DIM, coords2 + j*DIM);
			}
			res += intermediate * (IN_TYPE)mQueryWeights[i];
		}

		res = (IN_TYPE)mQuerySimilarity + (IN_TYPE)sigSelfSimilarity - 2.0*res;
		return (res > 0.000001) ? (NUM_TYPE)std::sqrt(res) : 0.0f;
	}


	/**
	 * \brief Compute self-similarity value of given signature.
	 * \param sig DB signature for which the self-similarity value is computed.
	 * \param alpha The similarity function alpha parameter.
	 * \return The self-similarty part of the SQFD (note that the results is NOT sqrt-ed).
	 */
	static NUM_TYPE selfSimilarity(const DBSignature<DIM, NUM_TYPE> &sig, IN_TYPE alpha)
	{
		IN_TYPE res = (IN_TYPE)0.0;
		const NUM_TYPE *coords = sig.getCoordinates();

		std::vector<NUM_TYPE> dbObjWeights;
		normalize(sig.getWeights(), sig.getCentroidCount(), dbObjWeights);

		for (std::size_t i = 0; i < sig.getCentroidCount(); ++i) {
			IN_TYPE intermediate = 0.0;
			for (std::size_t j = 0; j < sig.getCentroidCount(); ++j) {
				intermediate += (IN_TYPE)dbObjWeights[j] * similarity(coords + i*DIM, coords + j*DIM, alpha);
			}

			res += intermediate * (IN_TYPE)dbObjWeights[i];
		}

		return (NUM_TYPE)res;
	}

};


/**
 * Wrapper for SQFD so it can be used as binary functor (with some caching involved).
 */
template<int DIM, typename NUM_TYPE = float, typename IN_TYPE = float, int LP = 2000>
class BinaryDistanceFunctorSQFD
{
public:
	using sqfd_t = DistanceFunctorSQFD<DIM, NUM_TYPE, IN_TYPE, LP>;

private:
	const NUM_TYPE *mQueryRawData;				///< Pointer to raw data for last query object.
	IN_TYPE mAlpha;								///< Alpha parameter for the similarity function.
	std::unique_ptr<sqfd_t> mFnc;				///< Cache for last created SQFD functor.

public:
	BinaryDistanceFunctorSQFD(IN_TYPE alpha) : mQueryRawData(nullptr), mAlpha(alpha) {}

	/**
	 * \brief Main function of the functor.
	 * \param querySig The query signature. Query functor is cached.
	 * \parma objSig Another object signature (not cached).
	 * \return Distance between query signature and object signature.
	 */
	NUM_TYPE operator()(const DBSignature<DIM, NUM_TYPE>& querySig, const DBSignature<DIM, NUM_TYPE>& objSig)
	{
		if (mQueryRawData != querySig.getRawData()) {
			mQueryRawData = querySig.getRawData();
			mFnc = std::make_unique<sqfd_t>(querySig, mAlpha);
		}

		return (*mFnc.get())(objSig);
	}
};

#endif
