#ifndef KMEDOIDS_SHARED_POINTS_HPP
#define KMEDOIDS_SHARED_POINTS_HPP

#include <commons.hpp>

#include <vector>
#include <cstdint>


/**
 * \brief A list of points in R^DIM space. The points are stored in column-wise style.
 * \tparam DIM Dimension of the space of the points (float or double).
 * \tparam NUM_TYPE Numeric type in which the coordinates are kept.
 */
template<int DIM, typename NUM_TYPE = float>
class Points
{
private:
	std::vector<NUM_TYPE> mData[DIM];

public:
	virtual ~Points() {}	// enforcing virtual destructor for descendants
	
	/**
	 * \breaf Return reference to a coordinate of selected point. This methods is designed
	 *		to shield user from the data storage details.
	 * \param idx Index of the point.
	 * \param dim Dimension of the coordinate.
	 */
	NUM_TYPE& get(std::size_t idx, std::size_t dim)
	{
		return mData[dim][idx];
	}


	/**
	 * \breaf Return read-only reference to a coordinate of selected point. This methods is designed
	 *		to shield user from the data storage details.
	 * \param idx Index of the point.
	 * \param dim Dimension of the coordinate.
	 */
	const NUM_TYPE& get(std::size_t idx, std::size_t dim) const
	{
		return mData[dim][idx];
	}


	/**
	 * \brief Get the number of points in the list.
	 */
	std::size_t size() const
	{
		return mData[0].size();
	}


	/**
	 * \brief Clear the list of points.
	 */
	void clear()
	{
		resize(0);
	}


	/**
	 * \brief Change the number of points. If the list is grown,
	 *		missing points are filled with zeros.
	 * \param count New size of the list.
	 */
	virtual void resize(std::size_t count)
	{
		for (std::size_t d = 0; d < DIM; ++d)
			mData[d].resize(count);
	}


	/**
	 * \brief Change the capacity of the list (usually to avoid reallocations).
	 * \param count New capacity of the list.
	 */
	virtual void reserve(std::size_t count)
	{
		for (std::size_t d = 0; d < DIM; ++d)
			mData[d].reserve(count);
	}


	/**
	 * \brief Insert another point at the end of the list.
	 * \param point Array of NUM_TYPE of length DIM where the coordinates are.
	 */
	virtual void addPoint(const NUM_TYPE *point)
	{
		for (std::size_t d = 0; d < DIM; ++d)
			mData[d].push_back(point[d]);
	}
	

	/**
	 * \brief Insert another point at the end of the list.
	 * \param point Vector of length DIM where the coordinates are.
	 */
	void addPoint(const std::vector<NUM_TYPE>& point)
	{
		if (point.size() != DIM)
			throw (DBException() << "Point dimension " << point.size() << " mismatch the point list dimension " << DIM);
		addPoint(&(point[0]));
	}


	/**
	 * \brief Copy point from selected point list and append it at the end of this list.
	 * \param points A list of points from which the point is copied.
	 * \param idx Index of the point (in the points list) to be copied.
	 */
	virtual void addPoint(const Points<DIM, NUM_TYPE> &points, std::size_t idx)
	{
		for (std::size_t d = 0; d < DIM; ++d)
			mData[d].push_back(points.get(idx, d));
	}


	/**
	 * \brief Remove point at selected index and reduce the size by one.
	 * \param idx Index of the point to be removed.
	 * \note Relative order of points from idx to the end is NOT maintained.
	 */
	virtual void removePoint(std::size_t idx)
	{
		for (std::size_t d = 0; d < DIM; ++d) {
			swap(mData[d][idx], mData[d].back());
			mData[d].pop_back();
		}
	}
};




/**
 * \brief Extension of the points list that associates weights with points.
 * \tparam DIM Dimension of the space of the points.
 * \tparam NUM_TYPE Numeric type in which the coordinates are kept (float or double).
 * \tparam WEIGHT_TYPE Numeric type of the weights (usually an integer type).
 */
template<int DIM, typename NUM_TYPE = float, typename WEIGHT_TYPE = std::uint32_t>
class PointsWithWeights : public Points<DIM, NUM_TYPE>
{
private:
	std::vector<WEIGHT_TYPE> mWeights;	///< Associated weight values.

public:
	/**
	 * \brief Accessor to the weight of selected point.
	 * \param idx Index of the corresponding point.
	 */
	WEIGHT_TYPE& weight(std::size_t idx)
	{
		return mWeights[idx];
	}


	/**
	 * \brief Read-only accessor to the  weight of selected point.
	 * \param idx Index of the corresponding point.
	 */
	const WEIGHT_TYPE& weight(std::size_t idx) const
	{
		return mWeights[idx];
	}


	/**
	 * \brief Sort the points by their weights in descendant order.
	 */
	void sortByWeights()
	{
		// Compute a permutation that fixes the order of points.
		std::vector<std::pair<WEIGHT_TYPE, std::size_t> > order;
		order.reserve(this->size());
		for (std::size_t i = 0; i < this->size(); ++i)
			order.push_back(std::pair<WEIGHT_TYPE, std::size_t>(mWeights[i], i));
		sort_by_first_desc(order);

		// Flags that check whether an item is already at its proper place.
	    std::vector<bool> inPlace(order.size());
		for (std::size_t i = 0; i < this->size(); ++i) {
			if (inPlace[i] || order[i].second == i)
				continue;

			// Save first item of the cycle.
			NUM_TYPE tmpPoint[DIM];
			WEIGHT_TYPE tmpWeight;
			for (std::size_t d = 0; d < DIM; ++d)
				tmpPoint[d] = this->get(i, d);
			tmpWeight = mWeights[i];

			// Follow the permutation cycle and swap items.
			std::size_t pos = i;
	   		while (order[pos].second != i) {
				std::size_t nextPos = order[pos].second;

				for (std::size_t d = 0; d < DIM; ++d)
					this->get(pos, d) = this->get(nextPos, d);
				mWeights[pos] = mWeights[nextPos];
				inPlace[pos] = true;
				pos = nextPos;
			}

			// Fill the last position of the broken cycle with saved point.
			for (std::size_t d = 0; d < DIM; ++d)
				this->get(pos, d) = tmpPoint[d];
			mWeights[pos] = tmpWeight;
			inPlace[pos] = true;
		}
	}  


	/**
	 * \brief Remove points whose weight is lesser or equal to given threshold.
	 *		The point list is compacted and relative order of points is maintained.
	 * \param dropThreshold Largest weight of the points being dropped.
	 */
	void dropLightPoints(WEIGHT_TYPE dropThreshold = (WEIGHT_TYPE)0)
	{
		std::size_t frontIdx = 0;

		// Skip leading continuous part of weighted-enough points.
		while (frontIdx < this->size() && mWeights[frontIdx] > dropThreshold)
			++frontIdx;

		// Mark first emptied position and advance front index.
		std::size_t tailIdx = frontIdx++;

		while (frontIdx < this->size()) {
			if (mWeights[frontIdx] > dropThreshold) {
				// Current (front) item is not dropped -> copy it to the tail.
				for (std::size_t d = 0; d < DIM; ++d)
					this->get(tailIdx, d) = this->get(frontIdx, d);
				mWeights[tailIdx] = mWeights[frontIdx];

				++tailIdx;	// grow the tail
			}
			++frontIdx;
		}

		resize(tailIdx);
	}


	/*
	 * Reimplementation of Virtual Methods
	 */

	virtual void resize(std::size_t count)
	{
		Points<DIM, NUM_TYPE>::resize(count);
		mWeights.resize(count);
	}


	virtual void reserve(std::size_t count)
	{
		Points<DIM, NUM_TYPE>::reserve(count);
		mWeights.reserve(count);
	}


	virtual void addPoint(const NUM_TYPE *point)
	{
		Points<DIM, NUM_TYPE>::addPoint(point);
		mWeights.push_back((WEIGHT_TYPE)0);
	}
	

	virtual void addPoint(const Points<DIM, NUM_TYPE> &points, std::size_t idx)
	{
		Points<DIM, NUM_TYPE>::addPoint(points, idx);
		mWeights.push_back((WEIGHT_TYPE)0);
	}


	virtual void addPoint(const PointsWithWeights<DIM, NUM_TYPE, WEIGHT_TYPE> &points, std::size_t idx)
	{
		Points<DIM, NUM_TYPE>::addPoint(points, idx);
		mWeights.push_back(points.weight(idx));
	}


	virtual void removePoint(std::size_t idx)
	{
		Points<DIM, NUM_TYPE>::removePoint(idx);
		swap(mWeights[idx], mWeights.back());
		mWeights.pop_back();
	}
};


#endif
