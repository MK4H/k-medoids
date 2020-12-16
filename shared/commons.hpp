#ifndef KMEDOIDS_SHARED_COMMONS_HPP
#define KMEDOIDS_SHARED_COMMONS_HPP

#include <misc/exception.hpp>


typedef std::uint64_t db_offset_t;



/**
 * \brief Specific exception thrown from database objects when internal error occurs.
 */
class DBException : public bpp::StreamException
{
public:
	DBException() : bpp::StreamException() {}
	DBException(const char *msg) : bpp::StreamException(msg) {}
	DBException(const std::string &msg) : bpp::StreamException(msg) {}
	virtual ~DBException() throw() {}

	/*
	 * Overloading << operator that uses stringstream to append data to mMessage.
	 * Note that this overload is necessary so the operator returns object of exactly this class.
	 */
	template<typename T> DBException& operator<<(const T &data)
	{
		bpp::StreamException::operator<<(data);
		return *this;
	}
};


#endif
