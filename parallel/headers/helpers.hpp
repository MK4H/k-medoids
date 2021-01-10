#ifndef K_MEDOIDS_HELPERS_H_
#define K_MEDOIDS_HELPERS_H_

#include <mpi.h>
#include <iostream>
#include <stdint.h>
#include <limits.h>

#define MPICH(status) print_MPI_error(status, __LINE__, __FILE__, #status)

template<typename T>
void print(std::ostream &out, T * array, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        out << array[i] << ", ";
    }
    out << std::endl;
}

int print_MPI_error(int code, int line, const char *file,  const std::string &header) {
    if (code == MPI_SUCCESS) {
        return code;
    }
    char mpi_message[MPI_MAX_ERROR_STRING];
    int mpi_message_len;

    MPI_Error_string(code, mpi_message, &mpi_message_len);

    std::cerr << "Error at " << file << ":" << line << " at " << header << std::endl;
    std::cerr << std::string_view(mpi_message, mpi_message_len) << std::endl;

    return code;
}

// Adapted from https://stackoverflow.com/questions/40807833/sending-size-t-type-data-with-mpi
#if SIZE_MAX == UCHAR_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED_CHAR
#elif SIZE_MAX == USHRT_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED_SHORT
#elif SIZE_MAX == UINT_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED
#elif SIZE_MAX == ULONG_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#else
   #error "Unexpected size of std::size_t"
#endif


#endif