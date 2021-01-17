#ifndef K_MEDOIDS_HELPERS_H_
#define K_MEDOIDS_HELPERS_H_

#include <mpi.h>
#include <iostream>
#include <stdint.h>
#include <limits.h>
#include <streambuf>

// NOT PORTABLE, but i would love to see SLURM cluster on Windows
#include <unistd.h>
#include <fcntl.h>

#define MPICH(status) print_MPI_error(status, __LINE__, __FILE__, #status)

class fdRedirecter
{
public:
   fdRedirecter(const std::string &filenameBase) {

      std::string out = filenameBase + std::string(".out");
      std::string err = filenameBase + std::string(".err");
      // TODO: Check for errors
      outfd = open(out.c_str(), O_CREAT | O_TRUNC | O_WRONLY);
      errfd = open(err.c_str(), O_CREAT | O_TRUNC | O_WRONLY);
      dup2(outfd, fileno(stdout));
      dup2(errfd, fileno(stderr));
   }

   ~fdRedirecter() {
      close(outfd);
      close(errfd);
   }
private:
   int outfd;
   int errfd;
};

// Taken from http://wordaligned.org/articles/cpp-streambufs
class teebuf: public std::streambuf
{
public:
   // Construct a streambuf which tees output to both input
   // streambufs.
   teebuf(std::streambuf * sb1, std::streambuf * sb2)
      : sb1(sb1)
      , sb2(sb2)
   {
   }
private:
   // This tee buffer has no buffer. So every character "overflows"
   // and can be put directly into the teed buffers.
   virtual int overflow(int c)
   {
      if (c == EOF)
      {
         return !EOF;
      }
      else
      {
         int const r1 = sb1->sputc(c);
         int const r2 = sb2->sputc(c);
         return r1 == EOF || r2 == EOF ? EOF : c;
      }
   }

   // Sync both teed buffers.
   virtual int sync()
   {
      int const r1 = sb1->pubsync();
      int const r2 = sb2->pubsync();
      return r1 == 0 && r2 == 0 ? 0 : -1;
   }
private:
   std::streambuf * sb1;
   std::streambuf * sb2;
};




class teestream : public std::ostream
{
public:
    // Construct an ostream which tees output to the supplied
    // ostreams.
    teestream(std::ostream & o1, std::ostream & o2);
private:
    teebuf tbuf;
};

   teestream::teestream(std::ostream & o1, std::ostream & o2)
   : std::ostream(&tbuf)
   , tbuf(o1.rdbuf(), o2.rdbuf())
   {
}



class redirecter
{
public:
   redirecter(std::ostream & dst, std::ostream & src)
      : src(src), sbuf(src.rdbuf(dst.rdbuf())) {}
   ~redirecter() { src.rdbuf(sbuf); }
private:
   std::ostream & src;
   std::streambuf * const sbuf;
};




template<typename T>
void print(std::ostream &out, T * array, size_t size) {
   for (size_t i = 0; i < size; ++i) {
      out << array[i] << ", ";
   }
   out << std::endl;
}

template<typename T>
void print(std::ostream &out, const std::vector<T> &vec) {
   print(out, vec.data(), vec.size());
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