#ifndef DEFINITIONS
#define DEFINITIONS

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#include <string>
#include <vector>
#include <fstream>
#include <typeinfo>
#include <iomanip>
#include <random>
#include <algorithm>


#define myError(error, ...)                                                  \
{                                                                            \
  printf("!!!!!!!!!!!!!!!!!!!!!!!!\n");                                      \
  printf("Error - ");                                                        \
  printf((error), ##__VA_ARGS__);                                            \
  printf(" (%s: %d)\n", __FILE__, __LINE__);                                 \
  printf("!!!!!!!!!!!!!!!!!!!!!!!!\n");                                      \
  exit(1);                                                                   \
}

typedef double myfloat_t;
typedef myfloat_t mycomplex_t[2];
typedef std::vector<myfloat_t> myvector_t;
typedef std::vector<myvector_t> mymatrix_t;

#define mystof std::stod
#define mympi_float_t MPI_DOUBLE

typedef struct param_device{

  std::string struct_file, img_pfx;
  int n_pixels, n_neigh, n_imgs, n_atoms;
  bool with_rot = false;
  myfloat_t pixel_size, sigma, cutoff, norm;
  
  myvector_t grid;    

  void calc_neigh(){
    n_neigh = (int) std::ceil(sigma * cutoff / pixel_size);
  }

  void calc_norm(){ 
    norm = 1. / (2*M_PI * sigma*sigma * n_atoms);
  }

  void gen_grid(){

    myfloat_t grid_min = -pixel_size * (n_pixels - 1) * 0.5;

    grid.resize(n_pixels);
    for (int i=0; i<n_pixels; i++) grid[i] = grid_min + pixel_size * i;
  }

} myparam_t;

typedef struct image{

  myvector_t q = myvector_t(4, 0.0);
  myvector_t intensity;

  std::string fname;

} myimage_t;

typedef std::vector<myimage_t> mydataset_t;

#pragma omp declare reduction(vec_float_plus : std::vector<myfloat_t> : \
                              std::transform(omp_out.begin(), omp_out.end(), \
                              omp_in.begin(), omp_out.begin(), std::plus<myfloat_t>())) \
                              initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

#endif