#ifndef IMAGE_GEN
#define IMAGE_GEN

#include "definitions.h"

void run_gen(myparam_t, int, int, int);

void calc_img_omp(myvector_t &, myvector_t &, myparam_t *, int);
void calc_img(myvector_t &, myvector_t &, myparam_t *, int);

void print_image(myimage_t *, int);

#endif