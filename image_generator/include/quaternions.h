#ifndef QUATERNIONS
#define QUATERNIONS

#include "definitions.h"

void generate_quaternions(mydataset_t &, int);
void quaternion_rotation(myvector_t &, myvector_t &, myvector_t &);
void quaternion_rotation(myvector_t &, myvector_t &);

#endif