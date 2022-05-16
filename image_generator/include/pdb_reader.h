#ifndef PDB_READER
#define PDB_READER

#include "definitions.h"

void pdb_parser(std::string, myvector_t &, std::vector<std::string> &);
void get_coordinates(myparam_t *, myvector_t &, std::vector<std::string> &, int);

#endif