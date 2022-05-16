#include "pdb_reader.h"

void pdb_parser(std::string fname, myvector_t &coords, std::vector<std::string> &atom_names){

  std::ifstream pdb_file(fname);

  if (!pdb_file.good()){
    myError("Opening file: %s", fname.c_str());
  }

  std::string line;
  std::vector<std::string> tokens;
  myvector_t x, y, z;

  int counter = 0;
  while (std::getline(pdb_file, line)){

    if (line.find("ATOM") != std::string::npos){

      std::istringstream iss(line);
      for (std::string s; iss >> s; ) tokens.push_back(s);

      atom_names.push_back(tokens[2]);

      if (tokens[2] == "CA"){

        x.push_back(mystof(tokens[6]));
        y.push_back(mystof(tokens[7]));
        z.push_back(mystof(tokens[8]));
      }


      tokens.clear();
    }
  }

  for (size_t i=0; i<x.size(); i++) coords.push_back(x[i]);
  for (size_t i=0; i<y.size(); i++) coords.push_back(y[i]);
  for (size_t i=0; i<z.size(); i++) coords.push_back(z[i]);
}

void get_coordinates(myparam_t *param, myvector_t &coord, std::vector<std::string> &atom_names, int rank){

  int coord_size;
  if (rank==0){
    
    pdb_parser(param -> struct_file, coord, atom_names);
    coord_size = coord.size();
  } 

  // Broadcast number of atoms to each MPI process
  MPI_Bcast(&coord_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank != 0) coord.resize(coord_size);
  param -> n_atoms = coord.size()/3;

  // Broadcast coordinates
  MPI_Bcast(&coord[0], coord.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
}