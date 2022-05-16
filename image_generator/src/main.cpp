#include "definitions.h"
#include "config_reader.h"
#include "pdb_reader.h"
#include "image_gen.h"

int main(int argc, char *argv[])
{

  // Initialize the MPI environment
  MPI_Init(NULL, NULL);

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::string param_file;
  int ntomp;
  myparam_t param_device;

  parse_args(argc, argv, rank, world_size, param_file, ntomp);
  parse_input_file(param_file, &param_device, rank);

  omp_set_num_threads(ntomp);

  run_gen(param_device, rank, world_size, ntomp);

  // Finalize the MPI environment.
  MPI_Finalize();

  return 0;
}