#include "config_reader.h"

void parse_args(int argc, char* argv[], int rank, int world_size, std::string &param_file, int &ntomp){

  bool yesParams = false;
  bool yesNtomp = false;

  for (int i = 1; i < argc; i++){

    if ( strcmp(argv[i], "--input") == 0){
      param_file = argv[i+1]; 
      yesParams = true; 
      i++; continue;
    }

    else if ( strcmp(argv[i], "-ntomp") == 0){
      ntomp = std::atoi(argv[i+1]); 
      yesNtomp = true;
      i++; continue;
    } 

    else{
      myError("Unknown argument %s", argv[i]);
    }
  }

  if (!yesNtomp) ntomp = 1;
  if (!yesParams) param_file = "config.txt";

  if (rank == 0){

    int req_cores = world_size * ntomp;
    int avail_cores = std::thread::hardware_concurrency();
    if (req_cores > avail_cores) myError("%d Cores required, but only %d available\n", req_cores, avail_cores);

    printf("Using %d OMP Threads and %d MPI Ranks\n", ntomp, world_size);
    printf("INPUT FILE %s\n", param_file.c_str());
  }
}

void parse_input_file(std::string fname, myparam_t *PARAM, int rank){ 

  std::ifstream input(fname);
  if (!input.good()) myError("Opening file: %s", fname.c_str());

  char line[512] = {0};
  char saveline[512];

  if (rank==0) std::cout << "\n +++++++++++++++++++++++++++++++++++++++++ \n";
  if (rank==0) std::cout << "\n   READING EM2D PARAMETERS            \n\n";
  if (rank==0) std::cout << " +++++++++++++++++++++++++++++++++++++++++ \n";

  bool yesStrucFile = false;
  bool yesImgPfx = false, yesNumImgs = false;
  bool yesPixSi = false, yesNumPix = false; 
  bool yesSigma = false, yesCutoff = false;

  int mode = 0;

  while (input.getline(line, 512)){

    strcpy(saveline, line);
    char *token = strtok(line, " ");

    if (token == NULL || line[0] == '#' || strlen(token) == 0){
      // comment or blank line
    }

    if (token == NULL || line[0] == '#' || strlen(token) == 0){
    // comment or blank line
    }

    else if (strcmp(token, "STRUCTURE") == 0){

      token = strtok(NULL, " ");
      PARAM->struct_file = token;
      
      if (rank==0) std::cout << "Structural file " << PARAM->struct_file << "\n";
      yesStrucFile = true;
    }

    else if (strcmp(token, "PIXEL_SIZE") == 0){

    token = strtok(NULL, " ");
    PARAM->pixel_size = atof(token);
    
    if (PARAM->pixel_size < 0) myError("Negative pixel size");
    if (rank==0) std::cout << "Pixel Size " << PARAM->pixel_size << "\n";

    yesPixSi = true;
    }

    else if (strcmp(token, "N_PIXELS") == 0){
    
      token = strtok(NULL, " ");
      PARAM->n_pixels = int(atoi(token));

      if (PARAM->n_pixels < 0){

          myError("Negative Number of Pixels");
      }

      if (rank==0) std::cout << "Number of Pixels " << PARAM->n_pixels << "\n";
      yesNumPix = true;
    }

    else if (strcmp(token, "WITH_ROTATIONS") == 0){
    
      token = strtok(NULL, " ");
      PARAM->with_rot = true;

      if (rank==0) std::cout << "Using random rotations" << "\n";
    }
    // CV PARAMETERS
    else if (strcmp(token, "SIGMA") == 0){

      token = strtok(NULL, " ");
      PARAM->sigma = atof(token);

      if (PARAM->sigma < 0) myError("Negative standard deviation for the gaussians");
      
      if (rank==0) std::cout << "Sigma " << PARAM->sigma << "\n";
      yesSigma = true;
    }

    else if (strcmp(token, "CUTOFF") == 0){

      token = strtok(NULL, " ");
      PARAM->cutoff = atof(token);
      if (PARAM->cutoff < 0) myError("Negative cutoff");
      
      if (rank==0) std::cout << "Cutoff " << PARAM->cutoff << "\n";
      yesCutoff = true;
    }

    else if (strcmp(token, "IMG_PFX") == 0){
    
      token = strtok(NULL, " ");
      PARAM->img_pfx = token;

      if (rank == 0) std::cout << "IMG_PFX " << PARAM->img_pfx << "\n";
      yesImgPfx = true;
    } 

    else if (strcmp(token, "NUM_IMAGES") == 0){
    
      token = strtok(NULL, " ");
      PARAM->n_imgs = int(atoi(token));
      if (PARAM->n_imgs < 0) myError("Negative number of images");

      if (rank==0) std::cout << "Num. Images " << PARAM->n_imgs << "\n";
      yesNumImgs = true;
    }

    else {
      myError("Unknown mode or parameter %s", token);
    }   
  }
  input.close();

  if (rank == 0){

    if (not(yesStrucFile)){
      myError("Input missing: please provide STRUCTURE")
    }

    if (not(yesPixSi)){
      myError("Input missing: please provide PIXEL_SIZE");
    }

    if (not(yesNumPix)){
      myError("Input missing: please provide n_pixels");
    }

    if (not(yesSigma)){
      myError("Input missing: please provide SIGMA");
    }

    if (not(yesCutoff)){
      myError("Input missing: please provide CUTOFF");
    }

    if (not(yesImgPfx)){
      myError("Input missing: please provide IMG_PFX");
    }
  }
}