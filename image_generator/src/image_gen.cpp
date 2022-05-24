#include "image_gen.h"
#include "pdb_reader.h"
#include "quaternions.h"

void run_gen(myparam_t param_device, int rank, int world_size, int ntomp){

  // Parse pdb file
  myvector_t r_coord;
  std::vector <std::string> atom_names;

  get_coordinates(&param_device, r_coord, atom_names, rank);

  std::cout << r_coord[0] << " " << param_device.n_atoms << " " << param_device.n_pixels << std::endl;

  // Create grid
  param_device.gen_grid();
  param_device.calc_neigh();
  param_device.calc_norm();

  if (param_device.n_imgs < world_size) myError("The number of images must be bigger than the number of processes!")

  int imgs_per_process = param_device.n_imgs / world_size;
  int residual_images = param_device.n_imgs % world_size;
  
  int start_img;
  int end_img;

  mydataset_t exp_imgs;

  if (rank == 0){

    start_img = 0;
    end_img = start_img + imgs_per_process + residual_images;
    exp_imgs.resize(imgs_per_process + residual_images);
  }

  else {

    start_img = rank * imgs_per_process + residual_images;
    end_img = start_img + imgs_per_process;

    exp_imgs.resize(imgs_per_process);
  }

  int counter = 0;
  for (int i=start_img; i<end_img; i++){

    // Allocate memory for later
    exp_imgs[counter].intensity = myvector_t(param_device.n_pixels * param_device.n_pixels, 0.0);

    // Set image name
    exp_imgs[counter].fname = param_device.img_pfx + std::to_string(i) + ".txt";

    counter++;
  } 

  if (rank == 0) printf("\nGenerating images...\n");
  
  if (param_device.with_rot){

    generate_quaternions(exp_imgs, exp_imgs.size());

    myvector_t r_rot = r_coord;
    for (int i=0; i<exp_imgs.size(); i++){

      quaternion_rotation(exp_imgs[i].q, r_coord, r_rot);
      calc_img_omp(r_rot, exp_imgs[i].intensity, &param_device, ntomp);
      print_image(&exp_imgs[i], param_device.n_pixels);
    }
  }

  else {
    for (int i=0; i<exp_imgs.size(); i++){

      calc_img_omp(r_coord, exp_imgs[i].intensity, &param_device, ntomp);
      print_image(&exp_imgs[i], param_device.n_pixels);
    }
  }

  if (rank == 0) printf("...done\n");
}

void calc_img_omp(myvector_t &r_a, myvector_t &I_c, myparam_t *PARAM, int ntomp){

  #pragma omp parallel num_threads(ntomp)
  {
    int m_x, m_y;
    int ind_i, ind_j;

    // std::vector<size_t> x_sel, y_sel;
    myvector_t gauss_x(2*PARAM->n_neigh+3, 0.0);
    myvector_t gauss_y(2*PARAM->n_neigh+3, 0.0);

    //myvector_t I_c_thr = I_c;

    #pragma omp for reduction(vec_float_plus : I_c)
    for (int atom=0; atom<PARAM->n_atoms; atom++){

      m_x = (int) std::round(abs(r_a[atom] - PARAM->grid[0])/PARAM->pixel_size);
      m_y = (int) std::round(abs(r_a[atom + PARAM->n_atoms] - PARAM->grid[0])/PARAM->pixel_size);

      for (int i=0; i<=2*PARAM->n_neigh+2; i++){
        
        ind_i = m_x - PARAM->n_neigh - 1 + i;
        ind_j = m_y - PARAM->n_neigh - 1 + i;

        if (ind_i<0 || ind_i>=PARAM->n_pixels) gauss_x[i] = 0;
        else {

          myfloat_t expon_x = (PARAM->grid[ind_i] - r_a[atom])/PARAM->sigma;
          gauss_x[i] = std::exp( -0.5 * expon_x * expon_x );
        }
              
        if (ind_j<0 || ind_j>=PARAM->n_pixels) gauss_y[i] = 0;
        else{

          myfloat_t expon_y = (PARAM->grid[ind_j] - r_a[atom + PARAM->n_atoms])/PARAM->sigma;
          gauss_y[i] = std::exp( -0.5 * expon_y * expon_y );
        }
      }

      //Calculate the image and the gradient
      for (int i=0; i<=2*PARAM->n_neigh+2; i++){
        
        ind_i = m_x - PARAM->n_neigh - 1 + i;
        if (ind_i<0 || ind_i>=PARAM->n_pixels) continue;
        
        for (int j=0; j<=2*PARAM->n_neigh+2; j++){

          ind_j = m_y - PARAM->n_neigh - 1 + j;
          if (ind_j<0 || ind_j>=PARAM->n_pixels) continue;

          I_c[ind_i*PARAM->n_pixels + ind_j] += gauss_x[i] * gauss_y[j];
        }
      }
    }

    #pragma omp for
    for (int i=0; i<I_c.size(); i++) {
      I_c[i] *= PARAM->norm;
    }
  }
}

void calc_img(myvector_t &r_a, myvector_t &I_c, myparam_t *PARAM, int ntomp){

  myvector_t gauss_x(PARAM->n_pixels, 0.0);
  myvector_t gauss_y(PARAM->n_pixels, 0.0);

  for (int atom=0; atom<PARAM->n_atoms; atom++){

    for (int i=0; i<PARAM->n_pixels; i++){

      myfloat_t expon_x = (PARAM->grid[i] - r_a[atom]) / PARAM->sigma;
      gauss_x[i] = std::exp( -0.5 * expon_x * expon_x );

      myfloat_t expon_y = (PARAM->grid[i] - r_a[atom + PARAM->n_atoms]) / PARAM->sigma;
      gauss_y[i] = std::exp( -0.5 * expon_y * expon_y );
    }

    //Calculate the image and the gradient
    for (int i=0; i<PARAM->n_pixels; i++){

      for (int j=0; j<PARAM->n_pixels; j++){

        I_c[i*PARAM->n_pixels + j] += gauss_x[i] * gauss_y[j];
      }
    }
  }

  for (int i=0; i<I_c.size(); i++) {
    I_c[i] *= PARAM->norm;
  }
}

void print_image(myimage_t *IMG, int n_pixels){

  std::ofstream matrix_file;
  matrix_file.open (IMG->fname);

  for (int i=0; i<n_pixels; i++){
    for (int j=0; j<n_pixels; j++){

      matrix_file << std::scientific << std::showpos << IMG->intensity[i*n_pixels + j] << " " << " \n"[j==n_pixels-1];
    }
  }

  matrix_file.close();
}
