#include "quaternions.h"

void generate_quaternions(mydataset_t &dataset, int imgs_per_process){

  //Generate quats (Sonya's code)
  std::random_device seeder;
  std::mt19937 engine(seeder());

  std::uniform_real_distribution<myfloat_t> dist_quat(-1, 1);

  int counter = 0;
  myfloat_t qx, qy, qz, qw, q_norm;

  while (counter < imgs_per_process){

    qx = dist_quat(engine); qy = dist_quat(engine);
    qz = dist_quat(engine); qw = dist_quat(engine);

    q_norm = qx*qx + qy*qy + qz*qz + qw*qw;
    q_norm = std::sqrt(q_norm);

    if (0.2 <= q_norm && q_norm <= 1.0){

      dataset[counter].q[0] = qx/q_norm;
      dataset[counter].q[1] = qy/q_norm;
      dataset[counter].q[2] = qz/q_norm;
      dataset[counter].q[3] = qw/q_norm;

      counter++;
    }
  }  
}

void quaternion_rotation(myvector_t &q, myvector_t &r_ref, myvector_t &r_rot){

/**
 * @brief Rotates a biomolecule using the quaternions rotation matrix
 *        according to (https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion)
 * 
 * @param q vector that stores the parameters for the rotation myvector_t (4)
 * @param r_ref original coordinates 
 * @param r_rot stores the rotated values 
 * 
 * @return void
 * 
 */

  //Definition of the quaternion rotation matrix 

  myfloat_t q00 = 1 - 2*std::pow(q[1],2) - 2*std::pow(q[2],2);
  myfloat_t q01 = 2*q[0]*q[1] - 2*q[2]*q[3];
  myfloat_t q02 = 2*q[0]*q[2] + 2*q[1]*q[3];
  myvector_t q0{ q00, q01, q02 };
  
  myfloat_t q10 = 2*q[0]*q[1] + 2*q[2]*q[3];
  myfloat_t q11 = 1 - 2*std::pow(q[0],2) - 2*std::pow(q[2],2);
  myfloat_t q12 = 2*q[1]*q[2] - 2*q[0]*q[3];
  myvector_t q1{ q10, q11, q12 };

  myfloat_t q20 = 2*q[0]*q[2] - 2*q[1]*q[3];
  myfloat_t q21 = 2*q[1]*q[2] + 2*q[0]*q[3];
  myfloat_t q22 = 1 - 2*std::pow(q[0],2) - 2*std::pow(q[1],2);
  myvector_t q2{ q20, q21, q22};

  mymatrix_t Q{ q0, q1, q2 };
  
  int n_atoms = r_ref.size()/3;

  for (int i=0; i<n_atoms; i++){

    r_rot[i] = Q[0][0]*r_ref[i] + Q[0][1]*r_ref[i + n_atoms] + Q[0][2]*r_ref[i + 2*n_atoms];
    r_rot[i + n_atoms] = Q[1][0]*r_ref[i] + Q[1][1]*r_ref[i + n_atoms] + Q[1][2]*r_ref[i + 2*n_atoms];
    r_rot[i + 2*n_atoms] = Q[2][0]*r_ref[i] + Q[2][1]*r_ref[i + n_atoms] + Q[2][2]*r_ref[i + 2*n_atoms];
  }
}

void quaternion_rotation(myvector_t &q, myvector_t &r_ref){

/**
 * @brief Rotates a biomolecule using the quaternions rotation matrix
 *        according to (https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion)
 * 
 * @param q vector that stores the parameters for the rotation myvector_t (4)
 * @param x_data original coordinates x
 * @param y_data original coordinates y
 * @param z_data original coordinates z
 * @param x_r stores the rotated values x
 * @param y_r stores the rotated values x
 * @param z_r stores the rotated values x
 * 
 * @return void
 * 
 */

  //Definition of the quaternion rotation matrix 

  myfloat_t q00 = 1 - 2*std::pow(q[1],2) - 2*std::pow(q[2],2);
  myfloat_t q01 = 2*q[0]*q[1] - 2*q[2]*q[3];
  myfloat_t q02 = 2*q[0]*q[2] + 2*q[1]*q[3];
  myvector_t q0{ q00, q01, q02 };
  
  myfloat_t q10 = 2*q[0]*q[1] + 2*q[2]*q[3];
  myfloat_t q11 = 1 - 2*std::pow(q[0],2) - 2*std::pow(q[2],2);
  myfloat_t q12 = 2*q[1]*q[2] - 2*q[0]*q[3];
  myvector_t q1{ q10, q11, q12 };

  myfloat_t q20 = 2*q[0]*q[2] - 2*q[1]*q[3];
  myfloat_t q21 = 2*q[1]*q[2] + 2*q[0]*q[3];
  myfloat_t q22 = 1 - 2*std::pow(q[0],2) - 2*std::pow(q[1],2);
  myvector_t q2{ q20, q21, q22};

  mymatrix_t Q{ q0, q1, q2 };

  int n_atoms = r_ref.size()/3;

  myfloat_t x_tmp, y_tmp, z_tmp;
  for (unsigned int i=0; i<n_atoms; i++){

    x_tmp = Q[0][0]*r_ref[i] + Q[0][1]*r_ref[i + n_atoms] + Q[0][2]*r_ref[i + 2*n_atoms];
    y_tmp = Q[1][0]*r_ref[i] + Q[1][1]*r_ref[i + n_atoms] + Q[1][2]*r_ref[i + 2*n_atoms];
    z_tmp = Q[2][0]*r_ref[i] + Q[2][1]*r_ref[i + n_atoms] + Q[2][2]*r_ref[i + 2*n_atoms];

    r_ref[i] = x_tmp;
    r_ref[i + n_atoms] = y_tmp;
    r_ref[i + 2*n_atoms] = z_tmp;
  }
}