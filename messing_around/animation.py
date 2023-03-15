import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation

hsp_models = np.load("hsp90_models.npy")
quats = np.load("quaternion_list.npy")

coord = hsp_models[0, 0]
rot_mat = Rotation.from_quat(quats[0]).as_matrix()
coord = np.matmul(rot_mat, coord)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection="3d")
ax.scatter(coord[0, :], coord[1, :], coord[2, :], marker="o", s=30)

plt.show()