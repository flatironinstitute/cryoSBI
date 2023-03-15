import numpy as np
import torch
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import json

from cryo_sbi.inference.models import build_models
from cryo_sbi import CryoEmSimulator
from cryo_sbi.inference import priors

global cryosbi
global hsp_models
global grid_x, grid_y 
global quats_index

cryosbi = CryoEmSimulator("image_params_snr01_128.json")
hsp_models = np.load("hsp90_models.npy")
quats = np.load("quaternion_list.npy")
np.random.shuffle(quats)
quats = torch.tensor(quats)

quats_index = torch.zeros((4, 5))
quats_index[:, 1:] = quats[:4]
quats_index[:, 0] = torch.arange(0, 4) * 6

#rot_mat = Rotation.from_quat(quats[0]).as_matrix()
#coord = np.matmul(rot_mat, coord)

n_pixels = 64
pixel_size = 3
pix_range = 64 * 3 / 2

grid_x, grid_y = np.meshgrid(np.linspace(-pix_range, pix_range, n_pixels), np.linspace(-pix_range, pix_range, n_pixels))

def init_anim():

    for i in range(len(axes)):
    
        ax = fig.add_subplot(4, 8, axes[i])

        if axes[i] >= 29:
            ax.set_xlim([0, 20])
            ax.set_xticks(range(0, 20, 4))
            ax.set_yticks([])

        else:
            ax.set_xticks([])
            ax.set_yticks([])

    return

def update(frame, scatter, plot, axes):
    
    quat = quats_index[frame % 4]
    #image = cryosbi._simulator_with_quat(quat).numpy().T
    image = cryosbi.simulator(torch.tensor(frame)).numpy().T
    #coord = np.copy(hsp_models[frame//4 * 6, 0])
    coord = np.copy(hsp_models[frame, 0])

    #rot_mat = Rotation.from_quat(quat[1:]).as_matrix()
    #coord = np.matmul(rot_mat, coord)
    
    scatter._offsets3d = (coord[0], coord[2], coord[1])
    ax = fig.add_subplot(4, 8, axes[frame])
    ax.plot(np.linspace(0, 20, 20), np.linspace(0, 1, 20))

    if axes[frame] >= 29:
        ax.set_yticks([])
        ax.set_xticks([])

    else:
        ax.set_xticks([])
        ax.set_yticks([])

    plot[0].remove()
    plot[0] = ax3d.contourf(grid_x, grid_y, image, 100, zdir="z", offset=-60, cmap="Greys_r")
    

coord = hsp_models[0, 0]
image = cryosbi.simulator(torch.tensor([0])).numpy().T

fig = plt.figure(figsize=(10, 5))

ax3d = fig.add_subplot(4, 8, (1, 28), projection="3d")
#axes = [5, 6, 7, 8, 13, 14, 15, 16, 21, 22, 23, 24, 29, 30, 31, 32]
axes = [5, 13, 21, 29, 6, 14, 22, 30, 7, 15, 23, 31, 8, 16, 24, 32]

# 01 02 03 04 05 06 07 08
# 09 10 11 12 13 14 15 16
# 17 18 19 20 21 22 23 24
# 25 26 27 28 29 30 31 32

#scatter = ax3d.scatter(coord[0], coord[2], coord[1], s=30)
scatter = ax3d.scatter([], [], [])
#plot = ax3d.contourf(grid_x, grid_y, image, 100, zdir="z", offset=-60, cmap="Greys_r")
plot = [ax3d.contourf([0, 0], [0, 0], [[0, 0], [0, 0]], 100, zdir="z", offset=-60, cmap="Greys_r")]
#plot = None

ax3d.set_xticks([])
ax3d.set_yticks([])
ax3d.set_zticks([])

ax3d.set_xlim((-pix_range, pix_range))
ax3d.set_ylim((-pix_range, pix_range))
ax3d.set_zlim((-60, 40))

anim = animation.FuncAnimation(
    fig,
    update,
    init_func=init_anim,
    frames=len(axes),
    fargs=(scatter, plot, axes),
    interval=1000,
    blit=False,
    repeat=False
)

writergif = animation.PillowWriter(fps=1)
anim.save(f"animation.gif", writer=writergif)

plt.show()

# train_config = json.load(open("resnet18_encoder.json"))
# estimator = build_models.build_npe_flow_model(train_config)
# estimator.load_state_dict(torch.load("resnet18_encoder_snr01.estimator"))
# estimator.eval()