import numpy as np
import matplotlib.pyplot as plt

path_to_img = "imgs/img_state_1_1_0.txt"

img = np.loadtxt(path_to_img)

fig, ax = plt.subplots()

ax.imshow(img.T, origin="lower")

plt.show()
