import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Extract the directories.

# files = os.listdir('/home/bourne/test_np/')
# ds = []
# for file in files:
#     file = os.path.join('/home/bourne/test_np/',file)
#     d = np.load(file)
#     # d = np.load('trajs_n_rsym_1.npy')
#     ds.append(d)

ds = []
d = np.load('obs.npy', allow_pickle=True)
ds.append(d)

from ipdb import set_trace
set_trace()

# Extract the trajectories from observation from transition.
all_xs =[]
all_ys =[]
all_zs =[]
for d in ds:
    xs =[]
    ys =[]
    zs = []
    for transitions in d:
        x =[]
        y =[]
        z = []
        for ob in transitions[0]:
            x.append(ob[0][0])
            y.append(ob[0][1])
            z.append(ob[0][2])
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        xs.append(x)
        ys.append(y)
        zs.append(z)
    all_xs.append(xs)
    all_ys.append(ys)
    all_zs.append(zs)

# plot each trajectories
for (xs,ys,zs) in zip(all_xs,all_ys,all_zs):
    colour = ['k','c','orange','gold','m','g','b','red']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for (x,y,z) in zip(xs,ys,zs):
        ax.plot(x, y, z, c=colour.pop())
    ax.scatter(0.695, 0.75, 0.4, c='r')

plt.show()



