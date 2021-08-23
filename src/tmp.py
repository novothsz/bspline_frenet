from matplotlib.tri.triangulation import Triangulation
import numpy as np

n_angles = 36
n_radii = 8

radii = np.linspace(0.125, 1.0, n_radii)
angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)

x = np.append(0, (radii*np.cos(angles)).flatten())
y = np.append(0, (radii*np.sin(angles)).flatten())
z = np.sin(-x*y)


tri, args, kwargs = Triangulation.get_from_args_and_kwargs(x, y, z)
triangles = tri.get_masked_triangles()
xt = tri.x[triangles][..., np.newaxis]
yt = tri.y[triangles][..., np.newaxis]
zt = z[triangles][..., np.newaxis]

verts = np.concatenate((xt, yt, zt), axis=2)

print(verts)

import matplotlib.pyplot as plt
fig = plt.figure() 
ax = plt.axes(projection='3d')


x_obs = [verts_[:, 0] for verts_ in verts]
y_obs = [verts_[:, 1] for verts_ in verts]
z_obs = [verts_[:, 2] for verts_ in verts]
ax.scatter(x_obs, y_obs, z_obs, c=z_obs, cmap='viridis', linewidth=0.5);

