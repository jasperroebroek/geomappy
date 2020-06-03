"""
The plots here should create two similar plots next to each other.
Sizes and legends are not the same, but that does not matter. The last
plot should be a red square with a green patch inside.
"""
from matplotlib.colors import ListedColormap
import geomappy as mp
import matplotlib.pyplot as plt
import numpy as np

x = np.random.rand(10, 10, 3)
a = np.random.rand(10, 10)

f, ax = plt.subplots(ncols=2)
mp.plot_map(x, ax=ax[0])
ax[1].imshow(x)
plt.show()

f, ax = plt.subplots(ncols=2)
mp.plot_map(a, ax=ax[0])
ax[1].imshow(a)
plt.show()

f, ax = plt.subplots(ncols=2)
mp.plot_map(a, ax=ax[0], vmax=0.5)
ax[1].imshow(a, vmax=0.5)
plt.show()

f, ax = plt.subplots(ncols=2)
mp.plot_map(a, ax=ax[0], vmin=0.5)
ax[1].imshow(a, vmin=0.5)
plt.show()

f, ax = plt.subplots(ncols=2)
mp.plot_map(a, legend='legend', ax=ax[0])
ax[1].imshow(a)
plt.show()

f, ax = plt.subplots(ncols=2)
mp.plot_map(a, legend='colorbar', ax=ax[0], bins=[0.5])
ax[1].imshow(a > 0.5, cmap=ListedColormap(["Lightgrey", "Red"]))
plt.show()

f, ax = plt.subplots(ncols=2)
mp.plot_map(a > 0.5, legend='colorbar', ax=ax[0])
ax[1].imshow(a > 0.5, cmap=ListedColormap(["Lightgrey", "Red"]))
plt.show()

f, ax = plt.subplots(ncols=2)
mp.plot_map(a > 0.5, legend='legend', ax=ax[0])
ax[1].imshow(a > 0.5, cmap=ListedColormap(["Lightgrey", "Red"]))
plt.show()

b = np.ones((10, 10))
b[4,4] = 2
mp.plot_classified_map(b, [1,2], ['Red', "Green"])
plt.show()
