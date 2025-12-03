
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# create an array of points: (0, 0) and (10, 10)
points = np.array([[0, 0], [10, 10]])

# Use slicing to pass in the x-values first, then the y-values
plt.plot(points[:, 0], points[:, 1])
plt.show()


# create an array of 10 random points
points = np.random.rand(10,2) * 10

# use the 'o' shortcut to only plot points, no line
plt.plot(points[:, 0], points[:, 1], 'o')
plt.show()