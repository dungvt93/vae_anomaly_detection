import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys

print(sys.argv)
threshold = float(sys.argv[1])
f = h5py.File("file.h5")
img = np.array(f["a"])
print(len(np.where(img > 5)[0]))
exit()
img[img < threshold] = 0
plt.imshow(img)
plt.show()

