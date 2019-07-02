import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle

print(sys.argv)
threshold = float(sys.argv[1])
f = h5py.File("file.h5")
img = np.array(f["a"])
# print(len(np.where(img > 5)[0]))
#exit()
# img[img < threshold] = 0
plt.imshow(img)
plt.show()

#
# with open("dump1.txt", "rb") as fp:
#     has_proces = pickle.load(fp)
#
# with open("dump2.txt", "rb") as fp:
#     not_proces = pickle.load(fp)
#
# print(len(has_proces))
# print(len(not_proces))
# for i in not_proces:
#     if i not in has_proces:
#         print(i)
# print("*******************")
# for i in has_proces:
#     if i not in not_proces:
#         print(i)