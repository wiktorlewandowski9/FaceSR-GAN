import h5py
from PIL import Image

f = h5py.File('dataset_preparation/train.h5', 'r')

img = Image.fromarray(f['low_res']['0'][:])
img2 = Image.fromarray(f['high_res']['0'][:])

print(f['low_res']['0'].shape)
print(f['high_res']['0'].shape)

img.show()
img2.show()