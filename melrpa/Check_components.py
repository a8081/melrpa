#!C:/Users/Antonio/Documents/TFM/melrpa/melrpa/env/Scripts/python.exe
import sys
import numpy as np
from matplotlib import pyplot as plt
# ./Check_components.py media/screenshots/components_npy/image0003.png.npy

param_path = sys.argv[1] if len(sys.argv) > 1 else "media/screenshots/components_npy/image1.png.npy"


recortes = np.load(param_path, allow_pickle=True)
for i in range(0,len(recortes)):
  print("Length: " + str(len(recortes)))
  if recortes[i].any():
    print("\nComponente nº",i+1)
    plt.imshow(recortes[i], interpolation='nearest')
    plt.show()

  else:
    print("componente vacío")