#%%
import numpy as np
import cv2
import matplotlib.pyplot as plt
from imutils import paths

pts = list(paths.list_images('/home/mateusz/Desktop/EQ_vis/data/IDA_BD/IDA-BD/PRJ-3563/labels'))

p = pts[1]

im = cv2.imread(p, -1)

uq = np.unique(im)
im2 = np.where(im == 3, 1, 0)*255

#%%
i = 0
for p in pts:
    im = cv2.imread(p, -1)
    im2 = np.where(im==4, 255, 0)
    cv2.imwrite(f'/home/mateusz/Desktop/EQ_vis/data/IDA_BD/IDA-BD/PRJ-3563/lab4/{i}.png', im2)
    i += 1
#%%
lbls = []
pts = list(paths.list_images('/home/mateusz/Desktop/EQ_vis/DS/IDA-BD/i_B'))
for p in pts:
    lbls.append(int(p.split('_')[-1].split('.')[0]))
lbls = np.array(lbls)
uq2 = np.unique(lbls, return_counts=True)
#%%
import numpy as np
data = np.loadtxt('/home/mateusz/Downloads/lt/bazalosl_razem/dl_razem.txt', dtype=str)[:, 2]
data = np.array([l.split(',') for l in data], dtype=int)
#%%
import matplotlib.pyplot as plt
uq, bins = np.unique(data, return_counts=True)
zipped = sorted(zip(bins, uq))
Z = [x for _,x in zipped][::-1][:6]
