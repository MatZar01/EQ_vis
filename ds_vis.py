#%%
import numpy as np
import cv2
import matplotlib.pyplot as plt

# IDA-BD
image = cv2.imread('/home/mateusz/Desktop/EQ_visual/data/IDA-BD/PRJ-3563/images/AOI1-tile_1-3_post_disaster.png', -1)
post_label = cv2.imread('/home/mateusz/Desktop/EQ_visual/data/IDA-BD/PRJ-3563/masks/AOI1-tile_1-3_post_disaster.png', -1)

#%%
plt.imshow(post_label)
plt.show()

