import pickle
import matplotlib.pyplot as plt
import numpy as np

PATH = '/home/mateusz/Desktop/EQ_vis/lightning_logs/version_2/model_info.pkl'

data = pickle.load(open(PATH, 'rb'))
print(data['FUSE_METHOD'])
#%%