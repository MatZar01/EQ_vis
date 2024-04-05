import pickle
import matplotlib.pyplot as plt
import numpy as np

PATH = '/home/mateusz/Desktop/EQ_vis/result_graphs/2024-4-5+12:30.pkl'

data = pickle.load(open(PATH, 'rb'))
print(data['FUSE_METHOD'])
#%%