import pickle
import matplotlib.pyplot as plt
import numpy as np
import yaml

PATH = '/home/mateusz/Desktop/EQ_vis/lightning_logs/version_4/model_info.pkl'

data = pickle.load(open(PATH, 'rb'))
yaml.dump(data, open(f'{PATH.replace("pkl", "yml")}', 'w'))
print(data['FUSE_METHOD'])
#%%