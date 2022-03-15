#%%
from data_collect import create_dataset
from cnn import create_tensorflow_dataset
import os

import matplotlib
import matplotlib.pyplot as plt

DATASET_DIR = os.path.join(os.getcwd(), "datasets")

#create_dataset()
create_tensorflow_dataset(DATASET_DIR)
# %%
