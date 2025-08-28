#%%
from pathlib import Path
import os
import numpy as np
from utils.generate_data_functions import *

path_init = Path.cwd()/'.saved_data'
save_dir=path_init/'Data'
os.makedirs(save_dir,exist_ok=True)

#set the following parameters:
size_batches=100
nb_test_batches= 10
nb_train_batches=300
nb_BS_antennas=64
NOISE_VARIANCE=[2e-3, 2e-2]
NB_TIMEFRAMES=[1,3]
NB_RF_CHAINS=[16]

#Generate realistic channels using SionnaRT library
nb_batches=nb_train_batches+nb_test_batches
Dataset=generate_dataset(save_dir,nb_batches,size_batches,nb_BS_antennas)

#generate Directions of arrival, nb_DOa is equiv to nb of atoms in the dictionary that is a set of steering vectors from different DOAs
nb_DoA=1200
generate_DoA(save_dir,nb_DoA)

#Generate noisy channels with varying SNR for different noise variances and nbs of timeframes
#And Generate measurement matrix for different nbs of RF chains and nbs of timeframes
# noise_var_list=[2e-3, 6e-3, 2e-2, 6e-2, 2e-1]

for L in NB_RF_CHAINS:
    for T in NB_TIMEFRAMES:
        #generate measurmenet Matrix this matrix defines the nummber of measurments that is depending on the number of RF chains L and the number of timeframes T
        generate_M(save_dir,nb_BS_antennas,nb_train_batches,L,T)
        for noise_var in NOISE_VARIANCE:
            generate_channels_varying_SNR(save_dir,Dataset,size_batches,nb_test_batches,nb_train_batches,noise_var,T)



# %% testing data generation 
#load dataset:
# path_init=Path.cwd()/'.saved_data'
# data_path=f'Data/Dataset.npz'
# data=np.load(path_init/data_path)
# Dataset=data['Dataset']

#compute average SNR for a given noise variance:
# SNR_linear=np.linalg.norm(Dataset,2,axis=1)**2/ (nb_BS_antennas * noise_var)
# SNR_dB= 10*np.log10(SNR_linear)
# print("SNR_average=",SNR_dB.mean()) 