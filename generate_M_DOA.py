#%%
import numpy as np
from pathlib import Path
import utils.utils_function as utils_function
import os
import torch
#%%
M_dict={}
M_test_dict={}
batch_size=300
nb_channels_test=1000
nb_channel_train=100
nb_BS_antenna=64

T=3
L=16

path_init=Path.cwd()/'.saved_data'
file_name = f'Data/Measurement_matrix/L_{L}_T_{T}'  

save_dir = path_init/file_name
os.makedirs(save_dir, exist_ok=True)

# generate Measurement matrix for test 
M_list=[]
for t in range(T):
    phases = torch.rand(nb_BS_antenna, L) * (2 * torch.pi)
    m=torch.exp(1j* phases) # [LxA]
    M_list.append(m)

M_test=torch.cat((M_list),1)
if len(M_list)==1:
    M_tilde_test=M_test
else: 
    M_tilde_test=torch.block_diag(*M_list)

M_test=M_test.unsqueeze(0) #batched mx [*, TL, A]
M_tilde_test=M_tilde_test.unsqueeze(0) #batched mx [*, TL, AT]

M_test_dict['M_test']=M_test
M_test_dict['M_tilde_test']=M_tilde_test

#save data 
np.savez(save_dir/ 'test.npz', **M_test_dict) 


i=0

while i<batch_size:

    # generate Measurement matrix for test 
    M_list=[]
    for t in range(T):
        phases = torch.rand(nb_BS_antenna, L) * (2 * torch.pi)
        m=torch.exp(1j*phases) # [LxA]
        M_list.append(m)
    
    M_train=torch.cat((M_list),1)
    if len(M_list)==1:
        M_tilde_train=M_train
    else: 
        M_tilde_train=torch.block_diag(*M_list)

    M_train=M_train.unsqueeze(0)
    M_tilde_train=M_tilde_train.unsqueeze(0)

    M_dict['M_train']=M_train
    M_dict['M_tilde_train']=M_tilde_train

    #save data 
    np.savez(save_dir/ f"batch_{i}.npz", **M_dict) 

    i+=1


