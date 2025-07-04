#%%
import torch
from pathlib import Path
import numpy as np 
import utils.generate_channels_realizations as generate_channels_realizations
import utils.utils_function as utils_function
import os
from pathlib import Path

#%%
nb_channels_train=100
nb_channels_test= 1000
nb_train_batches=300

#%%
#get sionna dataset and normalize it 
path_init=Path.cwd()/'.saved_data'
data_path=f'Data/datasionna/Dataset.npz'
data=np.load(path_init/data_path)
Dataset=data['Dataset']
norm_channels=np.linalg.norm(Dataset,axis=1)
norm_max=np.max(norm_channels)
norm_min=np.min(norm_channels)
norm_factor=norm_max - norm_min

Dataset=Dataset/norm_factor
#%% 
permut=np.random.permutation(Dataset.shape[0])
Dataset=Dataset[permut]


#%% Data Parameter TO change w/ each generation
noise_var_list=[6e-4,2e-3, 6e-3, 2e-2, 6e-2, 2e-1]
T_list=[1,2,3]
for noise_var in noise_var_list:
    N_BS_antenna=Dataset.shape[1]
    SNR_linear=np.linalg.norm(Dataset,2,axis=1)**2/ (N_BS_antenna * noise_var)
    SNR_dB= 10*np.log10(SNR_linear)
    print("SNR_average=",SNR_dB.mean())
    for T in T_list: 
        #save test data
        test_data={}

        #Save data
        channel_test=Dataset[:nb_channels_test]

        #generate noisy channels,  the channel norm and sigma^2  
        h_noisy,noise_var_vect=generate_channels_realizations.generate_noisy_channels_varying_SNR(channel_test,noise_var,T)
        h=torch.tensor(channel_test,dtype=torch.complex128)

        test_data['h']          =h
        test_data['h_noisy']    =h_noisy
        test_data['sigma_2']    =noise_var_vect


        file_name = f"test_data.npz"  #Save data per batch
        save_dir=path_init / f'Data/{noise_var:.0e}/data_var_snr/T_{T}' 
        os.makedirs(save_dir,exist_ok=True)
        np.savez(save_dir/ file_name, **test_data) 

        batch=0
        stop=False

        while batch<nb_train_batches and stop==False:
            #get channels
            train_data={}
            
            file_name=f"batch_{batch}.npz" 

            channel_train = Dataset[(batch*nb_channels_train)+nb_channels_test:((batch+1)*nb_channels_train)+nb_channels_test]


            #If number of channels in the batch is not enough 
            if channel_train.shape[0]!=nb_channels_train:
                stop=True
            else:

                # generate noisy channels
                h_noisy,noise_var_vect=generate_channels_realizations.generate_noisy_channels_varying_SNR(channel_train,noise_var,T)
                h=torch.tensor(channel_train,dtype=torch.complex128)

                train_data['h']             =h#.repeat(1,T)
                train_data['h_noisy']       =h_noisy
                train_data['sigma_2']       =noise_var_vect
            

                #Save data
                np.savez(save_dir/ file_name, **train_data)
                batch=batch+1


'''# remove data with negative SNR values
index_to_remove=[]
for i in range(SNR_dB.shape[0]):
    if(SNR_dB[i]<0):
        print(SNR_dB[i])
        index_to_remove.append(i)
        

#new_dataset_ = np.delete(Dataset,index_to_remove,axis=0) 
#shuffle data 
#new_dataset = new_dataset_[np.random.permutation(new_dataset_.shape[0])]

#new_dataset = Dataset[np.random.permutation(Dataset.shape[0])]
'''

