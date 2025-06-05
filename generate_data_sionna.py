
#%%
from pathlib import Path
import numpy as np 
import utils.utils_function as utils_function
from sionna.rt import Transmitter
import sionna
from pathlib import Path
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

# %%
batch_size=600
nb_channels=100
nb_channels_test=1000
nb_BS_antenna=64
f0=28e9 #HZ 
BS_position=[100, -90, 30]
scene,nominal_ant_positions,real_ant_positions =utils_function.init_scene(BS_position,f0)

#%%
scene.preview()

#%%
#Save nominal and real positions
path_init = Path.cwd()/'.saved_data'
antenna_pos={}
antenna_pos['nominal_position']=nominal_ant_positions
antenna_pos['real_position']=real_ant_positions
file_name = f"antenna_position.npz"  #Save data per batch
save_dir=path_init /'Data'/'datasionna'
os.makedirs(save_dir,exist_ok=True)
np.savez( save_dir/ file_name, **antenna_pos)

# %%

#Generate UE positions 
nb_UE_scene=10000000
cent_loc =[100,20,25]

dx=2
dy=2
num_positions=np.sqrt(nb_UE_scene)


x_arr = np.arange(cent_loc[0] - num_positions  / 2, cent_loc[0] + num_positions  / 2, dx)
y_arr = np.arange(cent_loc[1] - num_positions  / 2, cent_loc[1] + num_positions  / 2, dy)

xx, yy = np.meshgrid(x_arr, y_arr)
    
#generate grid positions
locs_grid = np.stack(( xx.flatten(),yy.flatten(), np.full(xx.size,cent_loc[2]))).T

# %%
#generate train data
i=0
batch_idx=0
while batch_idx < batch_size:
    train_data={}
    channel_train=np.array((nb_channels,nb_BS_antenna))

    # take nb_channels positions
    cur_loc=locs_grid[i*nb_channels:(i+1)*nb_channels,:]


    #Add UE 
    for idx in range(cur_loc.shape[0]):
        tx = Transmitter(name='tx'+f'_{idx}',
                   position=[cur_loc[idx][0],cur_loc[idx][1],cur_loc[idx][2]],
                      orientation=[0,0,0])
        scene.add(tx)

    paths=scene.compute_paths()
    paths.normalize_delays=False
    
    a, tau = paths.cir()
   
    frequencies=tf.cast(0,tf.float32) 

    
    h = sionna.channel.cir_to_ofdm_channel(frequencies, a, tau, normalize=False)
    h_temp = h.numpy().squeeze() #[nb_BS_antenna , nb_chan_train]

    channel_train=h_temp.T

    
    # test if any channel in the batch has 0 norm 
    if(0 in (np.linalg.norm(channel_train,2,axis=1))):
        print('channel will not be added to the current batch ')


    else:
        #Save data
        train_data['channel_train']=channel_train
        path_init = Path.cwd()/'.saved_data'
        file_name = f"batch_{batch_idx}.npz"  #Save data per batch
        np.savez(save_dir/ file_name, **train_data)
        batch_idx+=1
        
    #remove UE 
    for idx in range(cur_loc.shape[0]):
        scene.remove(f'tx_{idx}')         
    
    i+=1

# %%
#%%
# generate DOA
DoA  =np.zeros((1200,3))
DoA[:,1]= np.random.uniform(-np.pi,np.pi,1200)

# save azimuth angles 
doa={}
doa['DoA']=DoA
np.savez(path_init / f'Data/DoA', **doa)